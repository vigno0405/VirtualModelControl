"""Virtual Model Control of robots with fewer motors than coordinates.

The motors deliver B u, so a wanted torque τ is realized only up to its part E τ, the defect, with
E = I − B B⁺ the projector on the null space of Bᵀ. ``controller`` builds the naive and the frozen
controller, each with an optional passive or tank correction; ``DirectionalForce`` tracks a force
along a direction with the stiffness the motors can realize.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..core.signals import Signals
from ..dynamics import compile_dynamics, needs_energy
from ..models.kinematics import Kinematics
from .controller import VMCController
from .state import StateController

WIDTH = 1.0  # [W] smoothing of the passive correction's ramp
REGULARIZER = 1e-3  # [rad²/s²] keeps the correction finite at rest
TANK_WIDTH = 0.05  # [J] smoothing of the tank's gate
FORCE_REG = 1e-6  # keeps the stiffness law finite when the force does not respond to it
FORCE_MARGIN = 0.05  # how far from the loss of positive definiteness the stiffness stays
FORCE_RATE = 0.2  # [1/s] how fast the stiffness follows its closed form


def projector(B: ArrayLike) -> np.ndarray:
    """E = I − B B⁺, the projector on the null space of Bᵀ: the torques no motor can give."""
    B = np.asarray(B, dtype=float)
    return np.eye(B.shape[0]) - B @ np.linalg.solve(B.T @ B, B.T)


def defect(B: ArrayLike, J: ArrayLike, F: ArrayLike, torque: ArrayLike = 0.0) -> np.ndarray:
    """The torque defect E (Jᵀ F − τ) of a wrench F at the Jacobian J: what the motors cannot
    realize of it. ``torque`` is what the robot supplies by itself (a passive joint's spring)."""
    return projector(B) @ (np.asarray(J, dtype=float).T @ np.asarray(F, dtype=float) - torque)


def feasible(B: ArrayLike, J: ArrayLike) -> np.ndarray:
    """The wrenches the motors realize in full at the Jacobian J: the columns of an orthonormal
    basis of {F : (J E)ᵀ F = 0}, (p, p − rank (J E))."""
    from scipy.linalg import null_space  # SciPy loads when it is first used

    return null_space((np.asarray(J, dtype=float) @ projector(B)).T)


class Frozen:
    """Reads the frozen configuration from the motors, to pass to a ``StateController``.

    The motors give the part (I − E) q of the configuration and (I − E) v of the velocity; the
    rest of q is put at the rest value E q̄ of the ``Underactuated`` ``actuation``. With
    ``dynamics`` (the robot's, compiled) the rest is the passive coordinates' static balance
    instead: the point where E ∇V = 0, with V the robot's springs and gravity.
    """

    def __init__(self, actuation: Any, dynamics: Any = None) -> None:
        values = constants(actuation.params)
        self.B = np.reshape(np.asarray(values["B"]), (actuation.n, actuation.m), order="F")
        self.rest = np.ravel(np.asarray(values["q_rest"]))
        self.E = projector(self.B)
        eigenvalues, vectors = np.linalg.eigh(self.E)
        self.basis = vectors[:, eigenvalues > 0.5]  # an orthonormal basis of the range of E
        self.dynamics = dynamics

    def point(self, theta: ArrayLike) -> np.ndarray:
        """The frozen configuration for motor angles θ."""
        held = self.B @ np.linalg.solve(self.B.T @ self.B, np.asarray(theta, dtype=float))
        if self.dynamics is None:
            return held + self.E @ self.rest
        d = self.dynamics
        live, zero = d.live_values(), np.zeros(d.n_u)
        n = self.rest.size

        def balance(z: np.ndarray) -> np.ndarray:
            gradient = d.residual(held + self.basis @ z, np.zeros(n), np.zeros(n), zero, live, 0.0)
            return self.basis.T @ np.asarray(gradient).ravel()

        from scipy.optimize import root  # SciPy loads when it is first used

        solution = root(balance, self.basis.T @ self.rest)
        z = solution.x if solution.success else self.basis.T @ self.rest
        return held + self.basis @ z

    def __call__(self, meas: Signals) -> tuple[np.ndarray, np.ndarray]:
        """(π(q), (I − E) v) of a measurement's motors."""
        rates = np.linalg.solve(self.B.T @ self.B, meas["motor_velocity"])
        return self.point(meas["motor_position"]), self.B @ rates


def _softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


class Passivation:
    """An output stage that keeps the motors from injecting more power than the robot dissipates.

    With θ̇ the motor rates, u the command and V̇ the power the robot's own dampers take (read from
    its state, so the measurement must hold ``q`` and ``v``), it subtracts α θ̇, with
    α = w softplus((θ̇ᵀu − V̇) / w) / (θ̇ᵀθ̇ + ε): the smallest change that leaves θ̇ᵀu ≤ V̇ (up to
    the smoothing ``width`` w [W] and the ``regularizer`` ε). It is passive.

    With ``tank`` (a level T [J]) it is the energy tank's version: the correction is scaled by
    softplus(−T / w_T) / ln 2, which is 1 at T = 0 and falls to 0 as T grows, and T integrates
    V̇ − θ̇ᵀu (what the dampers took, less what the motors injected). The command passes through
    unchanged while the tank holds energy. ``reset`` refills it. ``level``, ``alpha``, ``gate``
    and ``dissipation`` (V̇) hold the last step's.
    """

    def __init__(
        self,
        dynamics: Any,
        tank: float | None = None,
        width: float = WIDTH,
        regularizer: float = REGULARIZER,
        tank_width: float = TANK_WIDTH,
    ) -> None:
        self.dynamics, self.tank = dynamics, tank
        self.width, self.regularizer, self.tank_width = width, regularizer, tank_width
        self.reset()

    def reset(self) -> None:
        """Refill the tank and forget the last step."""
        self.level = self.tank
        self.alpha = self.gate = self.dissipation = 0.0
        self._t: float | None = None
        self._rate = 0.0

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """The passive (or tank) command for ``u``."""
        d = self.dynamics
        if self.level is not None and self._t is not None:
            self.level += (meas.t - self._t) * self._rate
        rate = meas["motor_velocity"]
        _, power = needs_energy(d, "the passivity correction")
        out = power(meas["q"], meas["v"], np.zeros(d.n_u), d.live_values(), meas.t)
        self.dissipation = -float(out[1])
        excess = float(rate @ u) - self.dissipation
        w = self.width
        self.alpha = w * _softplus(excess / w) / (float(rate @ rate) + self.regularizer)
        self.gate = 1.0
        if self.level is not None:
            self.gate = _softplus(-self.level / self.tank_width) / np.log(2.0)
        command = u - self.gate * self.alpha * rate
        self._t, self._rate = meas.t, self.dissipation - float(rate @ command)
        return command


def direction_gain(
    c0: float,
    cv: float,
    force: float,
    K: ArrayLike,
    direction: ArrayLike,
    reg: float = FORCE_REG,
    margin: float = FORCE_MARGIN,
) -> float:
    """The scalar s that makes the realized force c0 + s cv along ``direction`` closest to
    ``force``, with K + s n nᵀ positive definite: s = max(cv (force − c0) / (cv² + reg), floor),
    the floor being (1 − margin) times the s at which K + s n nᵀ loses it."""
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    floor = (1.0 - margin) * (-1.0 / float(n @ np.linalg.solve(np.asarray(K, dtype=float), n)))
    return max(cv * (force - c0) / (cv * cv + reg), floor)


class DirectionalForce:
    """Tracks a force along ``direction`` at ``site`` with one scalar of the stiffness.

    ``param`` names a live (3, 3) stiffness Param of the controller, K. The law keeps
    K' = K + s n nᵀ, which is symmetric for every s, and moves s towards the value that makes the
    force the motors realize (naive: (I − E) τ) along n equal ``force`` [N], the closed form of
    ``direction_gain`` with the least ``reg`` and ``margin`` that keep K' positive definite. A
    ``step`` reads the state (q, v) the law is evaluated at, and moves s by a fraction ``rate`` dt
    of the way; ``s`` is its value, ``reading`` the realized force along n at the stiffness K'.
    """

    def __init__(
        self,
        controller: Any,
        site: Any,
        param: str,
        direction: ArrayLike,
        force: float = 0.0,
        *,
        rate: float = FORCE_RATE,
        reg: float = FORCE_REG,
        margin: float = FORCE_MARGIN,
    ) -> None:
        compiled = controller.compiled
        shape = compiled.params[param].shape
        if shape != (3, 3):
            raise ValueError(f"{param!r} must be a (3, 3) stiffness, not {shape}")
        self.param, self.force, self.rate, self.reg, self.margin = param, force, rate, reg, margin
        self.n = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.K = np.reshape(controller.live_params()[param], (3, 3))
        self._where = compiled.live_slices()[param]
        self._kin = Kinematics(compiled.system.robot)
        self._site = site
        actuation = compiled.system.actuation
        B = getattr(actuation, "params", {}).get("B")
        self._E = projector(np.reshape(B.value, (actuation.n, actuation.m), order="F")) if B else 0
        self.s = 0.0
        self.reading = 0.0

    def _pieces(self, controller: Any, q: np.ndarray, v: np.ndarray) -> tuple[float, float]:
        """(c0, cv): the realized force along n at s = 0, and its slope in s."""
        x = controller.inputs()
        compiled, p = controller.compiled, controller.params.copy()
        tau = []
        for s in (0.0, 1.0):
            p[self._where] = np.ravel(self.K + s * np.outer(self.n, self.n), order="F")
            tau.append(np.array(compiled.tau(q, v, controller.z, p, x[-1])).ravel())
        realized = [t - self._E @ t for t in tau]
        g = self._kin.jacobian(q, self._site).T @ self.n
        c0 = float(g @ realized[0]) / float(g @ g)
        return c0, float(g @ (realized[1] - realized[0])) / float(g @ g)

    def step(self, controller: Any, q: ArrayLike, v: ArrayLike, dt: float) -> float:
        """One step on ``controller`` (or a ``Tank`` around it); returns the energy jump [J]."""
        q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        c0, cv = self._pieces(controller, q, v)
        goal = direction_gain(c0, cv, self.force, self.K, self.n, self.reg, self.margin)
        floor = (1.0 - self.margin) * (-1.0 / float(self.n @ np.linalg.solve(self.K, self.n)))
        self.s = max(self.s + dt * self.rate * (goal - self.s), floor)
        self.reading = c0 + self.s * cv
        K = self.K + self.s * np.outer(self.n, self.n)
        return float(controller.set({self.param: np.ravel(K, order="F")}))


def controller(
    compiled: Any,
    base: str = "naive",
    correction: str | None = None,
    *,
    gravity: bool = False,
    tank: float = 1.0,
    **stage: float,
) -> VMCController:
    """A controller for an underactuated robot, by flags.

    ``base="naive"`` renders the wrench as it is (B⁺ τ at the measured state, which must hold
    ``q`` and ``v``); ``"frozen"`` renders it at the frozen configuration the motors give, so it
    needs the motors only. With ``gravity`` the frozen configuration also balances the robot's
    own gravity. ``correction`` is ``None``, ``"passive"`` or ``"tank"`` (starting at ``tank``
    [J]), a ``Passivation`` stage, which needs ``q`` and ``v`` in the measurement; ``stage`` are
    its ``width``, ``regularizer`` and ``tank_width``.
    """
    if base not in ("naive", "frozen"):
        raise ValueError(f"base must be 'naive' or 'frozen', not {base!r}")
    if correction not in (None, "passive", "tank"):
        raise ValueError(f"correction must be None, 'passive' or 'tank', not {correction!r}")
    system = compiled.system
    dynamics = None
    if correction or gravity:
        dynamics = compile_dynamics(system.robot, actuation=system.actuation)
    level = tank if correction == "tank" else None
    stages = [] if correction is None else [Passivation(dynamics, level, **stage)]
    if base == "naive":
        return StateController(compiled, stages)
    if gravity:
        return StateController(compiled, stages, read=Frozen(system.actuation, dynamics))
    return VMCController(compiled, stages)


__all__ = [
    "DirectionalForce",
    "Frozen",
    "Passivation",
    "controller",
    "defect",
    "direction_gain",
    "feasible",
    "projector",
]
