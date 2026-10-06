"""Collocation: the whole motion at once, with the closed loop as constraints."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..compiler import compile as compile_system
from ..control.blending import blend_weight
from ..core.space import Euclidean, Product
from ..system import VirtualMechanismSystem
from .builder import Builder
from .trajectory import Trajectory


def _flat(space: Any) -> bool:
    """True for Euclidean spaces and their products, where q + v·dt stays on the space."""
    if isinstance(space, Product):
        return all(_flat(s) for s in space.spaces)
    return isinstance(space, Euclidean)


def _snapshot(initial: Any) -> tuple[Any, np.ndarray]:
    """The controller in place as it is now: its compiled law and its live Param values."""
    if isinstance(initial, VirtualMechanismSystem):
        compiled = compile_system(initial)
        return compiled, np.array(compiled.live_values(), dtype=float)
    if hasattr(initial, "compiled") and hasattr(initial, "params"):  # a running controller
        return initial.compiled, np.array(initial.params, dtype=float)
    raise TypeError(
        "initial is a VirtualMechanismSystem (robot and controller) or a running controller, "
        f"not a {type(initial).__name__}"
    )


SCHEMES = ("trapezoid", "hermite-simpson")


class Collocation:
    """The closed loop's motion from ``q0`` over ``horizon`` [s] at ``nodes`` equally spaced nodes.

    q, v and a are unknowns at every node, tied by the robot's dynamics (without inverting the
    mass matrix) and the ``scheme`` that integrates them: the trapezoid rule (second order), or
    Hermite-Simpson (fourth order), which also has the acceleration at the middle of every
    interval as an unknown. The motion starts at rest (or at ``v0``).

    ``initial`` is the controller in place, as it is now: a ``VirtualMechanismSystem`` or a
    running ``VMCController`` on the same robot. Its torques fade out as the new ones fade in over
    ``transition`` [s], with the blend of ``SwapController``, and the model starts at rest where
    the robot is, carrying the force that balances it under ``initial``. Both controllers' time
    counts from the first node. ``scales`` are the typical sizes of q, v and a (the solver sees
    values divided by them).
    """

    name = "collocation"
    motion = True

    def __init__(
        self,
        q0: ArrayLike,
        horizon: float,
        nodes: int,
        *,
        v0: ArrayLike | None = None,
        initial: Any = None,
        transition: float = 0.0,
        scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
        scheme: str = "trapezoid",
    ) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"scheme takes one of {SCHEMES}, got {scheme!r}")
        if nodes < 2:
            raise ValueError("a collocation needs at least 2 nodes")
        if horizon <= 0.0:
            raise ValueError("the horizon must be positive")
        if transition < 0.0:
            raise ValueError("the transition cannot be negative")
        if len(scales) != 3 or min(scales) <= 0.0:
            raise ValueError("scales are three positive numbers: the sizes of q, v and a")
        self.q0 = np.asarray(q0, dtype=float).ravel()
        self.v0 = None if v0 is None else np.asarray(v0, dtype=float).ravel()
        self.horizon, self.nodes = float(horizon), int(nodes)
        self.transition = float(transition)
        self.scales, self.scheme = tuple(float(s) for s in scales), scheme
        self._hold = None if initial is None else _snapshot(initial)

    def coordinates(self) -> tuple[Any, ...]:
        """No task coordinates."""
        return ()

    def build(self, builder: Builder) -> None:
        """Add the unknowns, the dynamics and integration constraints, and the trajectory."""
        system = builder.system
        space = system.robot.model.space
        if not _flat(space):
            raise NotImplementedError(
                "collocation needs a Euclidean configuration space (or a product of them)"
            )
        nq, nv, n = space.nq, space.nv, self.nodes
        v0 = np.zeros(nv) if self.v0 is None else self.v0
        if self.q0.size != nq or v0.size != nv:
            raise ValueError(f"q0 and v0 need {nq} and {nv} entries, got {self.q0.size}, {v0.size}")
        dt = self.horizon / (n - 1)
        t = np.arange(n) * dt
        sq, sv, sa = self.scales
        inf = np.inf
        v_guess = np.zeros(n * nv)
        v_guess[:nv] = v0
        Q = builder.variables.add("q", n * nq, -inf, inf, np.tile(self.q0, n), sq)
        V = builder.variables.add("v", n * nv, -inf, inf, v_guess, sv)
        A = builder.variables.add("a", n * nv, -inf, inf, np.zeros(n * nv), sa)
        qs = [Q[k * nq : (k + 1) * nq] for k in range(n)]
        vs = [V[k * nv : (k + 1) * nv] for k in range(n)]
        accel = [A[k * nv : (k + 1) * nv] for k in range(n)]

        compiled, dynamics, no_z = builder.compiled, builder.dynamics, ca.DM.zeros(0, 1)
        p_law = builder.pack(compiled.params, compiled.live)
        p_dyn = builder.pack(dynamics.params, dynamics.live)
        hold = self._check_hold(builder)
        blend = np.array([blend_weight(float(tk), self.transition) if hold else 1.0 for tk in t])

        bias = ca.DM.zeros(nv)
        if hold is not None:
            hold_law, p_hold = hold
            rest = ca.DM.zeros(nv)
            u_rest = hold_law(ca.DM(self.q0), rest, no_z, p_hold, 0.0)[0]
            bias = dynamics.residual(ca.DM(self.q0), rest, rest, u_rest, p_dyn, 0.0)

        def law(q: Any, v: Any, tk: float, w: float) -> Any:
            """The motor torques at (q, v, tk), blended with the controller in place."""
            u = compiled.law(q, v, no_z, p_law, tk)[0]
            return u if w >= 1.0 else w * u + (1.0 - w) * hold_law(q, v, no_z, p_hold, tk)[0]

        us, defects = [], []
        for k in range(n):
            us.append(law(qs[k], vs[k], float(t[k]), float(blend[k])))
            defects.append(
                dynamics.residual(qs[k], vs[k], accel[k], us[k], p_dyn, float(t[k])) - bias
            )
        if self.scheme == "trapezoid":
            positions = [
                space.difference(qs[k + 1], qs[k]) - 0.5 * dt * (vs[k] + vs[k + 1])
                for k in range(n - 1)
            ]
            velocities = [
                vs[k + 1] - vs[k] - 0.5 * dt * (accel[k] + accel[k + 1]) for k in range(n - 1)
            ]
        else:  # Hermite-Simpson: the middle of an interval from the cubic through its ends
            Am = builder.variables.add("am", (n - 1) * nv, -inf, inf, np.zeros((n - 1) * nv), sa)
            positions, velocities, middles = [], [], []
            for k in range(n - 1):
                a_c, tc = Am[k * nv : (k + 1) * nv], float(t[k] + 0.5 * dt)
                q_c = 0.5 * (qs[k] + qs[k + 1]) + dt / 8 * (vs[k] - vs[k + 1])
                v_c = 0.5 * (vs[k] + vs[k + 1]) + dt / 8 * (accel[k] - accel[k + 1])
                w_c = blend_weight(tc, self.transition) if hold else 1.0
                u_c = law(q_c, v_c, tc, w_c)
                middles.append(dynamics.residual(q_c, v_c, a_c, u_c, p_dyn, tc) - bias)
                positions.append(
                    space.difference(qs[k + 1], qs[k]) - dt / 6 * (vs[k] + 4 * v_c + vs[k + 1])
                )
                velocities.append(vs[k + 1] - vs[k] - dt / 6 * (accel[k] + 4 * a_c + accel[k + 1]))
        builder.constrain("start", ca.vertcat(qs[0] - self.q0, vs[0] - v0), 0.0, 0.0)
        builder.constrain("dynamics", ca.vertcat(*defects), 0.0, 0.0)
        if self.scheme != "trapezoid":
            builder.constrain("middles", ca.vertcat(*middles), 0.0, 0.0)
        builder.constrain("position", ca.vertcat(*positions), 0.0, 0.0)
        builder.constrain("velocity", ca.vertcat(*velocities), 0.0, 0.0)
        builder.output("u", ca.horzcat(*us))
        builder.trajectory = Trajectory(
            t=t,
            dt=dt,
            q=qs,
            v=vs,
            a=accel,
            u=us,
            blend=blend,
            shapes={"q": (n, nq), "v": (n, nv), "a": (n, nv)},
            evaluate=builder.evaluate,
        )

    def _check_hold(self, builder: Builder) -> tuple[Any, Any] | None:
        """The initial controller's law and its (fixed) live Params, or None."""
        if self._hold is None:
            return None
        compiled, values = self._hold
        if compiled.system.robot is not builder.system.robot:
            raise ValueError("the initial controller must control the same robot")
        if compiled.z0.size:
            raise NotImplementedError("controllers with virtual states cannot be planned yet")
        return compiled.law, ca.DM(values)
