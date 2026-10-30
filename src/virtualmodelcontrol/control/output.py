"""Output stages: optional corrections applied, in order, to the torques before they are sent.

None of them is on by default. Each takes the controller's motor torques u [N·m] and the
measurement and returns new torques. Motor angles follow the library convention (θ > 0 pulls).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.registry import register
from ..core.signals import Signals
from ..dynamics import needs_energy


@register("output", "friction_compensation")
@dataclass
class FrictionCompensation:
    """Static-friction feed-forward per motor (Stribeck): u + τ_max exp(−(θ̇/v)²) sign(u).

    ``max_torque`` [N·m] (scalar or per motor), ``velocity`` v [rad/s].
    """

    max_torque: ArrayLike
    velocity: float

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Add the friction torque in the direction of the command."""
        v = meas["motor_velocity"]
        return u + np.asarray(self.max_torque) * np.exp(-((v / self.velocity) ** 2)) * np.sign(u)


@register("output", "static_friction_compensation")
@dataclass
class StaticFrictionCompensation:
    """The kinetic friction torque added in the direction of the command, smooth in the command:
    u + fraction · Fc · u / √(u² + w²).

    ``kinetic`` Fc [N·m] (scalar or per motor), ``width`` w [N·m], ``fraction`` of Fc that is
    compensated. Against an actuator with ``StaticFriction(F, Fc)`` it delivers the command once
    the actuator moves and narrows the dead band from F to F − Fc. It adds energy (up to Fc |θ̇|),
    so nothing turns it on but you. ``symbolic`` is the same map for a plan (``Problem(output=)``).
    """

    kinetic: ArrayLike
    width: float = 0.01
    fraction: float = 1.0

    @classmethod
    def of(cls, friction: Any, fraction: float = 1.0) -> StaticFrictionCompensation:
        """The compensation of a ``StaticFriction``, at its current kinetic torque and width."""
        return cls(
            friction.params["kinetic"].value, float(friction.params["width"].value), fraction
        )

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Add the compensation."""
        a = np.sqrt(u * u + self.width**2)
        return u + self.fraction * np.asarray(self.kinetic) * u / a

    def symbolic(self, u: Any) -> Any:
        """The same map for CasADi expressions."""
        a = ca.sqrt(u * u + self.width**2)
        return u + self.fraction * ca.DM(np.asarray(self.kinetic, dtype=float)) * u / a


@register("output", "pretension")
@dataclass
class Pretension:
    """Linear pretension u − W ∘ θ; with ``threshold`` [rad], only where θ < −threshold.

    ``weights`` W [N·m/rad], scalar or per motor. Without a threshold it acts at every angle; with
    one, it acts only on tendons released past the threshold (a soft stop).
    """

    weights: ArrayLike
    threshold: float | None = None

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Subtract the pretension torque."""
        theta = meas["motor_position"]
        tau = np.asarray(self.weights) * theta
        if self.threshold is not None:
            tau = np.where(theta < -self.threshold, tau, 0.0)
        return u - tau


@register("output", "torque_offset")
@dataclass
class TorqueOffset:
    """A constant torque added to every command [N·m], scalar or per motor."""

    offset: ArrayLike

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Add the offset."""
        return u + np.asarray(self.offset)


@register("output", "torque_limit")
@dataclass
class TorqueLimit:
    """Clips every command to ±limit [N·m] (scalar or per motor)."""

    limit: ArrayLike

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Clip."""
        lim = np.asarray(self.limit)
        return np.clip(u, -lim, lim)


@register("output", "tension_floor")
@dataclass
class TensionFloor:
    """A floor under the torques of tendons that only pull, so that none is told to push.

    A motor commanded to push a tendon unwinds without end. The torques of each group of ``group``
    consecutive motors (the tendons of a segment) are lifted together, by as much as the lowest one
    is short of ``floor`` [N·m]. When the tendons of a segment are spaced evenly around it, equal
    torques on all of them add only an axial squeeze, so the bend that the command makes is the
    same. The corner is rounded over ``width`` [N·m], so a tendon is never below the floor and the
    lift is at most (ln ``group`` + ln 2) ``width`` above the exact one. ``symbolic`` is the same
    smooth map for a plan (``Problem(output=)``). With ``floor`` 0, no tendon pushes and nothing
    more: the needed tendons pull and the others do nothing.
    """

    floor: float
    width: float = 0.001
    group: int = 3

    def __post_init__(self) -> None:
        if not self.width > 0.0:
            raise ValueError(
                f"width must be positive, got {self.width} (a small width, such as 1e-4, makes "
                "the floor nearly exact)"
            )
        if self.group < 1:
            raise ValueError(f"group must be at least 1, got {self.group}")

    def _lift(self, u: list[Any], fmin: Any, exp: Any, log: Any, fmax: Any, fabs: Any) -> Any:
        """The lift of one group's torques ``u``: a soft max(0, floor - min(u))."""
        w, lowest = self.width, u[0]
        for x in u[1:]:
            lowest = fmin(lowest, x)
        soft_min = lowest - w * log(sum(exp(-(x - lowest) / w) for x in u))
        short = self.floor - soft_min
        return fmax(0.0, short) + w * log(1.0 + exp(-fabs(short) / w))

    def _check(self, n: int) -> None:
        if n % self.group:
            raise ValueError(f"{n} motor torques are not groups of {self.group} tendons")

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Lift each group to the floor."""
        u = np.array(u, dtype=float)
        self._check(u.size)
        for k in range(0, u.size, self.group):
            group = u[k : k + self.group]
            group += self._lift(list(group), np.minimum, np.exp, np.log, np.maximum, np.abs)
        return u

    def symbolic(self, u: Any) -> Any:
        """The same map for CasADi expressions."""
        n = int(u.shape[0])
        self._check(n)
        parts = []
        for k in range(0, n, self.group):
            group = [u[k + i] for i in range(self.group)]
            lift = self._lift(group, ca.fmin, ca.exp, ca.log, ca.fmax, ca.fabs)
            parts.append(u[k : k + self.group] + lift)
        return ca.vertcat(*parts)


@dataclass
class PlantState:
    """What a stage that corrects for the robot sees in a plan or a rollout, as CasADi expressions:
    the motor ``rates`` [rad/s] and the power [W] that the robot's own dampers take
    (``dissipation``, at least 0)."""

    rates: Any
    dissipation: Any


def plant_state(dynamics: Any, q: Any, v: Any, p: Any, t: Any) -> PlantState:
    """The plant's ``PlantState`` at (q, v), with ``p`` the packed live Params of ``dynamics``."""
    _, power = needs_energy(dynamics, "a correction that depends on the plant")
    rates = dynamics.motors(q, v, p)[1]
    return PlantState(rates, -power(q, v, ca.DM.zeros(dynamics.n_u), p, t)[1])


def apply_symbolic(stages: Iterable[Any], u: Any, plant: Callable[[], PlantState]) -> Any:
    """Run the stages' ``symbolic`` forms in order. A stage with ``needs_plant`` set takes the
    ``PlantState`` as a second argument (``plant()`` makes it, once); the others take only ``u``."""
    state = None
    for stage in stages:
        if getattr(stage, "needs_plant", False):
            state = plant() if state is None else state
            u = stage.symbolic(u, state)
        else:
            u = stage.symbolic(u)
    return u


def apply(stages: list[Any], u: np.ndarray, meas: Signals) -> np.ndarray:
    """Run the stages in order."""
    for stage in stages:
        u = stage(u, meas)
    return u
