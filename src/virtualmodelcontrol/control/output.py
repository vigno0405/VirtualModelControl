"""Output stages: optional corrections applied, in order, to the torques before they are sent.

None of them is on by default. Each takes the controller's motor torques u [N·m] and the
measurement and returns new torques. Motor angles follow the library convention (θ > 0 pulls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.registry import register
from ..core.signals import Signals


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


def apply(stages: list[Any], u: np.ndarray, meas: Signals) -> np.ndarray:
    """Run the stages in order."""
    for stage in stages:
        u = stage(u, meas)
    return u
