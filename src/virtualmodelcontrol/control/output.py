"""Output stages: optional corrections applied, in order, to the torques before they are sent.

None of them is on by default. Each takes the controller's motor torques u [N·m] and the
measurement and returns new torques. Motor angles follow the library convention (θ > 0 pulls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.signals import Signals


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


@dataclass
class TorqueOffset:
    """A constant torque added to every command [N·m], scalar or per motor."""

    offset: ArrayLike

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """Add the offset."""
        return u + np.asarray(self.offset)


@dataclass
class EfficiencyCorrection:
    """Divides by the transmission efficiency η, so the robot receives the computed torque."""

    efficiency: ArrayLike

    def __call__(self, u: np.ndarray, meas: Signals) -> np.ndarray:
        """u / η."""
        return u / np.asarray(self.efficiency)


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
