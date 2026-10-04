"""Joint I/O on the topics of the lab's Dynamixel driver: degrees, the bus order, raw signs."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.signals import Signals
from ..hardware.profile import CURRENT_RANGE, HardwareProfile

TOPICS = {
    "positions": "/joint_positions",  # Float64MultiArray [deg] from the start position, bus order
    "velocities": "/joint_velocities",  # Float64MultiArray [deg/s], bus order
    "torque": "/goal_torque",  # Float64MultiArray [N·m], bus order
    "simulated": "/robot_is_simulated",  # Bool, latched; True when a digital twin publishes
    "swap": "/vmc_swap",  # String: the name of the controller to swap to
}
"""The driver's topics, the flag a digital twin adds and the controller node's swap topic."""


class JointIO:
    """Converts between the driver's messages and the library's signals, through a profile."""

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile
        kt = np.array([m.constant for m in profile.commanded])
        self._limit = profile.to_bus(CURRENT_RANGE * kt)  # what the driver's int16 cast holds

    def measurement(self, t: float, positions: Any, velocities: Any) -> Signals:
        """Signals from the driver's angles [deg] and rates [deg/s]."""
        return Signals(
            t,
            motor_position=self.profile.angles_from_degrees(positions),
            motor_velocity=self.profile.angles_from_degrees(velocities),
        )

    def torque_message(self, torques: Any) -> list[float]:
        """Motor torques [N·m] as the driver takes them; non-finite ones are zero and none exceeds
        what its goal current can hold, so the driver's conversion never overflows."""
        tau = self.profile.bus_torques(torques)
        tau = np.where(np.isfinite(tau), tau, 0.0)
        return np.clip(tau, -self._limit, self._limit).tolist()

    def motor_torques(self, message: Any) -> np.ndarray:
        """Inverse of ``torque_message``: the motor torques [N·m] a message asks for."""
        signs = np.array([m.sign for m in self.profile.commanded])
        return signs * self.profile.from_bus(message)

    def angle_message(self, angles: Any) -> list[float]:
        """Motor angles or rates [rad, rad/s] as the driver publishes them [deg, deg/s]."""
        return self.profile.degrees(angles).tolist()
