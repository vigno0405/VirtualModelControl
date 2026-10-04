"""The driver's messages and the library's signals, both ways (no ROS needed)."""

import numpy as np

from virtualmodelcontrol.hardware import HardwareProfile, Motor
from virtualmodelcontrol.ros import JointIO

PROFILE = HardwareProfile((Motor(5), Motor(4, sign=-1.0)), bus_order=(4, 5))


def test_measurements_from_the_drivers_degrees():
    meas = JointIO(PROFILE).measurement(1.5, [90.0, 30.0], [-180.0, 0.0])  # IDs 4, 5
    assert meas.t == 1.5
    np.testing.assert_allclose(meas["motor_position"], np.radians([30.0, -90.0]))
    np.testing.assert_allclose(meas["motor_velocity"], np.radians([0.0, 180.0]))


def test_torque_messages_both_ways_and_within_the_drivers_range():
    io = JointIO(PROFILE)
    message = io.torque_message([0.2, 0.5])
    assert message == [-0.5, 0.2]
    np.testing.assert_allclose(io.motor_torques(message), [0.2, 0.5])
    limit = 32767 * 0.00115
    np.testing.assert_allclose(io.torque_message([1e3, np.nan]), [0.0, limit])


def test_angles_published_as_the_driver_does():
    io = JointIO(PROFILE)
    np.testing.assert_allclose(io.angle_message(np.radians([10.0, 20.0])), [-20.0, 10.0])
