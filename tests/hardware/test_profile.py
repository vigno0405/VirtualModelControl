"""Hardware profiles: a driver's degrees, order and signs to the library's SI and back."""

import numpy as np

from virtualmodelcontrol.hardware import HardwareProfile, Motor

PROFILE = HardwareProfile((Motor(1), Motor(2, sign=-1.0), Motor(3, hold=True)), rate=500.0)


def test_held_motors_are_not_commanded():
    assert PROFILE.ids == [1, 2] and [m.id for m in PROFILE.held] == [3]


def test_degrees_and_radians_both_ways_with_the_signs():
    degrees = np.array([90.0, 45.0])
    angles = PROFILE.angles_from_degrees(degrees)
    np.testing.assert_allclose(angles, [np.pi / 2, -np.pi / 4])
    np.testing.assert_allclose(PROFILE.degrees(angles), degrees)


def test_torques_take_the_signs_of_the_motors():
    np.testing.assert_allclose(PROFILE.bus_torques([1.0, 2.0]), [1.0, -2.0])


def test_bus_order_maps_a_drivers_vector_both_ways():
    profile = HardwareProfile((Motor(5), Motor(4), Motor(7, sign=-1.0)), bus_order=(4, 5, 7))
    published = np.array([10.0, 20.0, 30.0])  # IDs 4, 5, 7
    np.testing.assert_allclose(profile.from_bus(published), [20.0, 10.0, 30.0])
    np.testing.assert_allclose(profile.to_bus(profile.from_bus(published)), published)
    np.testing.assert_allclose(profile.angles_from_degrees(published), np.radians([20, 10, -30]))
    np.testing.assert_allclose(profile.bus_torques([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])


def test_yaml_and_dict_round_trip(tmp_path):
    profile = PROFILE.replace(bus_order=(2, 1, 3), rate=250.0)
    assert HardwareProfile.from_dict(profile.to_dict()) == profile
    assert HardwareProfile.load(profile.save(tmp_path / "robot.yaml")) == profile
