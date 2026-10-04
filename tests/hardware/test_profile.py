"""Hardware profiles: unit conversions both ways, against the formulas of the ROS driver."""

import numpy as np
import pytest

from virtualmodelcontrol.hardware import KT, HardwareProfile, Motor

KT_XC = KT["XC330-T288"]
PROFILE = HardwareProfile(
    (Motor(1), Motor(2, sign=-1.0), Motor(3, mode="hold")), baudrate=2_000_000, rate=500.0
)


def driver_degrees(raw, start):
    """The driver's published angle: (raw / 4096 * 2 pi - start / 4096 * 2 pi) in degrees."""
    return np.degrees(
        np.asarray(raw) / 4096.0 * 2.0 * np.pi - np.asarray(start) / 4096.0 * 2.0 * np.pi
    )


def driver_current(torque, kt):
    """The driver's goal current: a C++ cast of torque / kt to int16 (truncation)."""
    return int(np.trunc(torque / kt))


def test_held_motors_are_not_commanded():
    assert PROFILE.ids == [1, 2] and [m.id for m in PROFILE.held] == [3]


def test_ticks_and_angles_both_ways():
    start = np.array([100, 5000])
    np.testing.assert_allclose(
        PROFILE.angles(start + np.array([4096, 1024]), start), [2 * np.pi, -np.pi / 2]
    )
    angles = np.array([0.3, -1.7])
    ticks = PROFILE.goal_ticks(angles, start)
    np.testing.assert_allclose(PROFILE.angles(ticks, start), angles, atol=np.pi / 4096)


def test_published_degrees_give_the_same_angles_as_ticks():
    start, raw = np.array([100, 5000]), np.array([737, 3311])
    degrees = driver_degrees(raw, start)
    np.testing.assert_allclose(PROFILE.angles_from_degrees(degrees), PROFILE.angles(raw, start))
    np.testing.assert_allclose(PROFILE.degrees(PROFILE.angles_from_degrees(degrees)), degrees)


def test_velocities_both_ways():
    raw = np.array([100, -40])
    rates = PROFILE.velocities(raw)
    np.testing.assert_allclose(np.degrees(rates), [100 * 0.229 * 6.0, 40 * 0.229 * 6.0])
    np.testing.assert_array_equal(PROFILE.goal_velocities(rates), raw)


def test_currents_and_torques_both_ways():
    torques = np.array([0.1, 0.25])
    currents = PROFILE.goal_currents(torques)
    assert currents.tolist() == [driver_current(0.1, KT_XC), driver_current(-0.25, KT_XC)]
    np.testing.assert_allclose(PROFILE.torques(currents), torques, atol=KT_XC)


def test_large_or_bad_torques_saturate_and_never_flip_sign():
    assert PROFILE.goal_currents([1e3, 1e3]).tolist() == [32767, -32767]
    assert PROFILE.goal_currents([np.nan, np.inf]).tolist() == [0, 0]
    limited = HardwareProfile((Motor(1, torque_limit=0.8), Motor(2, torque_limit=0.8)))
    assert limited.goal_currents([5.0, -5.0]).tolist() == [695, -695]  # 0.8 / 0.00115


def test_bus_order_maps_a_drivers_vector_both_ways():
    profile = HardwareProfile((Motor(5), Motor(4), Motor(7, sign=-1.0)), bus_order=(4, 5, 7))
    published = np.array([10.0, 20.0, 30.0])  # IDs 4, 5, 7
    np.testing.assert_allclose(profile.from_bus(published), [20.0, 10.0, 30.0])
    np.testing.assert_allclose(profile.to_bus(profile.from_bus(published)), published)
    np.testing.assert_allclose(profile.angles_from_degrees(published), np.radians([20, 10, -30]))
    np.testing.assert_allclose(profile.bus_torques([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])


def test_yaml_and_dict_round_trip(tmp_path):
    profile = PROFILE.replace(bus_order=(2, 1, 3), port="/dev/ttyACM0")
    assert HardwareProfile.from_dict(profile.to_dict()) == profile
    assert HardwareProfile.load(profile.save(tmp_path / "robot.yaml")) == profile


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        Motor(1, mode="current")
