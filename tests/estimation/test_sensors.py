"""The simulated sensors: their readings are the truth plus what they were told to add."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import Encoders, Imus, Inversion, LoadCell, Markers
from virtualmodelcontrol.robots import helyx

SITES = ["base", "seg1", "seg2", "tip"]


@pytest.fixture(scope="module")
def arm():
    return helyx.add_dynamics(helyx.arm("290-145-145"))


@pytest.fixture(scope="module")
def state():
    rng = np.random.default_rng(4)
    return rng.uniform(-0.02, 0.02, 9), rng.uniform(-0.1, 0.1, 9)


def test_encoders_without_noise_read_what_the_plant_reports_through_the_transmission(arm, state):
    q, v = state
    plant = vmc.sim.ModelPlant(arm, q0=q, v0=v)
    theta, rates = Encoders(arm, noise=0.0, rate_noise=0.0).read(q, v)
    measured = plant.read()
    np.testing.assert_allclose(theta, measured["motor_position"], atol=1e-12)
    np.testing.assert_allclose(rates, measured["motor_velocity"], atol=1e-12)


def test_encoders_add_noise_of_the_asked_size_and_a_constant_slack():
    robot = vmc.Mechanism("three", model=vmc.models.JointSpace(3))
    sensor = Encoders(robot, noise=0.02, rate_noise=0.1, slack=0.4, seed=3)
    q, v = np.array([0.1, 0.2, 0.3]), np.array([1.0, -1.0, 0.5])
    reads = [sensor.read(q, v) for _ in range(4000)]
    theta, rates = np.array([r[0] for r in reads]), np.array([r[1] for r in reads])
    assert np.all(np.abs(sensor.slack) <= 0.4) and np.ptp(sensor.slack) > 0.1
    np.testing.assert_allclose(theta.mean(axis=0), q + sensor.slack, atol=2e-3)  # slack stays
    np.testing.assert_allclose(theta.std(axis=0), 0.02, rtol=0.05)
    np.testing.assert_allclose(rates.mean(axis=0), v, atol=8e-3)
    np.testing.assert_allclose(rates.std(axis=0), 0.1, rtol=0.05)


def test_the_same_seed_gives_the_same_readings_and_another_does_not():
    robot = vmc.Mechanism("three", model=vmc.models.JointSpace(3))
    q, v = np.zeros(3), np.zeros(3)
    a = Encoders(robot, slack=0.1, seed=1).read(q, v)
    b = Encoders(robot, slack=0.1, seed=1).read(q, v)
    c = Encoders(robot, slack=0.1, seed=2).read(q, v)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.allclose(a[0], c[0])


def test_markers_without_noise_are_the_kinematics_and_a_fit_to_them_gives_the_state_back(
    arm, state
):
    q, _ = state
    at = [0.5, 0.75, 1.0]
    sensor = Markers(arm, at, noise=0.0)
    seen = sensor.read(q)
    kin = vmc.Kinematics(arm)
    np.testing.assert_allclose(seen, [kin.position(q, s) for s in at], atol=1e-14)
    np.testing.assert_allclose(Inversion(arm, at)(seen), q, atol=1e-6)  # the way an estimator reads


def test_markers_noise_is_the_asked_standard_deviation_on_every_coordinate(arm, state):
    q, _ = state
    sensor = Markers(arm, [1.0], noise=1e-3, seed=2)
    clean = vmc.Kinematics(arm).position(q, 1.0)
    errors = np.array([sensor.read(q)[0] - clean for _ in range(4000)])
    np.testing.assert_allclose(errors.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(errors.std(axis=0), 1e-3, rtol=0.05)


def test_imus_without_noise_read_the_angular_velocity_and_gravity_in_their_axes(arm, state):
    q, v = state
    gyro, acc = Imus(arm, SITES, bias=0.0, gyro_noise=0.0, acc_noise=0.0).read(q, v)
    kin = vmc.Kinematics(arm)
    for i, site in enumerate(SITES):
        R = kin.rotation(q, site)
        np.testing.assert_allclose(gyro[i], R.T @ (kin.angular_jacobian(q, site) @ v), atol=1e-12)
        np.testing.assert_allclose(acc[i], R.T @ [0.0, 0.0, 9.81], atol=1e-12)
        assert np.linalg.norm(acc[i]) == pytest.approx(9.81)  # whatever the axes


def test_the_gyro_bias_is_drawn_once_and_the_noise_each_time(arm, state):
    q, v = state
    sensor = Imus(arm, SITES, bias=0.02, gyro_noise=0.005, acc_noise=0.0, seed=5)
    clean, _ = Imus(arm, SITES, bias=0.0, gyro_noise=0.0, acc_noise=0.0).read(q, v)
    errors = np.array([sensor.read(q, v)[0] - clean for _ in range(2000)])
    np.testing.assert_allclose(errors.mean(axis=0), sensor.bias, atol=5e-4)  # the same every time
    np.testing.assert_allclose(errors.std(axis=0), 0.005, rtol=0.07)
    assert np.abs(sensor.bias).max() > 0.005  # and a bias there is


def test_a_load_cell_is_the_force_with_a_bias_and_noise():
    cell = LoadCell(noise=0.05, bias=0.2, seed=1)
    force = np.array([0.0, 0.0, 2.0])
    reads = np.array([cell.read(force) for _ in range(4000)])
    np.testing.assert_allclose(reads.mean(axis=0), force + 0.2, atol=4e-3)
    np.testing.assert_allclose(reads.std(axis=0), 0.05, rtol=0.05)
    assert reads.shape == (4000, 3) and LoadCell().read(2.0).shape == ()


def test_the_slack_goes_both_ways_and_the_accelerometer_noise_is_the_asked_size(arm, state):
    wide = Encoders(vmc.Mechanism("many", model=vmc.models.JointSpace(60)), slack=0.4, seed=1)
    assert wide.slack.min() < -0.2 and wide.slack.max() > 0.2
    q, v = state
    sensor = Imus(arm, SITES, bias=0.0, gyro_noise=0.0, acc_noise=0.05, seed=2)
    clean = Imus(arm, SITES, bias=0.0, gyro_noise=0.0, acc_noise=0.0).read(q, v)[1]
    errors = np.array([sensor.read(q, v)[1] - clean for _ in range(2000)])
    np.testing.assert_allclose(errors.mean(axis=0), 0.0, atol=6e-3)
    np.testing.assert_allclose(errors.std(axis=0), 0.05, rtol=0.07)
