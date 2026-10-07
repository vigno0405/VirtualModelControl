"""Sensors that see combinations of coordinates, and robots with fewer motors than coordinates."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import KalmanFilter, Measurement
from virtualmodelcontrol.models import JointSpace, Underactuated

DT = 0.01
TENDONS = np.array([[1.0, -0.5, 0.0], [0.0, 1.0, 0.3]])  # two readings of three coordinates


def chain(actuation=None):
    """Two masses joined by a spring and a damper, the first on a spring to the ground, with a
    motor on the first only (or the ``actuation`` given)."""
    robot = vmc.Mechanism(
        "robot", model=JointSpace(2, unit="m"), actuation=actuation or Underactuated.joints(2, [0])
    )
    first, second = robot.joint(0), robot.joint(1)
    robot.add("m1", vmc.Inertance(first, 1.0))
    robot.add("m2", vmc.Inertance(second, 2.0))
    robot.add("ground", vmc.LinearSpring(first, 100.0))
    robot.add("link", vmc.LinearSpring(second - first, 40.0))
    robot.add("damper", vmc.LinearDamper(second - first, 3.0))
    robot.add("friction", vmc.LinearDamper(robot.joint(slice(0, 2)), 0.5))
    return robot


def system_of(robot):
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("ctrl"))


def textbook(x, P, y, H, R, h=None):
    """The Kalman update, with the model's reading ``h`` of the state (``H x`` if it is linear)."""
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    return x + K @ (y - (H @ x if h is None else h)), (np.eye(x.size) - K @ H) @ P


def test_a_matrix_gives_the_rows_of_a_sensor_of_combinations_on_q_and_on_v():
    both = Measurement([0.1, 0.2], [0.3, 0.4], 1e-4, 1e-3, matrix=TENDONS[:, :2], name="tendons")
    H = both.H(2)
    np.testing.assert_array_equal(H[:2], np.hstack([TENDONS[:, :2], np.zeros((2, 2))]))
    np.testing.assert_array_equal(H[2:], np.hstack([np.zeros((2, 2)), TENDONS[:, :2]]))
    only_q = Measurement([0.1, 0.2], None, 1e-4, matrix=TENDONS)
    np.testing.assert_array_equal(only_q.H(3), np.hstack([TENDONS, np.zeros((2, 3))]))
    only_v = Measurement(None, [0.1, 0.2], None, 1e-4, matrix=TENDONS)
    np.testing.assert_array_equal(only_v.H(3), np.hstack([np.zeros((2, 3)), TENDONS]))
    one = Measurement([0.5], None, 1e-4, matrix=[1.0, 2.0, 3.0])  # one reading: a row
    np.testing.assert_array_equal(one.H(3)[0, :3], [1.0, 2.0, 3.0])


def test_a_matrix_is_checked_against_its_readings_and_the_state():
    with pytest.raises(ValueError, match="2 rows for 3 values"):
        Measurement([0.1, 0.2, 0.3], None, 1e-4, matrix=TENDONS)
    with pytest.raises(ValueError, match="not both"):
        Measurement([0.1, 0.2], None, 1e-4, observed=[0, 1], matrix=TENDONS)
    with pytest.raises(ValueError, match="3 columns, the state 2"):
        Measurement([0.1, 0.2], None, 1e-4, matrix=TENDONS).H(2)


def test_the_filter_fuses_a_sensor_of_combinations_as_the_textbook_does():
    robot = chain()
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    kf.reset([0.2, -0.1])
    x, P = np.concatenate([kf.q, kf.v]), kf.P
    matrix = np.array([[1.0, 2.0], [0.0, -1.0]])
    seen = Measurement([0.3, 0.1], [0.05, -0.2], np.diag([1e-3, 2e-3]), 1e-2, matrix=matrix)
    kf.update([seen])
    R = np.diag([1e-3, 2e-3, 1e-2, 1e-2])
    H = np.block([[matrix, np.zeros((2, 2))], [np.zeros((2, 2)), matrix]])
    want, want_P = textbook(x, P, seen.y, H, R)
    np.testing.assert_allclose(np.concatenate([kf.q, kf.v]), want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(kf.P, want_P, rtol=1e-12, atol=1e-14)


def test_the_encoders_of_a_robot_with_fewer_motors_see_the_motors_not_the_configuration():
    robot = chain()
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    kf.reset([0.2, -0.1])
    x, P = np.concatenate([kf.q, kf.v]), kf.P
    theta, rate = [0.35], [0.12]  # the first mass; the second one is not measured
    kf.update([kf.encoder(theta, rate, 1e-4, 1e-3)])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # θ = Bᵀ q, θ̇ = Bᵀ v
    want, want_P = textbook(x, P, np.array([0.35, 0.12]), H, np.diag([1e-4, 1e-3]))
    np.testing.assert_allclose(np.concatenate([kf.q, kf.v]), want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(kf.P, want_P, rtol=1e-12, atol=1e-14)
    assert kf.P[1, 1] > 0.5 * P[1, 1]  # a reading of the first mass leaves the second one open


def test_the_motors_of_the_other_mass_are_a_different_matrix():
    robot = chain(Underactuated.joints(2, [1]))
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    measured = kf.encoder([0.4], [0.0], 1e-4, 1e-3)
    np.testing.assert_array_equal(measured.H(2)[0], [0.0, 1.0, 0.0, 0.0])
    mixed = np.array([[1.0], [0.5]])  # a motor that drives a combination of the two
    robot = chain(Underactuated(mixed))
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    np.testing.assert_allclose(kf.encoder([0.4], None, 1e-4).H(2)[0, :2], [1.0, 0.5])


def test_with_the_motors_alone_the_filter_follows_both_masses_through_the_coupling():
    robot = chain()
    plant = vmc.sim.ModelPlant(robot, q0=[0.0, 0.0], max_step=1e-3)
    kf = KalmanFilter(system_of(robot), DT, P0=1.0, Q=1e-6)
    kf.reset([0.0, 0.1])  # the second mass starts 0.1 from where it is
    rng, errors = np.random.default_rng(3), []
    for k in range(400):
        u = [6.0 * np.sin(2 * np.pi * 1.5 * plant.t) + 3.0 * np.sin(2 * np.pi * 4.0 * plant.t)]
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        plant.advance(DT)
        kf.predict(u)
        seen = plant.read()
        theta = seen["motor_position"] + rng.normal(0.0, 1e-4, 1)
        kf.update(
            [kf.encoder(theta, seen["motor_velocity"] + rng.normal(0.0, 1e-3, 1), 1e-8, 1e-6)]
        )
        if k >= 200:
            errors.append(kf.q - plant.q)
    assert np.abs(plant.q[1]) > 1e-3  # the second mass moves
    assert kf.rejected_total == 0
    assert np.sqrt(np.mean(np.square(errors))) < 0.1 * np.abs(plant.q).max()


def test_a_robot_with_a_motor_on_every_coordinate_keeps_the_encoder_it_had():
    robot = chain(vmc.models.Direct())
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    measured = kf.encoder([0.4, 0.2], [0.0, 0.0], 1e-4, 1e-3)
    assert measured.matrix is None
    np.testing.assert_array_equal(measured.y, [0.4, 0.2, 0.0, 0.0])


class Bent(Underactuated):
    """A motor whose angle also depends on the first coordinate, squared: not a matrix."""

    def motor_angles(self, q, p):
        return super().motor_angles(q, p) + 0.2 * q[0] ** 2

    def motor_rates(self, q, v, p):
        return super().motor_rates(q, v, p) + 0.4 * q[0] * v[0]


def test_a_motor_map_that_is_not_linear_gives_the_innovation_the_model_predicts():
    robot = chain(Bent(np.array([[1.0], [0.0]])))
    kf = KalmanFilter(system_of(robot), DT, gate=None)
    kf.reset([0.3, -0.1])
    kf.update([Measurement(None, [0.5, 0.1], None, 1e-6, name="rates")])  # moving: v is not 0
    x, P = np.concatenate([kf.q, kf.v]), kf.P
    assert x[2] == pytest.approx(0.5, abs=1e-3)
    seen = kf.encoder([0.37], [0.2], 1e-4, 1e-3, name="motors")
    assert seen.name == "motors"
    q0, v0 = x[0], x[2]
    J = 1.0 + 0.4 * q0  # the slope of the angle at the estimate
    np.testing.assert_allclose(seen.H(2)[0], [J, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(seen.H(2)[1], [0.0, 0.0, J, 0.0], atol=1e-12)
    kf.update([seen])
    h = np.array([q0 + 0.2 * q0**2, v0 + 0.4 * q0 * v0])  # the model's angle and rate here
    want, _ = textbook(x, P, np.array([0.37, 0.2]), seen.H(2), np.diag([1e-4, 1e-3]), h)
    np.testing.assert_allclose(np.concatenate([kf.q, kf.v]), want, rtol=1e-12, atol=1e-14)
