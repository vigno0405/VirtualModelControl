from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest
from scipy.linalg import expm

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core.params import Param, constants
from virtualmodelcontrol.estimation import KalmanFilter, Measurement
from virtualmodelcontrol.estimation.measurement import block_diagonal, covariance
from virtualmodelcontrol.robots import helyx

DT = 0.01
MASS, STIFFNESS, DAMPING, ETA = (
    np.array([1.0, 2.0]),
    np.array([100.0, 50.0]),
    np.array([2.0, 1.0]),
    0.5,
)


def linear_robot(eta=ETA, wall=False):
    """Two independent masses on springs and dampers, with the motors' efficiency ``eta``."""
    robot = vmc.Mechanism(
        "robot", model=vmc.models.JointSpace(2, unit="m"), actuation=vmc.models.Direct(eta)
    )
    both = robot.joint(slice(0, 2))
    for i in range(2):
        robot.add(f"mass{i}", vmc.Inertance(robot.joint(i), MASS[i]))
    robot.add("spring", vmc.LinearSpring(both, Param("stiffness", STIFFNESS, scope="design")))
    robot.add("damper", vmc.LinearDamper(both, Param("damping", DAMPING, scope="design")))
    if wall:  # surroundings: a stiff spring on the first coordinate
        robot.add("wall", vmc.LinearSpring(robot.joint(0), Param("wall", 1e4, scope="design")))
    return robot


def system_of(robot):
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("ctrl"))


def labs_step(x, P, u, Q):
    """The lab's discretisation: A = expm(Ac dt), B = Ac^-1 (A - I) Bc, with M, K, D constant."""
    M, K, D = np.diag(MASS), np.diag(STIFFNESS), np.diag(DAMPING)
    Minv, z, eye = np.linalg.inv(M), np.zeros((2, 2)), np.eye(2)
    Ac = np.block([[z, eye], [-Minv @ K, -Minv @ D]])
    Bc = np.vstack([z, Minv * ETA])
    A = expm(Ac * DT)
    B = np.linalg.solve(Ac, (A - np.eye(4)) @ Bc)
    return A @ x + B @ u, A @ P @ A.T + Q


def state(kf):
    return np.concatenate([kf.q, kf.v])


def test_the_prediction_is_the_labs_zero_order_hold_of_a_linear_robot():
    kf = KalmanFilter(system_of(linear_robot()), DT)
    kf.reset([0.3, -0.2])
    x, P, Q = np.array([0.3, -0.2, 0.0, 0.0]), 1e-2 * np.eye(4), 1e-6 * np.eye(4)
    for u in ([1.5, -0.5], [-2.0, 3.0], [0.0, 1.0]):
        kf.predict(u)
        x, P = labs_step(x, P, np.array(u), Q)
        np.testing.assert_allclose(state(kf), x, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(kf.P, P, rtol=1e-10, atol=1e-13)
    assert np.abs(x[2:]).max() > 1e-3  # the velocities the check ran through are not zero


def test_the_prediction_of_a_soft_arm_follows_the_simulated_arm():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))  # tendons, PCC kinematics, gravity
    q0 = np.linspace(-0.004, 0.006, 9)
    plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=1e-4)
    kf = KalmanFilter(system_of(arm), 1 / 500)
    kf.reset(q0)
    u = np.linspace(-0.3, 0.4, 9)
    for _ in range(10):
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        plant.advance(1 / 500)
        kf.predict(u)
    moved = np.abs(plant.q - q0).max()
    assert moved > 1e-4
    assert np.abs(kf.q - plant.q).max() < 0.02 * moved


def test_with_encoders_alone_the_error_is_below_their_noise_and_a_wrong_start_converges():
    robot = linear_robot()
    plant = vmc.sim.ModelPlant(robot, q0=[0.1, -0.1], max_step=1e-3)
    kf = KalmanFilter(system_of(robot), 0.005, P0=1.0)
    kf.reset([0.5, 0.5])
    rng, sq, sv = np.random.default_rng(5), 0.01, 0.05
    err_q, err_v = [], []
    for k in range(300):
        u = 20.0 * np.sin(2 * np.pi * 0.5 * plant.t) * np.ones(2)
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        plant.advance(0.005)
        kf.predict(u)
        seen = plant.read()
        theta, rate = seen["motor_position"] + rng.normal(0, sq, 2), seen["motor_velocity"]
        kf.update([kf.encoder(theta, rate + rng.normal(0, sv, 2), sq**2, sv**2)])
        if k >= 150:
            err_q.append(kf.q - plant.q)
            err_v.append(kf.v - plant.v)
    assert kf.rejected_total == 0
    assert np.sqrt(np.mean(np.square(err_q))) < 0.5 * sq
    assert np.sqrt(np.mean(np.square(err_v))) < 0.5 * sv


def test_a_measurement_is_gated_on_its_innovation_and_the_gate_can_be_off():
    kf = KalmanFilter(system_of(linear_robot()), DT)
    good = Measurement([0.01, -0.01], [0.0, 0.0], 1e-4, 1e-4, name="encoder")
    bad = Measurement([1.0, 1.0], [0.0, 0.0], 1e-4, 1e-4, name="mocap")
    worse = Measurement([-1.0, 2.0], [0.0, 0.0], 1e-4, 1e-4, name="imu")
    kf.update([good, bad, worse])
    assert kf.rejected == ("mocap", "imu") and kf.rejected_total == 2
    np.testing.assert_allclose(kf.q, [0.01, -0.01], atol=2e-4)  # only the good one was fused
    kf.update([bad])
    assert kf.rejected == ("mocap",) and kf.rejected_total == 3
    kf.update([good])
    assert kf.rejected == () and kf.rejected_total == 3
    free = KalmanFilter(system_of(linear_robot()), DT, gate=None)
    free.update([good, bad])
    assert free.rejected == () and free.rejected_total == 0
    assert abs(free.q[0] - 0.5) < 0.05  # the outlier counts as much as the good one


def test_the_gate_takes_a_measurement_whose_distance_is_exactly_the_limit():
    kf = KalmanFilter(system_of(linear_robot()), DT, gate=0.0)
    same = Measurement(kf.q, kf.v, 1e-4, 1e-4, name="same")
    kf.update([same])
    assert kf.rejected == ()  # d2 = 0 <= 0
    kf.update([Measurement(kf.q + 1e-6, kf.v, 1e-4, 1e-4, name="off")])
    assert kf.rejected == ("off",)


def test_switching_the_gate_off_clears_what_the_last_update_rejected():
    kf = KalmanFilter(system_of(linear_robot()), DT)
    out = Measurement([1.0, 1.0], [0.0, 0.0], 1e-4, 1e-4, name="out")
    kf.update([out])
    assert kf.rejected == ("out",)
    kf.gate = None
    kf.update([out])
    assert kf.rejected == () and kf.rejected_total == 1


def test_a_reset_starts_again_at_rest_with_the_starting_covariance():
    kf = KalmanFilter(system_of(linear_robot()), DT, P0=0.02)
    kf.reset([0.2, 0.2])
    kf.predict([1.0, 1.0])
    out = Measurement([5.0, 5.0], [0.0, 0.0], 1e-4, 1e-4, name="out")
    kf.update([out], expected=["encoder"])
    assert kf.rejected == ("out",) and kf.missing == ("encoder",)
    assert np.abs(kf.v).max() > 0 and not np.allclose(kf.P, 0.02 * np.eye(4))
    kf.reset()
    np.testing.assert_array_equal(state(kf), np.zeros(4))
    np.testing.assert_array_equal(kf.P, 0.02 * np.eye(4))
    assert kf.rejected == () and kf.missing == () and kf.rejected_total == 1  # the count stays


def test_a_sensor_of_some_coordinates_corrects_those_and_leaves_the_rest():
    kf = KalmanFilter(
        system_of(linear_robot()), DT, gate=None, P0=np.diag([0.01, 0.04, 0.01, 0.01])
    )
    kf.update([Measurement(q=[0.5], Rq=0.02, observed=[1])])
    np.testing.assert_allclose(kf.q, [0.0, 0.5 * 0.04 / 0.06], rtol=1e-12)
    np.testing.assert_allclose(np.diag(kf.P), [0.01, 0.04 * 0.02 / 0.06, 0.01, 0.01], rtol=1e-12)
    kf.update([Measurement(v=[0.3], Rv=0.01, observed=[0])])
    np.testing.assert_allclose(kf.v, [0.15, 0.0], rtol=1e-12)
    np.testing.assert_array_equal(Measurement([0, 0], [0, 0], 1, 1).H(2), np.eye(4))
    np.testing.assert_array_equal(
        Measurement([0], [0], 1, 1, observed=[1]).H(2), [[0, 1, 0, 0], [0, 0, 0, 1]]
    )


def test_two_sensors_fuse_each_with_its_own_covariance():
    kf = KalmanFilter(system_of(linear_robot()), DT, gate=None, P0=0.01)
    a = Measurement(q=[0.2], Rq=0.01, observed=[0], name="a")
    b = Measurement(q=[0.8], Rq=0.04, observed=[0], name="b")
    kf.update([a, b])
    weight = 1 / 0.01 + 1 / 0.01 + 1 / 0.04  # the prior (at 0), a and b, as information
    assert kf.q[0] == pytest.approx((0.2 / 0.01 + 0.8 / 0.04) / weight, rel=1e-12)
    assert kf.P[0, 0] == pytest.approx(1 / weight, rel=1e-12)


def test_the_sensors_that_offered_nothing_are_missing_and_a_rejected_one_is_not():
    kf = KalmanFilter(system_of(linear_robot()), DT)
    both = ("encoder", "mocap")
    kf.update([], expected=both)
    assert kf.missing == both and kf.rejected == ()
    enc = Measurement([0.0, 0.0], [0.0, 0.0], 1e-4, 1e-4, name="encoder")
    kf.update([enc], expected=both)
    assert kf.missing == ("mocap",)
    out = Measurement([1.0, 1.0], [0.0, 0.0], 1e-4, 1e-4, name="mocap")
    kf.update([enc, out], expected=both)
    assert kf.missing == () and kf.rejected == ("mocap",)
    kf.update([enc])
    assert kf.missing == ()  # nothing was expected


def test_the_robot_holds_the_arm_alone_and_the_efficiency_is_the_systems():
    plain = system_of(linear_robot())
    walled = system_of(linear_robot(wall=True))
    filters = {
        "plain": KalmanFilter(plain, DT),
        "arm_only": KalmanFilter(walled, DT, robot=linear_robot()),
        "walled": KalmanFilter(walled, DT),
        "other_eta": KalmanFilter(plain, DT, robot=linear_robot(eta=1.0)),
    }
    for kf in filters.values():
        kf.reset([0.1, 0.1])
        kf.predict([1.0, -1.0])
    ref = state(filters["plain"])
    np.testing.assert_allclose(state(filters["arm_only"]), ref, rtol=1e-12)
    np.testing.assert_allclose(state(filters["other_eta"]), ref, rtol=1e-12)
    assert np.abs(state(filters["walled"]) - ref).max() > 1e-3


def test_the_encoder_goes_through_the_transmission_and_the_motors_come_back():
    arm = helyx.arm("290-145-145")
    kf = KalmanFilter(system_of(arm), 1 / 500)
    theta, rate = np.linspace(-0.3, 0.4, 9), np.linspace(0.2, -0.1, 9)
    m = kf.encoder(theta, rate, 1e-3, 5e-4)
    assert m.name == "encoder" and m.y.size == 18
    kf.reset(m.y[:9])
    np.testing.assert_allclose(kf.theta, theta, atol=1e-12)
    back = arm.actuation.motor_rates(
        ca.DM(m.y[:9]), ca.DM(m.y[9:]), constants(arm.actuation.params)
    )
    np.testing.assert_allclose(np.array(back).ravel(), rate, atol=1e-12)  # the rates map back
    assert kf.encoder(theta, None, 1e-3).y.size == 9  # no rates
    plain = KalmanFilter(system_of(linear_robot()), DT, gate=None)
    seen = plain.encoder([0.2, 0.3], [1.0, 2.0], 1e-4, 1e-4)
    np.testing.assert_array_equal(seen.y, [0.2, 0.3, 1.0, 2.0])  # one motor per coordinate
    plain.update([seen])
    np.testing.assert_allclose(plain.theta_dot, plain.v, rtol=1e-12)


def test_the_filter_needs_as_many_velocities_as_configuration_coordinates():
    space = vmc.Product(vmc.Euclidean(2), vmc.SO2())  # planar body: nq = 4, nv = 3
    robot = SimpleNamespace(model=SimpleNamespace(space=space))
    with pytest.raises(ValueError, match="nq=4 and nv=3"):
        KalmanFilter(SimpleNamespace(robot=robot), DT)


def test_covariances_come_as_a_variance_one_per_coordinate_or_a_matrix():
    stacked = block_diagonal([np.ones((1, 1)), 2 * np.eye(2), np.full((1, 1), 3.0)])
    np.testing.assert_array_equal(stacked, np.diag([1.0, 2.0, 2.0, 3.0]))
    np.testing.assert_array_equal(covariance(2.0, 3), 2.0 * np.eye(3))
    np.testing.assert_array_equal(covariance([1.0, 2.0], 2), np.diag([1.0, 2.0]))
    np.testing.assert_array_equal(covariance(np.ones((2, 2)), 2), np.ones((2, 2)))
    with pytest.raises(ValueError, match="variances"):
        covariance([1.0, 2.0, 3.0], 2)
    kf = KalmanFilter(system_of(linear_robot()), DT, Q=[1.0, 2.0, 3.0, 4.0], P0=5.0)
    np.testing.assert_array_equal(kf.P, 5.0 * np.eye(4))
    with pytest.raises(ValueError, match="needs q, v or both"):
        Measurement()
    with pytest.raises(ValueError, match="Rq is needed"):
        Measurement(q=[0.0, 0.0])
    with pytest.raises(ValueError, match="2 observed coordinates for 3 values"):
        Measurement(q=[0.0, 0.0, 0.0], Rq=1.0, observed=[0, 1])
    with pytest.raises(ValueError, match="the measurement has 3 coordinates"):
        Measurement(q=[0.0, 0.0, 0.0], Rq=1.0).H(2)
