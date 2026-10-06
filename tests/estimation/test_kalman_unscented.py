"""The unscented prediction of the Kalman filter: the scaled unscented transform of the robot's
dynamics, exact on a linear robot and close to a Monte Carlo on a double pendulum, where the
linearised prediction is not."""

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.dynamics import compile_dynamics
from virtualmodelcontrol.estimation import KalmanFilter
from virtualmodelcontrol.models import SerialChain
from virtualmodelcontrol.robots import helyx

DT, SUBSTEPS = 0.2, 60


def double_pendulum():
    """Two links of 0.4 m about y, with 1 kg and 0.5 kg bobs, gravity and a little damping."""
    chain = SerialChain(
        ["revolute", "revolute"],
        axes=[[0, 1, 0], [0, 1, 0]],
        points=[[0, 0, 0], [0.4, 0, 0]],
        sites={"a": (1, [0.4, 0, 0]), "b": (2, [0.3, 0, 0])},
    )
    robot = vmc.Mechanism("double", model=chain, actuation=vmc.models.Direct(1.0))
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("ma", vmc.PointMass(robot.point("a"), 1.0))
    robot.add("mb", vmc.PointMass(robot.point("b"), 0.5))
    robot.add("gravity", vmc.Gravity(robot))
    robot.add("friction", vmc.LinearDamper(robot.joint(slice(0, 2)), 0.05))
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))


START = np.array([0.8, -0.6, 0.5, 0.4])  # q and v
SPREAD = 0.1**2 * np.eye(4)


def filters(Q=1e-12):
    system = double_pendulum()
    made = []
    for unscented in (False, True):
        kf = KalmanFilter(system, DT, P0=SPREAD, Q=Q, unscented=unscented, substeps=SUBSTEPS)
        kf.reset(START[:2])
        kf._x[2:] = START[2:]
        made.append(kf)
    return system, *made


def through(system, samples):
    """Columns of (q, v) through ``DT`` of the robot's own integrator, in the same sub-steps."""
    dynamics = compile_dynamics(system.robot, actuation=system.actuation)
    n, nq = samples.shape[1], samples.shape[0] // 2
    q, v = samples[:nq], samples[nq:]
    live = np.tile(np.ravel(dynamics.live_values())[:, None], (1, n))
    step, h = dynamics.step.map(n), DT / SUBSTEPS
    for k in range(SUBSTEPS):
        q, v = (np.array(m) for m in step(q, v, np.zeros((nq, n)), live, k * h, h))
    return np.vstack([q, v])


def textbook(system, start, spread, Q):
    """Wan and van der Merwe's scaled unscented transform, written out again: the points, their
    weights, the mean and the covariance, with κ = 0, β = 2 and the points a root of three
    standard deviations out, or fewer where the state has two entries."""
    n = len(start)
    alpha = min(1.0, np.sqrt(3.0 / n))
    lam = alpha**2 * n - n
    root = np.linalg.cholesky((n + lam) * spread)
    points = np.column_stack([start, *(start[:, None] + s * root for s in (1, -1))])
    w = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    wc = w.copy()
    w[0] = lam / (n + lam)
    wc[0] = w[0] + (1 - alpha**2 + 2.0)
    after = through(system, points)
    mean = after @ w
    cov = sum(wc[i] * np.outer(after[:, i] - mean, after[:, i] - mean) for i in range(2 * n + 1))
    return mean, cov + Q * np.eye(n)


def test_the_prediction_is_the_scaled_unscented_transform():
    system, _, ukf = filters(Q=1e-3)
    mean, cov = textbook(system, START, SPREAD, 1e-3)
    ukf.predict([0.0, 0.0])
    np.testing.assert_allclose(np.concatenate([ukf.q, ukf.v]), mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ukf.P, cov, rtol=1e-9, atol=1e-12)


def test_a_state_of_two_entries_has_its_points_a_standard_deviation_out():
    chain = SerialChain(
        ["revolute"], axes=[[0, 1, 0]], points=[[0, 0, 0]], sites={"b": (1, [0.5, 0, 0])}
    )
    robot = vmc.Mechanism("pendulum", model=chain, actuation=vmc.models.Direct(1.0))
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("bob", vmc.PointMass(robot.point("b"), 1.0))
    robot.add("gravity", vmc.Gravity(robot))
    system = vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))
    start, spread = np.array([1.2, 0.8]), 0.4**2 * np.eye(2)  # a wide start
    kf = KalmanFilter(system, DT, P0=spread, Q=1e-12, unscented=True, substeps=SUBSTEPS)
    kf.reset(start[:1])
    kf._x[1] = start[1]
    mean, cov = textbook(system, start, spread, 1e-12)
    kf.predict([0.0])
    np.testing.assert_allclose(np.concatenate([kf.q, kf.v]), mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(kf.P, cov, rtol=1e-9, atol=1e-12)


def test_the_unscented_prediction_follows_a_swing_the_linearised_one_cannot():
    system, ekf, ukf = filters()
    rng = np.random.default_rng(3)
    after = through(system, rng.multivariate_normal(START, SPREAD, 60000).T)
    mean, cov = after.mean(axis=1), np.cov(after)  # what the true distribution is
    for kf in (ekf, ukf):
        kf.predict([0.0, 0.0])
    ours, theirs = (np.concatenate([kf.q, kf.v]) for kf in (ukf, ekf))
    centre = through(system, START[:, None]).ravel()  # the mean point alone, not the transform
    assert np.linalg.norm(ours - mean) < 0.03  # [rad, rad/s] of a swing of 9
    assert np.linalg.norm(centre - mean) > 0.05  # the mean point alone is not enough
    assert np.linalg.norm(theirs - mean) > 1.0  # and the linearised one is far off
    assert np.abs(ukf.P - cov).max() < 0.08 * np.abs(cov).max()
    assert np.abs(ekf.P - cov).max() > 0.3 * np.abs(cov).max()


def test_on_a_linear_robot_the_unscented_prediction_is_the_linearised_one():
    robot = vmc.Mechanism("spring", model=vmc.models.JointSpace(2, unit="m"))
    both = robot.joint(slice(0, 2))
    for i, m in enumerate((1.0, 2.0)):
        robot.add(f"m{i}", vmc.Inertance(robot.joint(i), m))
    robot.add("spring", vmc.LinearSpring(both, [100.0, 50.0]))
    robot.add("damper", vmc.LinearDamper(both, [2.0, 1.0]))
    system = vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))
    a, b = (KalmanFilter(system, 0.01, unscented=flag, substeps=50) for flag in (False, True))
    for kf in (a, b):
        kf._x = np.array([0.1, -0.2, 0.3, 0.1])
    for _ in range(10):
        a.predict([1.0, -0.5])
        b.predict([1.0, -0.5])
    np.testing.assert_allclose(np.concatenate([b.q, b.v]), np.concatenate([a.q, a.v]), atol=2e-3)
    np.testing.assert_allclose(b.P, a.P, atol=3e-3)


def test_the_unscented_prediction_follows_a_time_varying_goal():
    mass = vmc.Mechanism("mass", model=vmc.models.JointSpace(1, unit="m"))
    goal = vmc.Custom(lambda t: 0.1 * ca.sin(8.0 * t), [vmc.Time()], dim=1, unit="m")
    mass.add("m", vmc.Inertance(mass.joint(0), 1.0))
    mass.add("pull", vmc.LinearSpring(mass.joint(0) - goal, 60.0))
    mass.add("damp", vmc.LinearDamper(mass.joint(0), 4.0))
    system = vmc.VirtualMechanismSystem(mass, vmc.Mechanism("idle"))
    plant = vmc.sim.ModelPlant(mass, max_step=1e-4)
    kf = KalmanFilter(system, 0.2, P0=1e-8, unscented=True, substeps=100)
    kf.reset([0.0])
    for _ in range(4):  # 0.8 s, in steps over which the goal turns by 1.6 rad
        kf.predict([], t=plant.t)
        plant.advance(0.2)
    assert abs(plant.q[0]) > 0.02  # the goal moves it
    assert abs(kf.q[0] - plant.q[0]) < 0.02 * abs(plant.q[0]) + 1e-4


def test_the_unscented_prediction_adds_the_process_noise():
    system = double_pendulum()
    quiet, noisy = (
        KalmanFilter(system, DT, P0=1e-4, Q=q, unscented=True, substeps=SUBSTEPS)
        for q in (1e-12, 1e-2)
    )
    for kf in (quiet, noisy):
        kf.predict([0.0, 0.0])
    np.testing.assert_allclose(noisy.P - quiet.P, 1e-2 * np.eye(4), atol=1e-6)


def test_the_unscented_prediction_of_a_soft_arm_follows_the_simulated_arm():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))  # tendons, PCC kinematics, gravity
    q0 = np.linspace(-0.004, 0.006, 9)
    plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=1e-4)
    system = vmc.VirtualMechanismSystem(arm, vmc.Mechanism("ctrl"))
    kf = KalmanFilter(system, 1 / 500, P0=1e-6, unscented=True)  # a start known to a mrad
    kf.reset(q0)
    u = np.linspace(-0.3, 0.4, 9)
    for _ in range(10):
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        plant.advance(1 / 500)
        kf.predict(u)
    moved = np.abs(plant.q - q0).max()
    assert moved > 1e-4
    assert np.abs(kf.q - plant.q).max() < 0.02 * moved
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)  # the covariance stays a covariance
