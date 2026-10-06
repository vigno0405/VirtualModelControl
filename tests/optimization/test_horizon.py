"""Moving-horizon estimation: the batch least squares it solves, on a linear robot, and a swing."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.dynamics import compile_dynamics
from virtualmodelcontrol.estimation import KalmanFilter, Measurement
from virtualmodelcontrol.optimization import MovingHorizon

DT = 0.02


def linear_system():
    robot = vmc.Mechanism(
        "springs", model=vmc.models.JointSpace(2, unit="m"), actuation=vmc.models.Direct(1.0)
    )
    both = robot.joint(slice(0, 2))
    for i, m in enumerate((1.0, 2.0)):
        robot.add(f"m{i}", vmc.Inertance(robot.joint(i), m))
    robot.add("spring", vmc.LinearSpring(both, [100.0, 50.0]))
    robot.add("damper", vmc.LinearDamper(both, [2.0, 1.0]))
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))


def truth_run(system, steps=40, seed=0):
    """A run under a torque, its true states, the torques, and noisy position readings."""
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.05, -0.03], max_step=1e-3)
    rng = np.random.default_rng(seed)
    states, commands, readings = [], [], []
    for _ in range(steps):
        states.append(np.concatenate([plant.q, plant.v]))
        readings.append(plant.q + rng.normal(0, 2e-3, 2))
        u = np.array([2.0 * np.sin(5 * plant.t), 1.5 * np.cos(3 * plant.t)])
        commands.append(u)
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        plant.advance(DT)
    return np.array(states), np.array(commands), np.array(readings)


def batch(system, prior, P, Q, commands, readings, R):
    """The same least squares by one linear solve: x_{k+1} = A x_k + B u_k from the model's step."""
    dynamics = compile_dynamics(system.robot, actuation=system.actuation)
    n, steps = 2, len(readings)
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    x0, u0 = np.zeros(4), np.zeros(2)

    live = dynamics.live_values()  # the springs' and dampers' values: left out, they are zero

    def step(x, u):
        return np.concatenate(
            [np.array(a).ravel() for a in dynamics.step(x[:n], x[n:], u, live, 0.0, DT)]
        )

    c = step(x0, u0)
    for j in range(4):
        e = np.eye(4)[j]
        A[:, j] = step(e, u0) - c
    for j in range(2):
        B[:, j] = step(x0, np.eye(2)[j]) - c
    rows, rhs = [], []
    total = 4 * steps
    sqrt = lambda M: np.linalg.cholesky(np.linalg.inv(M)).T  # noqa: E731
    first = np.zeros((4, total))
    first[:, :4] = sqrt(P)
    rows.append(first)
    rhs.append(sqrt(P) @ prior)
    for k in range(steps - 1):
        block = np.zeros((4, total))
        block[:, 4 * k : 4 * k + 4] = -sqrt(Q) @ A
        block[:, 4 * k + 4 : 4 * k + 8] = sqrt(Q)
        rows.append(block)
        rhs.append(sqrt(Q) @ (B @ commands[k] + c))
    for k in range(steps):
        block = np.zeros((2, total))
        block[:, 4 * k : 4 * k + 2] = sqrt(R)
        rows.append(block)
        rhs.append(sqrt(R) @ readings[k])
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return solution.reshape(steps, 4)


def test_the_window_is_the_least_squares_it_says_it_is():
    system = linear_system()
    _, commands, readings = truth_run(system)
    R, Q, P = 2e-3**2 * np.eye(2), 1e-4 * np.eye(4), 1e-2 * np.eye(4)
    window = 8
    mhe = MovingHorizon(system, DT, window=window, Q=Q, P=P)
    mhe.reset([0.05, -0.03])
    prior = np.array([0.05, -0.03, 0.0, 0.0])
    for k in range(len(readings)):
        mhe.step(commands[k - 1] if k else [], [Measurement(readings[k], None, R)])
    # the last window, solved at once by numpy, with the prior the filter slid to
    last = len(readings) - window - 1
    mhe_prior = mhe._prior
    want = batch(
        system, mhe_prior, mhe._arrival, Q, commands[last : last + window], readings[last:], R
    )
    np.testing.assert_allclose(mhe.states, want, atol=1e-6)
    assert last > 0 and not np.allclose(mhe_prior, prior)  # it did slide


def kalman(system, prior, P, Q, commands, readings, R):
    """A textbook Kalman filter of the same discrete model: x⁺ = A x + B u + c, y = q."""
    dynamics = compile_dynamics(system.robot, actuation=system.actuation)
    live = dynamics.live_values()

    def step(x, u):
        return np.concatenate(
            [np.array(a).ravel() for a in dynamics.step(x[:2], x[2:], u, live, 0.0, DT)]
        )

    A = np.column_stack([step(e, np.zeros(2)) for e in np.eye(4)])
    H = np.eye(4)[:2]
    x, S = prior, P
    for k in range(len(readings)):
        if k:
            x, S = step(x, commands[k - 1]), A @ S @ A.T + Q
        gain = S @ H.T @ np.linalg.inv(H @ S @ H.T + R)
        x, S = x + gain @ (readings[k] - H @ x), (np.eye(4) - gain @ H) @ S
    return x, S


def test_with_the_whole_history_in_the_window_it_is_the_kalman_filter():
    system = linear_system()
    _, commands, readings = truth_run(system, steps=25)
    R, Q, P = 2e-3**2 * np.eye(2), 1e-4 * np.eye(4), 1e-2 * np.eye(4)
    mhe = MovingHorizon(system, DT, window=40, Q=Q, P=P)
    mhe.reset([0.05, -0.03])
    for k in range(len(readings)):
        mhe.step(commands[k - 1] if k else [], [Measurement(readings[k], None, R)])
    x, S = kalman(system, np.array([0.05, -0.03, 0.0, 0.0]), P, Q, commands, readings, R)
    np.testing.assert_allclose(np.concatenate([mhe.q, mhe.v]), x, atol=1e-6)
    np.testing.assert_allclose(mhe.P, S, atol=1e-8)  # the same covariance, from the Hessian


def swing():
    chain = vmc.models.SerialChain(
        ["revolute"], axes=[[0, 1, 0]], points=[[0, 0, 0]], sites={"bob": (1, [0.5, 0, 0])}
    )
    robot = vmc.Mechanism("pendulum", model=chain, actuation=vmc.models.Direct(1.0))
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("bob", vmc.PointMass(robot.point("bob"), 1.0))
    robot.add("gravity", vmc.Gravity(robot))
    robot.add("friction", vmc.LinearDamper(robot.joint(0), 0.05))
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))


def test_a_swing_is_followed_from_noisy_angles_alone_velocity_included():
    system = swing()
    plant = vmc.sim.ModelPlant(system.robot, q0=[1.4], max_step=1e-3)
    rng = np.random.default_rng(5)
    mhe = MovingHorizon(system, 0.01, window=10, Q=1e-6 * np.eye(2), P=1e-1 * np.eye(2))
    mhe.reset([1.3])  # a wrong start
    errors = []
    for _ in range(120):
        reading = Measurement(plant.q + rng.normal(0, 3e-3, 1), None, 3e-3**2, name="angle")
        estimate = mhe.step([0.0], [reading])
        errors.append(estimate - np.concatenate([plant.q, plant.v]))
        plant.advance(0.01)
    late = np.abs(np.array(errors[40:]))
    assert late[:, 0].max() < 6e-3  # [rad] about twice the reading's noise
    assert late[:, 1].max() < 0.15  # [rad/s]: a velocity no sensor reads


def test_a_step_without_readings_goes_on_and_a_partial_sensor_is_enough():
    system = linear_system()
    _, commands, readings = truth_run(system, steps=30)
    R = 2e-3**2
    mhe = MovingHorizon(system, DT, window=10, Q=1e-4 * np.eye(4), P=1e-2 * np.eye(4))
    mhe.reset([0.05, -0.03])
    seen = []
    for k in range(len(readings)):
        if k % 3 == 1:
            heard = []  # nothing arrives
        elif k % 3 == 2:
            heard = [Measurement(readings[k][:1], None, R, observed=[0], name="first")]
        else:
            heard = [Measurement(readings[k], None, R, name="both")]
        seen.append(mhe.step(commands[k - 1] if k else [], heard))
    assert np.all(np.isfinite(seen)) and len(mhe.states) == 11
    truth = np.array(truth_run(system, steps=30)[0])
    assert np.abs(np.array(seen)[10:, :2] - truth[10:, :2]).max() < 8e-3


def test_the_covariance_shrinks_as_the_readings_come_and_reset_forgets_them():
    system = linear_system()
    _, commands, readings = truth_run(system, steps=20)
    R = 2e-3**2
    mhe = MovingHorizon(system, DT, window=10, Q=1e-4 * np.eye(4), P=1e-2 * np.eye(4))
    mhe.reset([0.05, -0.03])
    widths = []
    for k in range(len(readings)):
        mhe.step(commands[k - 1] if k else [], [Measurement(readings[k], None, R)])
        widths.append(np.trace(mhe.P))
    assert widths[-1] < 0.5 * widths[0] and np.all(np.linalg.eigvalsh(mhe.P) > 0)
    mhe.reset([0.0, 0.0])
    assert len(mhe.states) == 0
    mhe.step([], [Measurement(readings[0], None, R)])
    assert len(mhe.states) == 1 and abs(mhe.q[0] - readings[0][0]) < 1e-2


def test_the_encoder_goes_through_the_transmission_as_the_filters_does():
    system = linear_system()
    mhe = MovingHorizon(system, DT)
    kf = KalmanFilter(system, DT)
    a, b = (
        mhe.encoder([0.1, 0.2], [0.3, 0.4], 1e-6, 1e-4),
        kf.encoder([0.1, 0.2], [0.3, 0.4], 1e-6, 1e-4),
    )
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.R, b.R)


def test_a_floating_body_is_refused():
    chain = vmc.models.SerialChain(
        ["floating"], axes=[None], points=[[0, 0, 0]], sites={"c": (1, [0, 0, 0])}
    )
    body = vmc.Mechanism("body", model=chain)
    body.add("m", vmc.PointMass(body.point("c"), 1.0))
    with pytest.raises(ValueError, match="velocity as configuration"):
        MovingHorizon(vmc.VirtualMechanismSystem(body, vmc.Mechanism("idle")), DT)


def test_the_times_of_the_steps_reach_a_time_varying_robot():
    mass = vmc.Mechanism(
        "mass", model=vmc.models.JointSpace(1, unit="m"), actuation=vmc.models.Direct(1.0)
    )
    goal = vmc.Custom(lambda t: 0.1 * ca.sin(20.0 * t), [vmc.Time()], dim=1, unit="m")
    mass.add("m", vmc.Inertance(mass.joint(0), 1.0))
    mass.add("pull", vmc.LinearSpring(mass.joint(0) - goal, 300.0))
    mass.add("damp", vmc.LinearDamper(mass.joint(0), 6.0))
    system = vmc.VirtualMechanismSystem(mass, vmc.Mechanism("idle"))
    dynamics = compile_dynamics(mass, actuation=system.actuation)
    live = dynamics.live_values()
    x, truth = np.zeros(2), []  # a run of the model's own steps, which the window must reproduce
    for k in range(30):
        truth.append(x)
        x = np.concatenate(
            [np.array(a).ravel() for a in dynamics.step(x[:1], x[1:], [0.0], live, 0.01 * k, 0.01)]
        )
    mhe = MovingHorizon(system, 0.01, window=10, Q=1e-12 * np.eye(2), P=1e-2 * np.eye(2))
    mhe.reset([0.0])
    for k, state in enumerate(truth):
        reading = Measurement(state[:1], None, 1e-10)
        estimate = mhe.step([0.0], [reading], t=0.01 * k)
    assert np.abs(estimate - truth[-1]).max() < 1e-5  # position and velocity, to the step's own


def test_a_short_window_that_slides_is_the_kalman_filter_at_every_step():
    system = linear_system()
    _, commands, readings = truth_run(system, steps=30)
    R, Q, P = 2e-3**2 * np.eye(2), 1e-4 * np.eye(4), 1e-2 * np.eye(4)
    mhe = MovingHorizon(system, DT, window=3, Q=Q, P=P)
    mhe.reset([0.05, -0.03])
    prior = np.array([0.05, -0.03, 0.0, 0.0])
    for k in range(len(readings)):
        estimate = mhe.step(commands[k - 1] if k else [], [Measurement(readings[k], None, R)])
        x, S = kalman(system, prior, P, Q, commands[:k], readings[: k + 1], R)
        np.testing.assert_allclose(estimate, x, atol=1e-6, err_msg=f"step {k}")
        np.testing.assert_allclose(mhe.P, S, atol=1e-8, err_msg=f"step {k}")
