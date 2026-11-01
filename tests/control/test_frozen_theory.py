"""The frozen controller against its theory, for both rests: a gradient, so passive.

With π(θ) the configuration the motors give (the actuated joints from θ, the passive ones at a
rest) and P = ∂π/∂θ, the torque is Pᵀ τ(π, P θ̇). A constant rest has P = (B⁺)ᵀ, so it is the naive
law B⁺ τ at π. A moving rest follows the passive joints' static balance and adds the chain-rule
term. Both are the gradient of the virtual potential at π(θ), and both dampers are the Rayleigh
gradient of ½ ḋᵀ C d."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from underactuated_lab import LabRead, build
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import StateController
from virtualmodelcontrol.control import underactuated as ua
from virtualmodelcontrol.robots import planar

GOAL = np.array([0.45, 0.15, 0.0])
STIFFNESS, DAMPING = 150.0, 25.0
THETA = np.array([0.35, -0.25])


def virtual(arm, spring=True, damper=True, compensation=True):
    ctrl = vmc.Mechanism("ctrl")
    if spring:
        ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - GOAL, STIFFNESS))
    if damper:
        ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), DAMPING))
    if compensation:
        ctrl.add("gravity", vmc.GravityCompensation(arm))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))


@pytest.fixture(scope="module")
def arm():
    return build("three")


@pytest.fixture(scope="module")
def B(arm):
    return np.asarray(arm.actuation.params["B"].value)


def torque(controller, theta, rate=(0.0, 0.0)):
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.asarray(rate, dtype=float))
    controller.reset(0.0, meas)
    return np.asarray(controller.step(0.0, meas)["law_torque"]).ravel()


def jacobian(function, x, h=1e-6):
    return np.column_stack([(function(x + h * e) - function(x - h * e)) / (2 * h)
                            for e in np.eye(len(x))])  # fmt: skip


def asymmetry(function, x):
    J = jacobian(function, x)
    return abs(J - J.T).max() / abs(J).max()


def reads(arm):
    """The two rests as reads of the motors."""
    dynamics = vmc.compile_dynamics(arm)
    return {"constant": ua.Frozen(arm.actuation), "moving": ua.Moving(arm.actuation, dynamics)}


def frozen_state(read, theta, rate):
    """(π(θ), P θ̇, P) of a read, with P = (B⁺)ᵀ for a constant rest."""
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.asarray(rate, dtype=float))
    if hasattr(read, "pullback"):
        return read.pullback(meas)
    q, v = read(meas)
    return q, v, np.linalg.pinv(read.B).T


def test_a_constant_rest_is_the_naive_law_at_the_frozen_configuration(arm, B):
    compiled = virtual(arm)
    frozen, read = ua.controller(compiled, "frozen"), ua.Frozen(arm.actuation)
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta, rate = rng.uniform(-0.6, 0.6, 2), rng.uniform(-1.0, 1.0, 2)
        q, v, _ = frozen_state(read, theta, rate)
        tau = np.asarray(compiled.tau(q, v, np.zeros(0), compiled.live_values(), 0.0)).ravel()
        np.testing.assert_allclose(torque(frozen, theta, rate), np.linalg.pinv(B) @ tau, atol=1e-12)


@pytest.mark.parametrize("rest", ["constant", "moving"])
def test_the_frozen_torque_has_no_curl_and_does_no_work_on_a_loop_of_motor_angles(arm, rest):
    controller = ua.controller(virtual(arm), "frozen", rest=rest)  # spring, compensation, damper
    for theta in (THETA, THETA + np.array([0.3, -0.2]), THETA - np.array([0.2, 0.4])):
        assert asymmetry(lambda x: torque(controller, x), theta) < 1e-7
    angles = np.linspace(0.0, 2.0 * np.pi, 241)
    loop = THETA + 0.15 * np.column_stack([np.cos(angles), np.sin(angles)])
    u = np.array([torque(controller, theta) for theta in loop])
    work = np.sum(0.5 * (u[1:] + u[:-1]) * np.diff(loop, axis=0))
    assert abs(work) < 1e-9  # J


def test_the_rest_that_follows_the_balance_without_the_pullback_does_work_on_a_loop(arm):
    compiled = virtual(arm)  # the lab's law: the balance at every reading, torque B⁺ τ
    old = StateController(compiled, [], read=LabRead(arm))
    assert asymmetry(lambda x: torque(old, x), THETA) > 1e-3
    angles = np.linspace(0.0, 2.0 * np.pi, 241)
    loop = THETA + 0.15 * np.column_stack([np.cos(angles), np.sin(angles)])
    u = np.array([torque(old, theta) for theta in loop])
    assert abs(np.sum(0.5 * (u[1:] + u[:-1]) * np.diff(loop, axis=0))) > 1e-2  # J


@pytest.mark.parametrize("rest", ["constant", "moving"])
def test_the_torque_is_minus_the_gradient_of_the_virtual_potential_at_the_frozen_pose(arm, rest):
    kin, read = vmc.Kinematics(arm), reads(arm)[rest]
    controller = ua.controller(virtual(arm, damper=False, compensation=False), "frozen", rest=rest)

    def potential(theta):
        q = frozen_state(read, theta, [0.0, 0.0])[0]
        return 0.5 * STIFFNESS * np.sum((np.asarray(kin.position(q, "tip")).ravel() - GOAL) ** 2)

    h = 1e-5
    gradient = np.array([(potential(THETA + h * e) - potential(THETA - h * e)) / (2 * h)
                         for e in np.eye(2)])  # fmt: skip
    np.testing.assert_allclose(torque(controller, THETA), -gradient, atol=1e-6)


def test_the_moving_rest_pulls_back_with_the_derivative_of_the_frozen_configuration(arm):
    moving = reads(arm)["moving"]
    meas = vmc.Signals(0.0, motor_position=THETA, motor_velocity=np.zeros(2))
    q, _, P = moving.pullback(meas)
    numeric = jacobian(lambda x: frozen_state(moving, x, [0.0, 0.0])[0], THETA)
    np.testing.assert_allclose(P, numeric, atol=1e-7)
    constant = frozen_state(reads(arm)["constant"], THETA, [0.0, 0.0])[2]
    assert abs(P - constant).max() > 1e-2  # the passive joint moves with the motors
    dynamics = vmc.compile_dynamics(arm)  # and it sits at its balance
    gradient = np.asarray(
        dynamics.residual(q, np.zeros(3), np.zeros(3), np.zeros(2), dynamics.live_values(), 0.0)
    ).ravel()
    assert abs(gradient[1]) < 1e-9 and abs(q[1]) > 1e-2


def test_the_first_order_difference_between_the_frozen_and_the_naive_law(arm, B):
    """u_fr − u_n = −B⁺ ∂τ/∂q|π E (q − q̄) + O(‖E (q − q̄)‖²) for a constant rest."""
    compiled = virtual(arm, damper=False, compensation=False)
    frozen, naive = ua.controller(compiled, "frozen"), ua.controller(compiled, "naive")
    read, E, Bp = ua.Frozen(arm.actuation), ua.projector(B), np.linalg.pinv(B)
    live = compiled.live_values()

    def tau(q):
        return np.asarray(compiled.tau(q, np.zeros(3), np.zeros(0), live, 0.0)).ravel()

    gaps = []
    for eps in (0.04, 0.02, 0.01):
        q = np.array([THETA[0], eps, THETA[1]])
        point = read.point(THETA)
        first = -Bp @ jacobian(tau, point) @ (E @ (q - np.concatenate([[0], [read.rest[1]], [0]])))
        meas = vmc.Signals(
            0.0, motor_position=THETA, motor_velocity=np.zeros(2), q=q, v=np.zeros(3)
        )
        naive.reset(0.0, meas)
        difference = torque(frozen, THETA) - np.asarray(naive.step(0.0, meas)["law_torque"])
        gaps.append(abs(difference - first).max())
    assert gaps[0] / 0.04**2 == pytest.approx(gaps[2] / 0.01**2, rel=0.3)  # second order
    assert gaps[2] < 1e-3 * abs(first).max() * 10


@pytest.mark.parametrize("rest", ["constant", "moving"])
def test_the_damping_is_the_rayleigh_gradient_of_the_frozen_velocity_and_dissipates(arm, rest):
    """u_d = −Pᵀ J_πᵀ C J_π P θ̇ and θ̇ᵀ u_d = −‖ḋ‖²_C ≤ 0 (R = ½ ḋᵀ C ḋ, ḋ = J_π P θ̇)."""
    kin, read = vmc.Kinematics(arm), reads(arm)[rest]
    controller = ua.controller(virtual(arm, spring=False, compensation=False), "frozen", rest=rest)
    rng = np.random.default_rng(2)
    for _ in range(6):
        theta, rate = rng.uniform(-0.6, 0.6, 2), rng.uniform(-2.0, 2.0, 2)
        q, v, P = frozen_state(read, theta, rate)
        J = np.asarray(kin.jacobian(q, "tip"))[:3]
        expected = -P.T @ J.T @ (DAMPING * np.eye(3)) @ J @ v
        u = torque(controller, theta, rate)
        np.testing.assert_allclose(u, expected, atol=1e-10)
        d_dot = J @ v
        assert rate @ u == pytest.approx(-DAMPING * d_dot @ d_dot, abs=1e-10)
        assert rate @ u <= 1e-12


def lyapunov(arm, read, controller, compensated=True, reduced=False):
    """E = T + V + P(π(θ)) along a simulation: the robot's energy and the virtual potential at the
    frozen configuration, whose gradient the torque is. The weight compensation is the gradient of
    −V_g(π(θ)) for a constant rest and of −V(π(θ)), the robot's whole potential at the balance,
    for a moving one (``reduced``)."""
    kin, dynamics = vmc.Kinematics(arm), vmc.compile_dynamics(arm)
    flat = build("three")
    flat.params["gravity"].value = np.zeros(3)
    structural = vmc.compile_dynamics(flat)
    live, B = dynamics.live_values(), np.asarray(arm.actuation.params["B"].value)

    def potential(theta):
        q = frozen_state(read, theta, [0.0, 0.0])[0]
        total = float(dynamics.energy(q, np.zeros(3), live, 0.0)[1])
        gravity = total - float(structural.energy(q, np.zeros(3), structural.live_values(), 0.0)[1])
        spring = 0.5 * STIFFNESS * np.sum((np.asarray(kin.position(q, "tip")).ravel() - GOAL) ** 2)
        if not compensated:
            return spring
        return spring - (
            total if reduced else gravity
        )  # the spring's energy and the compensation's

    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(5e-4), T=1.0).arrays()
    energy = []
    for q, v in zip(log["q"][::10], log["v"][::10], strict=True):
        T, V = dynamics.energy(q, v, live, 0.0)
        energy.append(float(T) + float(V) + potential(B.T @ q))
    return np.array(energy)


@pytest.mark.parametrize("rest", ["constant", "moving"])
def test_the_energy_of_the_closed_loop_never_rises(arm, rest):
    compiled = virtual(arm)
    controller = ua.controller(compiled, "frozen", rest=rest)
    read = reads(arm)[rest]
    energy = lyapunov(arm, read, controller, reduced=rest == "moving")
    assert np.diff(energy).max() < 1e-6  # J: it never rises
    assert energy[-1] < energy[0] - 1.0  # and it is spent


@pytest.mark.parametrize("rest", ["wrong model", "two steps"])
def test_the_springs_and_dampers_stay_passive_when_the_balance_is_wrong(arm, rest):
    """Passive for any smooth rest: a model with a passive joint 30 % too soft, or two Newton
    steps only, move the rest but not the gradient structure."""
    compiled = virtual(arm, compensation=False)
    if rest == "wrong model":
        soft = planar.add_dynamics(
            planar.arm("three-link"), stiffness=0.7 * planar.PASSIVE_STIFFNESS
        )
        read = ua.Moving(soft.actuation, vmc.compile_dynamics(soft))
    else:
        read = ua.Moving(arm.actuation, vmc.compile_dynamics(arm), iterations=2)
    energy = lyapunov(arm, read, StateController(compiled, [], read=read), compensated=False)
    assert np.diff(energy).max() < 1e-6
    assert energy[-1] < energy[0] - 0.1


def tip_error(arm, q):
    kin = vmc.Kinematics(arm)
    return 1000 * np.linalg.norm(np.asarray(kin.position(q, "tip")).ravel()[:2] - GOAL[:2])


def reach(arm, controller, T=4.0):
    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    return vmc.sim.run(plant, controller, vmc.sim.SimClock(5e-4), T=T).arrays()["q"][-1]


def test_the_moving_rest_reaches_the_goal_where_the_plain_rest_stops_short(arm):
    compiled = virtual(arm)
    plain = tip_error(arm, reach(arm, ua.controller(compiled, "frozen")))
    moving = tip_error(arm, reach(arm, ua.controller(compiled, "frozen", rest="moving")))
    assert plain > 20.0  # mm: the plain rest holds the passive joint straight, and it sags
    assert moving < 0.01  # mm: no pose to choose


def test_a_rest_balanced_at_the_end_pose_reaches_the_goal_too(arm):
    compiled = virtual(arm)
    end = reach(arm, ua.controller(compiled, "frozen", rest="moving"))
    balanced = ua.controller(
        compiled, "frozen", at=np.asarray(arm.actuation.params["B"].value).T @ end
    )
    assert tip_error(arm, reach(arm, balanced)) < 0.01


def test_a_plan_with_the_rest_free_finds_the_balanced_rest(arm):
    compiled = virtual(arm)
    start = reach(arm, ua.controller(compiled, "frozen"), T=2.0) + 0.05
    errors, gradients = {}, {}
    for name, free in (("fixed", ()), ("free", ("arm.q_rest",))):
        problem = opt.Problem(compiled.system)  # reads the motors: the frozen law
        problem.add(opt.Equilibrium(start))
        problem.add(opt.Cost(arm.point("tip") - GOAL, 1.0, name="reach"))
        problem.free(*free)
        result = problem.solve()
        errors[name] = tip_error(arm, result.q[0])
        if free:
            dynamics = vmc.compile_dynamics(arm)
            pose = result.q[0].copy()
            pose[1] = np.asarray(result.params["arm.q_rest"]).ravel()[1]
            gradients[name] = float(
                np.asarray(dynamics.residual(pose, np.zeros(3), np.zeros(3), np.zeros(2),
                                             dynamics.live_values(), 0.0)).ravel()[1]
            )  # fmt: skip
    assert errors["fixed"] > 20.0 and errors["free"] < 0.01  # mm
    assert abs(gradients["free"]) < 1e-3  # the rest balances the passive joint at the plan's pose


def test_the_elements_of_the_moving_rest_add_up_to_its_command(arm):
    controller = ua.controller(virtual(arm), "frozen", rest="moving")
    meas = vmc.Signals(0.0, motor_position=THETA, motor_velocity=np.array([0.3, -0.2]))
    controller.reset(0.0, meas)
    command = controller.step(0.0, meas)
    shares = sum(parts["torque"] for parts in controller.elements().values())
    np.testing.assert_allclose(shares, command["law_torque"], atol=1e-10)
