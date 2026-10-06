"""Underactuated control: what the motors realize, and the physics of the controllers."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from underactuated_lab import build, law, measurement
from virtualmodelcontrol.control import StateController
from virtualmodelcontrol.control import underactuated as ua
from virtualmodelcontrol.models import JointSpace

B3 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


def test_projector_defect_and_feasible_forces():
    E = ua.projector(B3)
    np.testing.assert_allclose(E, np.diag([0, 1, 0]), atol=1e-12)
    J = np.array([[-0.5, -0.3, -0.1], [0.4, 0.2, 0.1]])
    F = np.array([2.0, -1.0])
    np.testing.assert_allclose(ua.defect(B3, J, F), E @ J.T @ F)
    np.testing.assert_allclose(ua.defect(B3, J, F, torque=[0, 0.7, 0]), [0, (J.T @ F)[1] - 0.7, 0])
    basis = ua.feasible(B3, J)  # one direction: the second joint sees a single task direction
    assert basis.shape == (2, 1)
    np.testing.assert_allclose(ua.defect(B3, J, 3.0 * basis[:, 0]), 0, atol=1e-12)
    assert abs(ua.defect(B3, J, (J @ E)[:, 1])).max() > 1e-3  # a wrench off the set
    # a passive joint the task cannot see leaves every wrench feasible
    blind = np.array([[-0.5, 0.0, -0.1], [0.4, 0.0, 0.1]])
    assert ua.feasible(B3, blind).shape == (2, 2)
    assert ua.feasible(np.eye(3)[:, :2], J).shape == (2, 1)  # E has rank one here, J E rank one


def test_direction_gain_closed_form_and_floor():
    K = np.diag([100.0, 50.0, 10.0])
    n = [0.0, 1.0, 0.0]
    assert ua.direction_gain(1.0, 2.0, 5.0, K, n) == pytest.approx(2.0 * 4.0 / (4.0 + 1e-6))
    floor = 0.95 * (-50.0)  # K' = K + s n n^T loses positivity at s = -50
    assert ua.direction_gain(1.0, 2.0, -1e6, K, n) == pytest.approx(floor)
    assert ua.direction_gain(1.0, 0.0, 9.0, K, n) == pytest.approx(0.0)  # no response: no change
    scaled = ua.direction_gain(1.0, 2.0, 5.0, K, [0.0, 7.0, 0.0])  # any length of direction
    assert scaled == pytest.approx(ua.direction_gain(1.0, 2.0, 5.0, K, n))


def arm_with_law(gravity=False, g=None):
    robot = build("three")
    if g is not None:
        robot.params["gravity"].value = g
    return robot, law("three", robot, gravity)


def torque_at(compiled, q, v):
    """The torques of the virtual elements at (q, v), and the realized ones (I - E) tau."""
    tau = np.array(compiled.tau(q, v, np.zeros(0), compiled.live_values(), 0.0)).ravel()
    return tau, tau - ua.projector(B3) @ tau


def test_the_frozen_torque_is_a_gradient_and_the_naive_one_is_not():
    robot, compiled = arm_with_law()
    rng = np.random.default_rng(3)
    q = rng.uniform(-0.8, 0.8, 3)
    frozen = ua.Frozen(robot.actuation)

    def realized(x, evaluate_at):
        return torque_at(compiled, evaluate_at(x), np.zeros(3))[1]

    def jacobian(function):
        h = 1e-6
        return np.column_stack([(function(q + h * e) - function(q - h * e)) / (2 * h)
                                for e in np.eye(3)])  # fmt: skip

    naive = jacobian(lambda x: realized(x, lambda y: y))
    pi = jacobian(lambda x: realized(x, lambda y: frozen.point(B3.T @ y)))
    assert abs(pi - pi.T).max() < 1e-6 * abs(pi).max()  # exact gradient: symmetric
    assert abs(naive - naive.T).max() > 1e-2 * abs(naive).max()  # not a gradient


def test_frozen_and_naive_agree_on_the_self_balance_manifold():
    robot, compiled = arm_with_law()
    q = np.array([0.5, 0.0, -0.2])  # the passive joint is at its rest
    naive, frozen = (ua.controller(compiled, b) for b in ("naive", "frozen"))
    meas = measurement(robot, q, np.zeros(3))
    np.testing.assert_allclose(
        naive.step(0.0, meas)["law_torque"], frozen.step(0.0, meas)["law_torque"], atol=1e-12
    )
    q[1] = 0.4  # off it: they differ
    meas = measurement(robot, q, np.zeros(3))
    assert abs(naive.step(0.0, meas)["law_torque"] - frozen.step(0.0, meas)["law_torque"]).max() > 1


def test_the_frozen_controller_needs_only_the_motors():
    robot, compiled = arm_with_law()
    frozen = ua.controller(compiled, "frozen")
    plain = vmc.Signals(0.0, motor_position=[0.2, -0.1], motor_velocity=[0.3, 0.0])
    full = measurement(robot, np.array([0.2, 0.9, -0.1]), np.array([0.3, 5.0, 0.0]))
    np.testing.assert_allclose(
        frozen.step(0.0, plain)["law_torque"], frozen.step(0.0, full)["law_torque"], atol=1e-12
    )
    assert "q" not in plain  # and the controller ran


def test_a_passive_joint_left_free_swings_and_keeps_its_energy():
    robot, drift = planar_free(), []
    for step in (1e-4, 2.5e-5):
        plant = vmc.sim.ModelPlant(robot, q0=[0.0, 0.6, 0.0], max_step=step)
        start = plant.energy()
        for _ in range(300):  # no torque: the spring swings the joint and the arm answers
            plant.advance(1e-3)
        drift.append(abs(plant.energy() - start) / start)
    assert abs(plant.q[1]) < 0.6 and abs(plant.q[0]) > 1e-3  # the actuated joint is dragged along
    assert drift[0] < 0.05 and drift[1] < 0.5 * drift[0]  # the integrator's, falling with its step


def planar_free():
    from virtualmodelcontrol.robots import planar

    robot = planar.add_dynamics(planar.arm(gravity=(0.0, 0.0, 0.0)), damping=0.0)
    return robot


def run_naive(correction, steps=2500, dt=2e-4, **flags):
    robot, compiled = arm_with_law()
    controller = ua.controller(compiled, "naive", correction, **flags)
    plant = vmc.sim.ModelPlant(robot, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    dynamics = vmc.compile_dynamics(robot)
    p = dynamics.live_values()
    controller.reset(0.0, plant.read())
    energy = []
    for _ in range(steps):
        plant.write(controller.step(plant.t, plant.read()))
        plant.advance(dt)
        T, V = dynamics.energy(plant.q, plant.v, p, plant.t)
        energy.append(float(T) + float(V))
    return np.array(energy), controller


def test_the_passive_correction_keeps_the_robot_from_gaining_energy():
    free, _ = run_naive(None)
    passive, controller = run_naive("passive")
    assert free.max() - free[0] > 0.5  # the uncorrected controller injects energy
    assert np.diff(passive).max() < 1e-3  # the robot's own energy never rises
    assert controller.output[0].alpha >= 0.0


def test_the_tank_conserves_the_robot_energy_plus_the_tank_level():
    total, controller = run_naive("tank", tank=1.0)
    level = controller.output[0].level
    # E + T: the tank is charged by what the dampers took and drained by what the motors injected
    energy, _ = run_naive("tank", steps=2500, tank=1.0)
    assert energy[-1] + level == pytest.approx(energy[0] + 1.0, abs=0.05)
    assert total[0] > 0


def test_passivation_bounds_the_injected_power_and_leaves_a_dissipative_command_alone():
    robot = arm_with_law()[0]
    dynamics = vmc.compile_dynamics(robot)
    stage = ua.Passivation(dynamics, width=1e-6, regularizer=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(300):
        q, v = rng.uniform(-1, 1, 3), rng.uniform(-2, 2, 3)
        meas = measurement(robot, q, v, 0.0)
        u = rng.uniform(-50, 50, 2)
        rate = meas["motor_velocity"]
        stage.reset()
        out = stage(u, meas)
        assert rate @ out <= stage.dissipation + 1e-6  # the guarantee
        if rate @ u < stage.dissipation - 1e-3:  # already passive: untouched
            np.testing.assert_allclose(out, u, atol=1e-9)
    assert stage.dissipation >= 0.0


def test_the_tank_gate_and_level():
    robot = arm_with_law()[0]
    stage = ua.Passivation(vmc.compile_dynamics(robot), tank=0.0)
    q, v = np.array([0.2, 0.5, 0.1]), np.array([0.4, -0.6, 0.3])
    u = np.array([3.0, -2.0])
    out = stage(u, measurement(robot, q, v, 0.0))
    assert stage.gate == pytest.approx(1.0)  # an empty tank corrects in full
    drawn = stage.dissipation - measurement(robot, q, v)["motor_velocity"] @ out
    stage(u, measurement(robot, q, v, 0.01))
    assert stage.level == pytest.approx(0.01 * drawn)
    stage.level = 5.0
    stage(u, measurement(robot, q, v, 0.02))
    assert stage.gate < 1e-3  # a full tank lets the command through
    stage.reset()
    assert stage.level == 0.0
    assert ua.Passivation(vmc.compile_dynamics(robot)).level is None


def test_the_controller_resets_its_output_stages():
    compiled = arm_with_law()[1]
    controller = ua.controller(compiled, "frozen", "tank", tank=2.0)
    controller.output[0].level = 9.0
    controller.reset(0.0)
    assert controller.output[0].level == 2.0


def test_gravity_moves_the_frozen_point_to_the_passive_joints_balance():
    robot = arm_with_law()[0]
    dynamics = vmc.compile_dynamics(robot)
    plain, loaded = ua.Frozen(robot.actuation), ua.Frozen(robot.actuation, dynamics)
    theta = [0.4, -0.3]
    assert plain.point(theta)[1] == 0.0
    point = loaded.point(theta)
    gradient = np.array(dynamics.residual(point, np.zeros(3), np.zeros(3), np.zeros(2),
                                          dynamics.live_values(), 0.0)).ravel()  # fmt: skip
    assert abs(gradient[1]) < 1e-8 and abs(point[1]) > 1e-2  # sagging under its own weight
    np.testing.assert_allclose(point[[0, 2]], theta, atol=1e-12)


def test_a_state_controller_is_the_motor_controller_when_every_joint_has_a_motor():
    robot = vmc.Mechanism("r", model=JointSpace(2))
    ctrl = vmc.Mechanism("c")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)) - [0.5, -0.5], [20.0, 30.0]))
    # a virtual mass as well, so the virtual state z and its balance exist
    ctrl.add("m", vmc.Inertance(ctrl.add_state("z", 1), 0.2))
    ctrl.add("link", vmc.LinearSpring(robot.joint(0) - ctrl.states["z"], 5.0))
    ctrl.add("damp", vmc.LinearDamper(ctrl.states["z"], 1.0))
    ramp = vmc.Custom(lambda t: 0.3 * t, [vmc.Time()], dim=1, unit="rad")  # moves with time
    ctrl.add("ramp", vmc.LinearSpring(robot.joint(1) - ramp, 7.0))
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    motors, states = vmc.VMCController(compiled), StateController(compiled)
    for c in (motors, states):
        c.reset(5.0)  # time in the law counts from the reset
    rng = np.random.default_rng(5)
    for t in 5.0 + np.arange(1, 6) * 0.01:
        q, v = rng.normal(size=2), rng.normal(size=2)
        meas = vmc.Signals(t, motor_position=q, motor_velocity=v, q=q, v=v)
        a, b = motors.step(t, meas), states.step(t, meas)
        np.testing.assert_allclose(a["motor_torque"], b["motor_torque"], atol=1e-12)
    np.testing.assert_allclose(motors.z, states.z, atol=1e-12)
    np.testing.assert_allclose(motors.energy(), states.energy(), rtol=1e-12)
    for key, value in motors.balance().items():
        assert states.balance()[key] == pytest.approx(value, rel=1e-12, abs=1e-12)
    for name, parts in motors.elements().items():
        for key, value in parts.items():
            np.testing.assert_allclose(states.elements()[name][key], value, atol=1e-12)
    jump = states.set({"c.hold.stiffness": [40.0, 30.0]})
    assert jump == pytest.approx(motors.set({"c.hold.stiffness": [40.0, 30.0]}), rel=1e-12)
    assert states.read(meas)[0] is meas["q"]


def test_the_frozen_controller_is_passive_on_the_energy_balance():
    robot, compiled = arm_with_law()
    frozen = ua.controller(compiled, "frozen")
    plant = vmc.sim.ModelPlant(robot, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    log = vmc.sim.run(plant, frozen, vmc.sim.SimClock(2e-4), T=0.5, record=["energy"])
    balance = vmc.sim.energy_balance(log)
    assert balance["margin"].min() > -1e-3  # it never gives more than it holds
    assert abs(balance["injected"]).max() < 5e-3


def test_directional_force_tracking_reaches_its_force_and_keeps_the_stiffness_positive():
    robot = build("three")
    compiled = law("three", robot, matrix=True)
    controller = ua.controller(compiled, "naive")
    tracker = ua.DirectionalForce(
        controller, "tip", "ctrl.reach.stiffness", [1, 0, 0], 3.0, rate=2.0
    )
    q, v = np.array([0.5, 0.3, 0.2]), np.zeros(3)
    meas = measurement(robot, q, v)
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    for _ in range(4000):
        tracker.step(controller, q, v, 1e-2)
    c0, cv = tracker._pieces(controller, q, v)
    assert tracker.reading == pytest.approx(3.0, abs=1e-2)  # less the regularizer's bias
    assert c0 + tracker.s * cv == pytest.approx(tracker.reading, rel=1e-12)
    K = np.reshape(controller.live_params()["ctrl.reach.stiffness"], (3, 3))
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > 0
    np.testing.assert_allclose(K, 150.0 * np.eye(3) + tracker.s * np.diag([1.0, 0, 0]))
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        ua.DirectionalForce(ua.controller(law("three", robot), "naive"), "tip",
                            "ctrl.reach.stiffness", [1, 0, 0], 3.0)  # fmt: skip


def test_the_force_floor_keeps_the_stiffness_positive_for_any_force():
    robot = build("three")
    controller = ua.controller(law("three", robot, matrix=True), "naive")
    tracker = ua.DirectionalForce(controller, "tip", "ctrl.reach.stiffness", [0, 1, 0], 0.0)
    q, v = np.array([0.5, 0.3, 0.2]), np.zeros(3)
    controller.reset(0.0, measurement(robot, q, v))
    controller.step(0.0, measurement(robot, q, v))
    c0, cv = tracker._pieces(controller, q, v)
    tracker.force = c0 - 1e5 * np.sign(cv)  # asks for a huge drop of stiffness
    for _ in range(500):
        tracker.step(controller, q, v, 1e-1)
    K = np.reshape(controller.live_params()["ctrl.reach.stiffness"], (3, 3))
    assert np.linalg.eigvalsh(K).min() > 0
    assert tracker.s == pytest.approx(0.95 * -150.0, rel=1e-3)


def test_the_frozen_point_holds_the_passive_joint_at_its_rest_only():
    from virtualmodelcontrol.robots import planar

    robot = planar.arm(rest=[0.3, 0.4, 0.5])  # only the passive joint's rest counts
    point = ua.Frozen(robot.actuation).point([0.1, -0.2])
    np.testing.assert_allclose(point, [0.1, 0.4, -0.2], atol=1e-12)


def test_the_correction_width_scales_the_ramp():
    robot = arm_with_law()[0]
    q, v, u = np.array([0.2, 0.5, 0.1]), np.array([0.4, -0.6, 0.3]), np.array([30.0, -20.0])
    meas = measurement(robot, q, v, 0.0)
    stage = ua.Passivation(vmc.compile_dynamics(robot), width=0.5, regularizer=0.01)
    command = stage(u, meas)
    rate = meas["motor_velocity"]
    excess = rate @ u - stage.dissipation
    alpha = 0.5 * np.log1p(np.exp(excess / 0.5)) / (rate @ rate + 0.01)
    np.testing.assert_allclose(command, u - alpha * rate, atol=1e-10)


def test_the_floor_of_the_stiffness_does_not_depend_on_the_length_of_the_direction():
    K = np.diag([100.0, 50.0, 10.0])
    assert ua.direction_gain(1.0, 2.0, -1e6, K, [0.0, 7.0, 0.0]) == pytest.approx(0.95 * -50.0)


def test_a_long_step_stops_at_the_stiffness_floor():
    robot = build("three")
    controller = ua.controller(law("three", robot, matrix=True), "naive")
    tracker = ua.DirectionalForce(controller, "tip", "ctrl.reach.stiffness", [0, 1, 0], 0.0)
    q, v = np.array([0.5, 0.3, 0.2]), np.zeros(3)
    controller.reset(0.0, measurement(robot, q, v))
    controller.step(0.0, measurement(robot, q, v))
    c0, cv = tracker._pieces(controller, q, v)
    tracker.force = c0 - 1e5 * np.sign(cv)
    tracker.step(controller, q, v, 100.0)  # rate dt = 20: an overshoot without the floor
    assert tracker.s == pytest.approx(0.95 * -150.0, rel=1e-12)


def test_a_state_controller_reports_its_elements_and_energy_on_an_underactuated_robot():
    robot, compiled = arm_with_law()
    controller = ua.controller(compiled, "naive")
    meas = measurement(robot, np.array([0.3, 0.2, 0.1]), np.array([0.5, -0.4, 0.2]))
    controller.reset(0.0, meas)
    command = controller.step(0.0, meas)
    shares = sum(parts["torque"] for parts in controller.elements().values())
    np.testing.assert_allclose(shares, command["law_torque"], atol=1e-10)  # as motor torques
    assert set(controller.balance()) == {"stored", "kinetic", "port", "dissipation", "source"}
    plant = vmc.sim.ModelPlant(robot, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1e-3), T=0.01, record=["energy"])
    assert np.isfinite(vmc.sim.energy_balance(log)["energy"]).all()
