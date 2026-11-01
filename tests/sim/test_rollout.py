"""Rollouts: the closed loop compiled with mapaccum, against run, exact solutions and SciPy."""

import time

import casadi as ca
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import virtualmodelcontrol as vmc
from small_plan import DAMPING, FRICTION, MASS, STIFFNESS, exact_position, mass_spring
from virtualmodelcontrol.models import JointSpace
from virtualmodelcontrol.robots import helyx

DT = 1 / 330


def soft_arm():
    """The soft arm held at a point by its tip, with gravity compensation, and a start."""
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - [0.08, 0.0, 0.68], 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return vmc.VirtualMechanismSystem(arm, ctrl), np.linspace(0.0, 0.02, 9)


def virtual_mass():
    """The mass-spring pulled to its goal through a virtual mass on a spring and a damper."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, MASS))
    robot.add("friction", vmc.LinearDamper(x, FRICTION))
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("phi", unit="m")
    ctrl.add("virtual_mass", vmc.Inertance(phi, 0.5))
    ctrl.add("tie", vmc.LinearSpring(x - phi, 20.0))
    ctrl.add("tie_damper", vmc.LinearDamper(x - phi, 2.0))
    ctrl.add("pull", vmc.LinearSpring(phi - vmc.Ref("goal", 1, [1.0]), STIFFNESS))
    ctrl.add("drag", vmc.LinearDamper(phi, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl)


def run_arrays(system, q0, T, dt, v0=None, z0=None):
    """The log of vmc.sim.run on the same system."""
    plant = vmc.sim.ModelPlant(system.robot, q0=q0, v0=v0)
    controller = vmc.VMCController(vmc.compile(system))
    return vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=T, z0=z0).arrays()


def sampled_position(dt, steps):
    """Exact x at the start of each step of the mass-spring whose force is held for dt."""
    A = np.array([[0, 1, 0], [0, -FRICTION / MASS, 1 / MASS], [0, 0, 0]])
    step, s, out = expm(dt * A), np.zeros(2), []
    for _ in range(steps):
        out.append(s[0])
        force = -STIFFNESS * (s[0] - 1.0) - DAMPING * s[1]
        s = (step @ [s[0], s[1], force])[:2]
    return np.array(out)


def test_the_rollout_reproduces_run_on_the_soft_arm():
    system, q0 = soft_arm()
    log = run_arrays(system, q0, 1.0, DT)
    r = vmc.sim.rollout(system, q0, 1.0, DT)
    assert r["q"].shape == (330, 9) and r["u"].shape == (330, 9) and "z" not in r
    np.testing.assert_allclose(r["t"], DT * np.arange(330), atol=1e-12)
    for mine, theirs in (("q", "q"), ("v", "v"), ("u", "motor_torque")):
        error = np.abs(r[mine] - log[theirs]).max()
        print(f"soft arm, rollout against run: max |{mine}| difference {error:.2e}")
        np.testing.assert_allclose(r[mine], log[theirs], atol=1e-8)
    assert np.abs(r["v"]).max() > 1e-3  # it moves


def test_the_rollout_reproduces_run_with_a_virtual_state():
    system = virtual_mass()
    start = {"v0": [0.3], "z0": [0.2, -0.1]}
    log = run_arrays(system, [0.0], 2.0, 0.01, **start)
    r = vmc.sim.rollout(system, [0.0], 2.0, 0.01, **start)
    for mine, theirs in (("q", "q"), ("v", "v"), ("u", "motor_torque")):
        np.testing.assert_allclose(r[mine], log[theirs], atol=1e-8)
    # run logs z after the step's update, the rollout at the start of the step
    np.testing.assert_allclose(r["z"][0], [0.2, -0.1])
    np.testing.assert_allclose(r["z"][1:], log["z"][:-1], atol=1e-8)
    assert r["z"][-1, 0] > 0.5  # the virtual mass moved


def test_rk4_is_fourth_order():
    system, _, _ = mass_spring()
    sampled = sampled_position(0.1, 20)
    errors = [
        np.abs(
            vmc.sim.rollout(system, [0.0], 2.0, 0.1, integrator="rk4", max_step=h)["q"][:, 0]
            - sampled
        ).max()
        for h in (0.1, 0.05, 0.025)
    ]
    assert 12 < errors[0] / errors[1] < 20 and 12 < errors[1] / errors[2] < 20, errors
    # the sampled solution is the exact one when the controller is sampled fast
    fast = sampled_position(1e-3, 2000)
    np.testing.assert_allclose(fast, exact_position(1e-3 * np.arange(2000)), atol=2e-3)


def test_cvodes_agrees_with_a_very_small_implicit_step():
    system, _, _ = mass_spring()
    small = vmc.sim.rollout(system, [0.0], 2.0, 0.01, max_step=1e-5)["q"][:, 0]
    q = vmc.sim.rollout(system, [0.0], 2.0, 0.01, integrator="cvodes")["q"][:, 0]
    np.testing.assert_allclose(q, small, atol=1e-4)  # the implicit step is first order
    np.testing.assert_allclose(q, sampled_position(0.01, 200), atol=1e-6)


def test_ode_agrees_with_the_rollout_and_the_exact_solution():
    system, _, _ = mass_spring()
    times = 0.002 * np.arange(1000)
    kwargs = {"t_eval": times, "method": "Radau", "rtol": 1e-9, "atol": 1e-12}
    sol = solve_ivp(vmc.sim.ode(system), (0.0, times[-1]), [0.0, 0.0], **kwargs)
    np.testing.assert_allclose(sol.y[0], exact_position(times), atol=1e-6)
    r = vmc.sim.rollout(system, [0.0], 2.0, 0.002)
    np.testing.assert_allclose(sol.y[0], r["q"][:, 0], atol=1e-3)

    system = virtual_mass()  # x, v, then the virtual state's position and velocity
    times = 0.001 * np.arange(2000)
    kwargs["t_eval"] = times
    sol = solve_ivp(vmc.sim.ode(system), (0.0, times[-1]), np.zeros(4), **kwargs)
    r = vmc.sim.rollout(system, [0.0], 2.0, 0.001)
    np.testing.assert_allclose(sol.y[0], r["q"][:, 0], atol=1e-3)
    np.testing.assert_allclose(sol.y[2], r["z"][:, 0], atol=1e-2)


def test_the_rollout_is_differentiable_in_the_live_params():
    system, _, _ = mass_spring()
    runtime = ["*stiffness*"]
    compiled = vmc.compile(system, runtime)
    p0, where = compiled.live_values(), compiled.live_slices()["ctrl.spring.stiffness"]

    def final(p):
        return vmc.sim.rollout(system, [0.0], 1.0, 0.01, runtime=runtime, p=p)["q"][-1, 0]

    p = ca.MX.sym("p", p0.size)
    f = ca.Function("final", [p], [final(p)])
    gradient = float(ca.Function("g", [p], [ca.gradient(final(p), p)[where]])(p0))
    h = 1e-4
    up, down = p0.copy(), p0.copy()
    up[where] += h
    down[where] -= h
    central = (float(final(up)) - float(final(down))) / (2 * h)
    assert float(f(p0)) == pytest.approx(float(final(p0)), abs=1e-12)
    assert abs(gradient) > 1e-3
    assert gradient == pytest.approx(central, rel=1e-5)


def test_given_torques_replace_the_controller_and_reproduce_a_logged_run():
    system, q0 = soft_arm()
    log = run_arrays(system, q0, 1.0, DT)
    r = vmc.sim.rollout(system, q0, 1.0, DT, u=log["motor_torque"])
    np.testing.assert_array_equal(
        r["u"], log["motor_torque"]
    )  # the controller's command is not used
    np.testing.assert_allclose(r["q"], log["q"], atol=1e-8)
    np.testing.assert_allclose(r["v"], log["v"], atol=1e-8)
    assert np.abs(log["motor_torque"]).max() > 1e-3
    passive = vmc.sim.rollout(system, q0, 1.0, DT, u=np.zeros((330, 9)))
    assert np.abs(passive["q"] - log["q"]).max() > 1e-4  # whatever the controller would have done


def test_given_torques_need_a_row_per_step():
    system, _, _ = mass_spring()
    with pytest.raises(ValueError, match="100 steps"):
        vmc.sim.rollout(system, [0.0], 1.0, 0.01, u=np.zeros((99, 1)))


def test_an_unknown_integrator_is_refused():
    system, _, _ = mass_spring()
    with pytest.raises(ValueError, match="integrator"):
        vmc.sim.rollout(system, [0.0], 1.0, 0.01, integrator="euler")


def test_the_rollout_is_faster_than_run_on_the_soft_arm():
    system, q0 = soft_arm()
    start = time.perf_counter()
    plant = vmc.sim.ModelPlant(system.robot, q0=q0)
    controller = vmc.VMCController(vmc.compile(system))
    built = time.perf_counter()
    vmc.sim.run(plant, controller, vmc.sim.SimClock(DT), T=5.0)
    ran = time.perf_counter()
    vmc.sim.rollout(system, q0, 5.0, DT)
    rolled = time.perf_counter()
    vmc.sim.rollout(system, q0, DT, DT)  # one step: what is left is the compiling
    compiled = time.perf_counter()
    print(f"soft arm, 5 s: run {ran - built:.2f} s, plus {built - start:.2f} s to compile")
    print(f"soft arm, 5 s: rollout {rolled - ran:.2f} s, {compiled - rolled:.2f} s of it compiling")


def arm_reach(stiffness=30.0, damping=6.0):
    """The three-link arm, one passive joint, pulled to a point by a spring on its tip."""
    from virtualmodelcontrol.robots import planar

    arm = planar.add_dynamics(planar.arm("three-link"))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - [0.45, 0.15, 0.0], stiffness))
    ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), damping))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.VirtualMechanismSystem(arm, ctrl)


def test_the_rollout_of_a_robot_with_fewer_motors_reads_the_motors_as_run_does():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = arm_reach()
    q0, T, dt = np.array([0.3, 0.3, 0.3]), 0.5, 1 / 200

    def followed(controller):
        plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=1e-3)
        return vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=T).arrays()["q"]

    r = vmc.sim.rollout(system, q0, T, dt)["q"]
    frozen = followed(vmc.VMCController(vmc.compile(system)))
    naive = followed(ua.controller(vmc.compile(system), "naive"))
    np.testing.assert_allclose(r, frozen, atol=1e-8)  # what VMCController does
    assert np.abs(r - naive).max() > 0.01  # and not the whole-state controller


def test_the_rollout_applies_the_passive_correction_as_the_controller_does():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = arm_reach(150.0, 3.0)  # underdamped: the arm gets more than its dampers take
    q0, T, dt = np.array([0.3, 0.3, 0.3]), 0.5, 1 / 200
    compiled = vmc.compile(system)
    stage = ua.Passivation(vmc.compile_dynamics(arm, actuation=system.actuation), width=1e-2)
    plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=1e-3)
    ran = vmc.sim.run(plant, vmc.VMCController(compiled, [stage]), vmc.sim.SimClock(dt), T=T)
    rolled = vmc.sim.rollout(system, q0, T, dt, output=[stage])
    plain = vmc.sim.rollout(system, q0, T, dt)
    np.testing.assert_allclose(rolled["q"], ran.arrays()["q"], atol=1e-6)
    assert np.abs(rolled["q"] - plain["q"]).max() > 1e-3  # the correction does something here


def test_the_rollout_has_the_three_controllers_of_an_underactuated_robot():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = arm_reach()
    q0, T, dt = np.array([0.3, 0.3, 0.3]), 0.5, 1 / 200
    compiled = vmc.compile(system)

    def followed(controller):
        plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=1e-3)
        return vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=T).arrays()["q"]

    cases = {
        "naive": ({"reads": "state"}, ua.controller(compiled, "naive")),
        "frozen": ({}, ua.controller(compiled, "frozen")),
        "moving": ({"rest": "moving"}, ua.controller(compiled, "frozen", rest="moving")),
    }
    paths = {}
    for name, (options, controller) in cases.items():
        paths[name] = vmc.sim.rollout(system, q0, T, dt, **options)["q"]
        np.testing.assert_allclose(paths[name], followed(controller), atol=1e-6, err_msg=name)
    assert np.abs(paths["moving"] - paths["frozen"]).max() > 1e-3  # a different rest
    assert np.abs(paths["naive"] - paths["frozen"]).max() > 1e-3
    f = vmc.sim.ode(system, rest="moving")
    assert np.isfinite(f(0.0, np.concatenate([q0, np.zeros(3)]))).all()


def test_the_moving_rest_is_for_the_frozen_controller_of_a_robot_with_fewer_motors():
    _, system = arm_reach()
    q0 = np.array([0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match="rest is"):
        vmc.sim.rollout(system, q0, 0.1, 0.01, rest="floating")
    with pytest.raises(ValueError, match="reads is"):
        vmc.sim.rollout(system, q0, 0.1, 0.01, reads="both")
    with pytest.raises(ValueError, match="fewer motors"):
        vmc.sim.rollout(system, q0, 0.1, 0.01, reads="state", rest="moving")
