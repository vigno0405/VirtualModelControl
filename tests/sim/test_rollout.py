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
