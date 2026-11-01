"""A controller written for the motors of a larger robot: the plan is what the simulator does."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import FRICTION, MASS
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import Direct, JointSpace, Underactuated
from virtualmodelcontrol.robots import turtle

DT = 1 / turtle.CONTROL_RATE


def flywheel(robot, omega=6.0, depth=0.5):
    """The turtle's controller: a flywheel with a spring to each crank, written for the cranks."""
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, 0.24))
    ctrl.add("drive", vmc.SpeedRegulator(phi, 0.3, omega, 0.5))
    behind = vmc.Ref("delta", 1, value=np.pi, unit="rad")
    for i, (phase, side) in enumerate([(phi, 1.0), (phi - behind, -1.0)]):
        e = robot.joint(i) - phase
        spring = vmc.PhaseSpring(vmc.Stack(e, phase), 1.0, depth=depth, side=side)
        ctrl.add(f"spring{i}", spring)
        ctrl.add(f"damper{i}", vmc.LinearDamper(e, 0.06))
    return ctrl


# Solved to 1e-10, so that a plan is compared with the simulation and not with where the solver
# stops: under the default options the end point differs by platform (on macOS 1e-4 from the
# simulation in v, on Linux 4e-6), under these it is 5e-12 on Linux.
TIGHT = {"ipopt.tol": 1e-10, "ipopt.acceptable_iter": 0, "ipopt.constr_viol_tol": 1e-10}


def crawl(steps, warm=False, depth=0.5):
    """The crawler run by the simulator under the controller of the cranks, and the same run
    planned (from rest, or from the simulated run itself with ``warm``): the log, the plan and
    the problem."""
    cranks = turtle.robot()
    system = vmc.VirtualMechanismSystem(cranks, flywheel(cranks, depth=depth))
    q0 = [0, 0, 0.04, 1, 0, 0, 0, 0.0, -np.pi]
    body = turtle.crawler()
    plant = vmc.sim.ModelPlant(body, q0=q0, max_step=DT)
    controller = vmc.VMCController(vmc.compile(system))
    log = vmc.sim.run(
        plant, controller, vmc.sim.SimClock(DT), T=(steps + 1) * DT, z0=turtle.initial_state
    ).arrays()
    problem = opt.Problem(system, plant=body, options=TIGHT)
    block = opt.Shooting(q0, steps * DT, steps + 1, v0=np.zeros(8), z0=[0.0, 0.0], running=False)
    problem.add(block)
    guess = None
    if warm:  # the log holds the flywheel after its update at each step: the next node's
        z = np.vstack([[0.0, 0.0], np.asarray(log["z"])[:steps]])
        guess = {
            "q": np.asarray(log["q"])[: steps + 1],
            "v": np.asarray(log["v"])[: steps + 1],
            "z": z,
        }
    return log, problem.solve(warm_start=guess), problem


@pytest.mark.parametrize("steps, warm", [(25, False), (60, True)])
def test_a_plan_of_the_crawler_under_the_flywheel_is_the_simulation_step_for_step(steps, warm):
    log, plan, problem = crawl(steps, warm)
    assert plan.converged and plan.violation < 1e-8
    q, v, z = (np.asarray(log[key])[:steps] for key in ("q", "v", "z"))
    assert np.ptp(q[:, 7]) > 5e-5 and np.abs(z[:, 1]).max() > 0.02  # the cranks turn, it spins
    assert np.abs(q[:, 0]).max() > 1e-6  # and the body moves
    np.testing.assert_allclose(plan.q[:-1], q, atol=1e-7)
    np.testing.assert_allclose(plan.v[:-1], v, atol=1e-6)
    np.testing.assert_allclose(plan.z[1 : steps + 1], z, atol=1e-7)
    np.testing.assert_allclose(plan.u[:-1], np.asarray(log["law_torque"])[:steps], atol=1e-7)
    norms = np.linalg.norm(plan.q[:, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)  # the nodes are on the manifold
    assert "manifold" in problem.build().constraints


def test_the_terms_of_a_plan_on_the_plant_are_in_its_coordinates():
    _, plan, problem = crawl(10)
    x = problem.plant.joint(0)  # the body's x: a coordinate of the plant, not of the cranks' robot
    problem.add(opt.Cost(x - 1.0, 1.0, name="distance"))
    again = problem.solve(warm_start=plan)
    assert again.converged and again.costs["distance"] == pytest.approx(10 * DT, rel=0.05)


def line(actuation=None, name="plant"):
    """Two masses joined by a spring, with a motor on the first (or the ``actuation``): the
    plant."""
    robot = vmc.Mechanism(
        name, model=JointSpace(2, unit="m"), actuation=actuation or Underactuated.joints(2, [0])
    )
    first, second = robot.joint(0), robot.joint(1)
    robot.add("m1", vmc.Inertance(first, MASS))
    robot.add("m2", vmc.Inertance(second, 2 * MASS))
    robot.add("link", vmc.LinearSpring(second - first, 30.0))
    robot.add("rub", vmc.LinearDamper(robot.joint(slice(0, 2)), FRICTION))
    return robot


def hand(plant_robot, link=12.0):
    """A controller for the motor alone: a mass with a virtual spring, in its own robot."""
    own = vmc.Mechanism("hand", model=JointSpace(1, unit="m"))
    ctrl = vmc.Mechanism("ctrl")
    z = ctrl.add_state("z", 1, unit="m")
    ctrl.add("mass", vmc.Inertance(z, 0.3))
    ctrl.add("link", vmc.LinearSpring(own.joint(0) - z, link))
    ctrl.add("anchor", vmc.LinearSpring(z - 0.4, 6.0))
    ctrl.add("damper", vmc.LinearDamper(z, 1.5))
    ctrl.add("brake", vmc.LinearDamper(own.joint(0), 2.0))
    return vmc.VirtualMechanismSystem(own, ctrl)


def test_a_controller_that_reads_the_motors_of_an_underactuated_plant_is_planned_as_it_runs():
    plant, system = line(), hand(line())
    sim = vmc.sim.ModelPlant(plant, q0=[0.0, 0.0], max_step=0.01)
    controller = vmc.VMCController(vmc.compile(system))
    log = vmc.sim.run(
        sim, controller, vmc.sim.SimClock(0.01), T=1.0, z0=lambda meas: np.array([0.0, 0.0])
    ).arrays()
    problem = opt.Problem(system, plant=plant)
    problem.add(opt.Shooting([0.0, 0.0], 1.0, 101, z0=[0.0, 0.0], running=False))
    plan = problem.solve()
    assert plan.converged and np.abs(np.asarray(log["q"])[:, 1]).max() > 1e-3  # both masses move
    np.testing.assert_allclose(plan.q[:-1], np.asarray(log["q"]), atol=1e-6)
    np.testing.assert_allclose(plan.v[:-1], np.asarray(log["v"]), atol=1e-5)
    np.testing.assert_allclose(plan.z[1:], np.asarray(log["z"]), atol=1e-6)
    collocation = opt.Problem(system, plant=plant)
    collocation.add(opt.Collocation([0.0, 0.0], 1.0, 41, scheme="hermite-simpson"))
    smooth = collocation.solve()
    reference = vmc.sim.rollout(system, [0.0], 1.0, 0.001, max_step=0.001)  # the controller alone
    assert smooth.converged and smooth.z.shape == (41, 2) and reference is not None
    assert np.abs(smooth.q[:, 1] - np.interp(smooth.t, plan.t, plan.q[:, 1])).max() < 5e-3


def test_a_plant_that_is_a_copy_of_the_robot_gives_the_plan_of_the_robot_itself():
    """The motors of a copy reach the controller through its transmission and the program is the
    same one: the law, the energy of a tank and the dampers' power all come out equal."""

    def solved(split):
        system, x, _ = _held()
        problem = opt.Problem(system, plant=_held()[0].robot if split else None)
        problem.add(opt.Shooting([0.0], 1.6, 9, steps=["ctrl.link.stiffness"], substeps=3))
        problem.add(opt.Cost(x - 1.0, 1.0, name="reach"))
        problem.add(opt.Effort(0.01))
        problem.add(opt.TankBudget(0.05))
        return problem.solve()

    plain, split = solved(False), solved(True)
    assert plain.converged and split.converged
    np.testing.assert_allclose(split.q, plain.q, atol=1e-6)
    np.testing.assert_allclose(split.z, plain.z, atol=1e-6)
    np.testing.assert_allclose(split.u, plain.u, atol=1e-5)
    assert abs(split.cost - plain.cost) < 1e-6 * (1 + abs(plain.cost))


def _held():
    """A mass on a line held through a virtual mass whose link stiffness steps."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, MASS))
    robot.add("friction", vmc.LinearDamper(x, FRICTION))
    ctrl = vmc.Mechanism("ctrl")
    z = ctrl.add_state("z", 1, unit="m")
    ctrl.add("inertia", vmc.Inertance(z, 0.5))
    link = vmc.Param("stiffness", 8.0, bounds=(1.0, 60.0), scope="stage")
    ctrl.add("link", vmc.LinearSpring(x - z, link))
    ctrl.add("anchor", vmc.LinearSpring(z - 1.0, 3.0))
    ctrl.add("damper", vmc.LinearDamper(z, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), x, ctrl


def test_an_equilibrium_and_a_collocation_run_on_a_plant_too():
    for build in (lambda: opt.Equilibrium([0.2]), lambda: opt.Collocation([0.2], 2.0, 21)):
        system, _, _ = _held()
        own = opt.Problem(system)
        own.add(build())
        other = opt.Problem(system, plant=_held()[0].robot)
        other.add(build())
        a, b = own.solve(), other.solve()
        assert a.converged and b.converged
        np.testing.assert_allclose(b.q, a.q, atol=1e-6)
        np.testing.assert_allclose(b.z, a.z, atol=1e-6)


def test_the_controller_and_the_plant_must_have_the_same_number_of_motors():
    problem = opt.Problem(hand(line()), plant=line(Direct(), name="both"))
    problem.add(opt.Shooting([0.0, 0.0], 1.0, 5))
    with pytest.raises(ValueError, match="commands 1 motors and the plant 'both' has 2"):
        problem.build()


def test_the_builder_reads_the_motors_of_the_plant_for_the_law_the_energy_and_the_dampers():
    from virtualmodelcontrol.optimization.builder import Builder

    plant, system = line(), hand(line())
    problem = opt.Problem(system, plant=plant)
    builder = Builder(system, problem.params, [], [], plant)
    compiled = builder.compiled
    p, t = compiled.live_values(), 0.7
    q, v, z = np.array([0.3, -0.2]), np.array([0.5, 0.1]), np.array([0.25, -0.3])
    own = [np.array([0.3]), np.array([0.5])]  # what the controller sees: its motor's angle and rate
    u, zdot = (np.array(x).ravel() for x in builder.command(compiled, q, v, z, p, t))
    want_u, want_zdot = (np.array(x).ravel() for x in compiled.law(*own, z, p, t))
    np.testing.assert_allclose([*u, *zdot], [*want_u, *want_zdot], rtol=1e-12)
    assert np.abs(u).max() > 0.1 and np.abs(zdot).max() > 0.1
    stored = float(builder.stored(compiled, q, v, z, p, t))
    assert stored == pytest.approx(sum(float(x) for x in compiled.energy(*own, z, p, t)), rel=1e-12)
    taken = float(builder.dissipation(compiled, q, v, z, p, t))
    assert taken == pytest.approx(float(compiled.power(*own, z, p, t)[1]), rel=1e-12)
    assert stored > 0.05 and taken < -0.1
    robot_own = Builder(system, problem.params, [], [], None)  # no plant: the robot is the plant
    assert robot_own.plant is system.robot


def test_an_equilibrium_of_the_plant_rests_the_controller_that_reads_its_motors():
    plant, system = line(), hand(line())
    problem = opt.Problem(system, plant=plant)
    problem.add(opt.Equilibrium([0.0, 0.0]))
    plan = problem.solve()
    assert plan.converged
    np.testing.assert_allclose(plan.q, [[0.4, 0.4]], atol=1e-7)  # both masses at the anchor
    np.testing.assert_allclose(plan.z, [[0.4, 0.0]], atol=1e-7)


def test_a_tank_on_a_plant_counts_the_energy_of_the_controller_at_the_motors_of_the_plant():
    import casadi as ca

    name = "ctrl.link.stiffness"
    link = vmc.Param("stiffness", 12.0, bounds=(2.0, 60.0), scope="stage")
    plant, system = line(), hand(line(), link)
    problem = opt.Problem(system, solver="ipopt-exact", plant=plant)
    problem.add(opt.Shooting([0.0, 0.0], 1.6, 9, steps=[name], substeps=2, running=False))
    problem.add(opt.Cost(plant.joint(0) - 0.4, 1.0, name="reach"))
    problem.add(opt.Effort(0.01))
    problem.add(opt.TankBudget(0.05))
    plan = problem.solve()
    nlp = problem.build()
    x = problem.initial_guess(plan)
    taken = ca.Function("taken", [nlp.x, nlp.p], nlp.trajectory.dissipated)(x, np.zeros(0))
    controller = vmc.VMCController(vmc.compile(system, runtime=[name]))
    levels, level = [], 0.05
    for k in range(8):  # the real controller at the motors of the plan's node k, and its step
        meas = vmc.Signals(0.0, motor_position=plan.q[k, :1], motor_velocity=plan.v[k, :1])
        controller.reset(0.0, meas, z0=plan.z[k])
        jump = controller.set({name: plan.steps[name][k]})
        level = level + (float(np.array(taken).ravel()[k - 1]) if k else 0.0) - jump
        levels.append(level)
    x[nlp.variables.slices["level:tank"]] = levels
    g = np.array(nlp.functions()[1](x, np.zeros(0))).ravel()
    assert plan.converged and np.abs(g[nlp.constraints["tank"]]).max() < 1e-7
    assert np.abs(plan.steps[name] - 12.0).max() > 0.1  # the stiffness did step


def _arm_reach():
    from virtualmodelcontrol.robots import planar

    arm = planar.add_dynamics(planar.arm("three-link"))
    tip, goal = arm.point("tip"), [0.45, 0.15, 0.0]
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - goal, 30.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 6.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.VirtualMechanismSystem(arm, ctrl)


def _violation(reads, make_controller, rest="constant", sub=10, plant=None):
    """How far a simulation of the arm's reach is from the plan's constraints (the closed loop's
    dynamics at every step) under the reading ``reads``: the plan is not solved, since a cold
    start from rest converges on one machine and not on another, but the simulation is put in as
    the starting point and the solver stopped there."""
    from virtualmodelcontrol.control import underactuated as ua  # noqa: F401

    arm, system = _arm_reach()
    q0, nodes = np.array([0.3, 0.3, 0.3]), 6
    h = 0.5 / (nodes - 1) / sub
    simulated = vmc.sim.ModelPlant(arm, q0=q0, max_step=h)
    log = vmc.sim.run(
        simulated, make_controller(vmc.compile(system)), vmc.sim.SimClock(h), T=0.5 + 2 * h
    ).arrays()
    start = {"q": np.asarray(log["q"])[::sub][:nodes], "v": np.asarray(log["v"])[::sub][:nodes]}
    problem = opt.Problem(
        system, plant=plant, reads=reads, rest=rest, options={"ipopt.max_iter": 0}
    )
    problem.add(opt.Shooting(q0, 0.5, nodes, substeps=sub))
    return problem.solve(warm_start=start).violation


def test_a_plan_of_an_underactuated_robot_reads_the_motors_like_the_controller_does():
    from virtualmodelcontrol.control import underactuated as ua

    def frozen(compiled):
        return vmc.VMCController(compiled)

    def naive(compiled):
        return ua.controller(compiled, "naive")

    # a simulation of a controller satisfies the plan that reads what that controller reads...
    assert _violation(None, frozen) < 1e-4  # the default: what VMCController does
    assert _violation("motors", frozen) < 1e-4
    assert _violation("state", naive) < 1e-8  # the whole state, as a StateController
    # ...and breaks the plan that reads the other thing
    assert _violation("state", frozen) > 1e-2
    assert _violation(None, naive) > 1e-2


def test_reads_has_two_values_and_the_state_needs_the_robots_own_plant():
    _, system = _arm_reach()
    with pytest.raises(ValueError, match="'state' or 'motors'"):
        opt.Problem(system, reads="both")
    problem = opt.Problem(system, plant=_arm_reach()[0], reads="state")
    problem.add(opt.Equilibrium([0.3, 0.3, 0.3]))
    with pytest.raises(ValueError, match="reads its motors"):
        problem.solve()


def test_a_term_can_use_a_site_of_the_plant_when_the_plant_is_another_robot():
    _, system = _arm_reach()
    plant, _ = _arm_reach()
    problem = opt.Problem(system, plant=plant)
    problem.add(opt.Equilibrium([0.3, 0.3, 0.3]))
    problem.add(opt.Cost(plant.point("tip") - [0.45, 0.15, 0.0], 1.0, name="reach"))
    result = problem.solve()
    assert result.converged and np.isfinite(result.costs["reach"])


def test_a_plan_can_include_the_passive_correction_and_matches_the_controller_that_has_it():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = _arm_reach()
    ctrl = vmc.Mechanism("ctrl")  # underdamped, so that the arm gets more than its dampers take
    ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - [0.45, 0.15, 0.0], 150.0))
    ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), 3.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    stage = ua.Passivation(vmc.compile_dynamics(arm, actuation=system.actuation), width=1e-2)
    q0, nodes, sub = np.array([0.3, 0.3, 0.3]), 6, 10
    h = 0.5 / (nodes - 1) / sub

    def violation(reads, output, controller):
        plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=h)
        log = vmc.sim.run(plant, controller, vmc.sim.SimClock(h), T=0.5 + 2 * h).arrays()
        start = {"q": np.asarray(log["q"])[::sub][:nodes], "v": np.asarray(log["v"])[::sub][:nodes]}
        problem = opt.Problem(system, reads=reads, output=output, options={"ipopt.max_iter": 0})
        problem.add(opt.Shooting(q0, 0.5, nodes, substeps=sub))
        return problem.solve(warm_start=start).violation

    compiled = vmc.compile(system)
    corrected = ua.controller(compiled, "frozen", "passive", width=1e-2)
    assert violation(None, [stage], corrected) < 1e-4  # the plan has what the controller has
    assert violation(None, [], corrected) > 1e-3  # and without it the plan is another one
    naive = ua.controller(compiled, "naive", "passive", width=1e-2)
    assert violation("state", [stage], naive) < 1e-4  # the whole-state controller too


def test_the_correction_in_a_plan_uses_the_dampers_of_the_plan_not_the_ones_stored():
    from virtualmodelcontrol.control import underactuated as ua
    from virtualmodelcontrol.robots import planar

    def build(damping):
        arm = planar.add_dynamics(planar.arm("three-link"), damping=damping)
        ctrl = vmc.Mechanism("ctrl")
        ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - [0.45, 0.15, 0.0], 150.0))
        ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), 3.0))
        ctrl.add("gravity", vmc.GravityCompensation(arm))
        return arm, vmc.VirtualMechanismSystem(arm, ctrl)

    stored, real = build(0.4), build(2.0)  # the plan is made for 0.4 and told 2.0 at the solve
    q0, nodes, sub = np.array([0.3, 0.3, 0.3]), 6, 10
    h = 0.5 / (nodes - 1) / sub
    stage = ua.Passivation(
        vmc.compile_dynamics(stored[0], actuation=stored[1].actuation), width=1e-2
    )
    controller = ua.controller(vmc.compile(real[1]), "frozen", "passive", width=1e-2)
    plant = vmc.sim.ModelPlant(real[0], q0=q0, max_step=h)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(h), T=0.5 + 2 * h).arrays()
    start = {"q": np.asarray(log["q"])[::sub][:nodes], "v": np.asarray(log["v"])[::sub][:nodes]}
    problem = opt.Problem(stored[1], output=[stage], options={"ipopt.max_iter": 0})
    problem.add(opt.Shooting(q0, 0.5, nodes, substeps=sub))
    problem.parameter("arm.damper.damping")
    told = problem.solve({"arm.damper.damping": 2.0}, warm_start=start)
    assert told.violation < 1e-4
    unknowing = problem.solve({"arm.damper.damping": 0.4}, warm_start=start)
    assert unknowing.violation > 1e-3  # the dampers are part of what the plan computes


def test_the_moving_rest_is_planned_as_the_controller_runs_it():
    from virtualmodelcontrol.control import underactuated as ua

    def moving(compiled):
        return ua.controller(compiled, "frozen", rest="moving")

    def frozen(compiled):
        return ua.controller(compiled, "frozen")

    # finer steps than the others: the moving rest is stiffer than the integrator's 10 ms allows
    assert _violation(None, moving, "moving", 40) < 1e-4  # the plan is the closed loop
    assert _violation("motors", moving, "moving", 40) < 1e-4
    assert _violation(None, moving, "constant", 40) > 1e-3  # the constant rest is another plan
    assert _violation(None, frozen, "moving", 40) > 1e-3
    assert _violation(None, frozen, "constant", 40) < 1e-4


def test_the_moving_rest_is_for_a_controller_that_reads_the_motors_of_a_robot_with_fewer_motors():
    _, system = _arm_reach()
    with pytest.raises(ValueError, match="rest is"):
        opt.Problem(system, rest="floating")
    problem = opt.Problem(system, reads="state", rest="moving")
    problem.add(opt.Equilibrium([0.3, 0.3, 0.3]))
    with pytest.raises(ValueError, match="fewer motors"):
        problem.solve()
    direct = vmc.Mechanism("r", model=JointSpace(2), actuation=Direct())
    direct.add("m", vmc.PointMass(direct.joint(0), 1.0))
    ctrl = vmc.Mechanism("c")
    ctrl.add("hold", vmc.LinearSpring(direct.joint(slice(0, 2)) - [0.1, 0.2], 5.0))
    problem = opt.Problem(vmc.VirtualMechanismSystem(direct, ctrl), rest="moving")
    problem.add(opt.Equilibrium([0.0, 0.0]))
    with pytest.raises(ValueError, match="fewer motors"):
        problem.solve()


def test_the_moving_rest_is_planned_on_a_plant_that_is_a_copy_of_the_robot():
    """The controller balances its own model of the robot, and the plant gives it the motors."""
    from virtualmodelcontrol.control import underactuated as ua

    def moving(compiled):
        return ua.controller(compiled, "frozen", rest="moving")

    assert _violation(None, moving, "moving", 40, plant=_arm_reach()[0]) < 1e-4
    assert _violation(None, moving, "constant", 40, plant=_arm_reach()[0]) > 1e-3


def test_a_plan_with_the_moving_rest_has_the_command_and_the_virtual_state_of_the_controller():
    """A virtual mass between the tip and the goal: the plan reads it at the frozen configuration,
    as the controller does, not at the plant's state."""
    from virtualmodelcontrol.control import underactuated as ua
    from virtualmodelcontrol.optimization.builder import Builder
    from virtualmodelcontrol.robots import planar

    arm = planar.add_dynamics(planar.arm("three-link"))
    tip, ctrl = arm.point("tip"), vmc.Mechanism("ctrl")
    z = ctrl.add_state("z", 3, unit="m")
    ctrl.add("mass", vmc.Inertance(z, 0.2))
    ctrl.add("hand", vmc.LinearSpring(tip - z, 30.0))
    ctrl.add("pull", vmc.LinearSpring(z - [0.45, 0.15, 0.0], 20.0))
    ctrl.add("drag", vmc.LinearDamper(z, 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    builder = Builder(system, opt.Problem(system).params, [], [], rest="moving")
    compiled = builder.compiled
    q, v = np.array([0.3, 0.2, -0.1]), np.array([0.1, -0.2, 0.3])
    state = np.array([0.4, 0.1, 0.0, 0.2, -0.1, 0.0])
    p = compiled.live_values()
    u, zdot = (np.array(x).ravel() for x in builder.command(compiled, q, v, state, p, 0.0))
    B = np.asarray(arm.actuation.params["B"].value)
    meas = vmc.Signals(0.0, motor_position=B.T @ q, motor_velocity=B.T @ v)
    controller = ua.controller(compiled, "frozen", rest="moving")
    controller.reset(0.0, meas, z0=state)
    np.testing.assert_allclose(u, controller.step(0.0, meas)["law_torque"], atol=1e-9)
    qf, vf = controller.read(meas)
    expected = np.array(compiled.law(qf, vf, state, p, 0.0)[1]).ravel()
    np.testing.assert_allclose(zdot, expected, atol=1e-9)
    assert abs(expected - np.array(compiled.law(q, v, state, p, 0.0)[1]).ravel()).max() > 1e-3


def _underdamped_reach():
    """The arm under a spring that releases more energy than the dampers take."""
    arm, _ = _arm_reach()
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - [0.45, 0.15, 0.0], 150.0))
    ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), 3.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.VirtualMechanismSystem(arm, ctrl)


def test_the_passive_correction_is_planned_with_the_moving_rest():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = _underdamped_reach()
    stage = ua.Passivation(vmc.compile_dynamics(arm, actuation=system.actuation), width=1e-2)
    q0, nodes, sub = np.array([0.3, 0.3, 0.3]), 6, 40
    h = 0.5 / (nodes - 1) / sub

    def violation(output):
        compiled = vmc.compile(system)
        controller = ua.controller(compiled, "frozen", "passive", rest="moving", width=1e-2)
        plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=h)
        log = vmc.sim.run(plant, controller, vmc.sim.SimClock(h), T=0.5 + 2 * h).arrays()
        start = {"q": np.asarray(log["q"])[::sub][:nodes], "v": np.asarray(log["v"])[::sub][:nodes]}
        problem = opt.Problem(system, rest="moving", output=output, options={"ipopt.max_iter": 0})
        problem.add(opt.Shooting(q0, 0.5, nodes, substeps=sub))
        return problem.solve(warm_start=start).violation

    assert violation([stage]) < 1e-4  # the plan has what the controller has
    assert violation([]) > 1e-3  # and without the stage it is another controller


def test_a_collocation_and_an_mpc_plan_the_moving_rest():
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = _arm_reach()
    tip, q0 = arm.point("tip"), np.array([0.3, 0.3, 0.3])
    problem = opt.Problem(system, rest="moving")
    problem.add(opt.Collocation(q0, 0.5, 11))
    problem.add(opt.Cost(tip - [0.45, 0.15, 0.0], 1.0, name="reach"))
    result = problem.solve()
    assert result.converged and np.isfinite(result.q).all()
    # an MPC that plans the stiffness of the spring, applied to the controller that runs
    problem = opt.Problem(system, rest="moving")
    # 2.5 ms steps: with 10 ms ones the plan of either rest does not converge
    problem.add(opt.Shooting(q0, 0.5, 6, steps=["ctrl.reach.stiffness"], substeps=40))
    problem.add(opt.Cost(tip - [0.45, 0.15, 0.0], 1.0, name="reach"))
    problem.add(opt.Effort(0.01))
    mpc = opt.MPC(problem)
    controller = ua.controller(vmc.compile(system, runtime=["ctrl.reach.stiffness"]), "frozen",
                               rest="moving")  # fmt: skip
    B = np.asarray(arm.actuation.params["B"].value)
    meas = vmc.Signals(0.0, motor_position=B.T @ q0, motor_velocity=np.zeros(2))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    jump = mpc.step(controller, q0, np.zeros(3), latency=0.0)
    assert mpc.applied and np.isfinite(jump)


def test_the_moving_rest_of_a_plan_balances_with_the_robot_params_of_the_solve():
    """The passive spring's stiffness set at the solve, not at the build, places the balance."""
    from virtualmodelcontrol.control import underactuated as ua

    arm, system = _arm_reach()
    q0, nodes, sub = np.array([0.3, 0.3, 0.3]), 6, 40
    h = 0.5 / (nodes - 1) / sub
    problem = opt.Problem(system, rest="moving", options={"ipopt.max_iter": 0})
    problem.add(opt.Shooting(q0, 0.5, nodes, substeps=sub))
    problem.parameter("arm.spring.stiffness")
    problem.build()  # at the stiffness it has now
    stiffness = system.params["arm.spring.stiffness"]
    stiffness.value = 0.7 * np.asarray(stiffness.value)  # a softer passive joint, after the build
    controller = ua.controller(vmc.compile(system), "frozen", rest="moving")
    plant = vmc.sim.ModelPlant(arm, q0=q0, max_step=h)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(h), T=0.5 + 2 * h).arrays()
    start = {"q": np.asarray(log["q"])[::sub][:nodes], "v": np.asarray(log["v"])[::sub][:nodes]}
    assert problem.solve(warm_start=start).violation < 1e-4
