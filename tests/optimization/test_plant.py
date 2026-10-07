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
