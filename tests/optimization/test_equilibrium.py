"""The equilibrium block: the closed loop at rest, found with the free Params."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import JointSpace
from virtualmodelcontrol.robots import helyx

FORCE, STIFFNESS, GOAL = -2.0, 4.0, "ctrl.spring.goal"


def pushed_mass():
    """A mass held by a spring to x = 1 while a constant force pushes it: it rests at 1 + F / k."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, 1.0))
    robot.add("friction", vmc.LinearDamper(x, 1.0))
    robot.add("push", vmc.ForceSource(x, FORCE))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(x - vmc.Ref("goal", 1, [1.0]), STIFFNESS))
    ctrl.add("damper", vmc.LinearDamper(x, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def problem(*terms, free=()):
    system, x = pushed_mass()
    p = opt.Problem(system)
    p.add(opt.Equilibrium([0.0]))
    p.free(*free)
    for term in terms:
        p.add(term(x))
    return p


def test_the_closed_loop_rests_where_the_controller_balances_the_robots_forces():
    r = problem().solve()
    assert r.converged and r.violation < 1e-9
    assert r.q.shape == (1, 1) and r.t.tolist() == [0.0] and r.blend.tolist() == [1.0]
    assert r.q[0, 0] == pytest.approx(1.0 + FORCE / STIFFNESS, abs=1e-9)  # 0.5
    assert r.v[0, 0] == 0.0 and r.a[0, 0] == 0.0
    assert r.u[0, 0] == pytest.approx(-FORCE)  # the spring pulls back what the force pushes


def test_a_free_goal_is_found_that_puts_the_equilibrium_where_it_is_wanted():
    r = problem(lambda x: opt.Cost(x - 0.3, name="reach"), free=[GOAL]).solve()
    assert r.converged and r.q[0, 0] == pytest.approx(0.3, abs=1e-6)
    assert r.params[GOAL][0] == pytest.approx(0.3 - FORCE / STIFFNESS, abs=1e-5)  # 0.8
    assert r.costs["reach"] < 1e-10


def test_a_cost_and_an_effort_count_the_one_node_once():
    reach = problem(lambda x: opt.Cost(x - 0.0, 3.0, name="reach"), lambda x: opt.Effort(5.0))
    r = reach.solve()
    q, u = r.q[0, 0], r.u[0, 0]
    assert r.costs["reach"] == pytest.approx(3.0 * q**2)
    assert r.costs["effort"] == pytest.approx(5.0 * u**2)


def test_a_bound_holds_at_the_equilibrium():
    free = problem(lambda x: opt.Cost(x - 0.8, name="reach"), free=[GOAL]).solve()
    capped = problem(
        lambda x: opt.Cost(x - 0.8, name="reach"), lambda x: opt.Bound(x, upper=0.4), free=[GOAL]
    ).solve()
    assert free.q[0, 0] == pytest.approx(0.8, abs=1e-5)
    assert capped.converged and capped.q[0, 0] == pytest.approx(0.4, abs=1e-6)


def test_a_free_stiffness_and_a_parameter_goal_share_the_program():
    p = problem(lambda x: opt.Cost(x - 0.6, name="reach"))
    p.parameter(GOAL)
    r1, r2 = p.solve({GOAL: [1.0]}), p.solve({GOAL: [2.0]})
    assert r1.q[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert r2.q[0, 0] == pytest.approx(1.5, abs=1e-6)  # the equilibrium follows the goal


def soft_arm():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    return arm, arm.point(s=1.0)


def holding(arm, tip, goal):
    """The arm held by a spring from its tip to ``goal``, gravity compensated."""
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - vmc.Ref("goal", 3, goal), 80.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return vmc.VirtualMechanismSystem(arm, ctrl)


def settle(arm, system, T=20.0):
    plant = vmc.sim.ModelPlant(arm)
    controller = vmc.VMCController(vmc.compile(system))
    vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=T)
    return plant


def test_the_equilibrium_of_the_soft_arm_is_where_a_long_run_comes_to_rest():
    arm, tip = soft_arm()
    system = holding(arm, tip, [0.12, 0.0, 0.55])
    p = opt.Problem(system)
    p.add(opt.Equilibrium(np.zeros(9)))
    r = p.solve()
    assert r.converged and r.violation < 1e-4
    plant = settle(arm, system)
    assert np.abs(plant.v).max() < 1e-6  # it has come to rest
    np.testing.assert_allclose(r.q[0], plant.q, atol=1e-4)  # and where the equilibrium is
    short = vmc.Kinematics(arm).position(r.q[0], 1.0) - [0.12, 0.0, 0.55]
    assert np.linalg.norm(short) > 0.02  # the arm's own stiffness holds the tip short of the goal


def test_the_goal_that_gives_an_equilibrium_is_found_and_the_arm_settles_there():
    arm, tip = soft_arm()
    kin = vmc.Kinematics(arm)
    forward = opt.Problem(holding(arm, tip, [0.12, 0.0, 0.55]))
    forward.add(opt.Equilibrium(np.zeros(9)))
    target = kin.position(forward.solve().q[0], 1.0)  # a tip position that the arm can hold

    other = holding(arm, tip, [0.0, 0.0, 0.70])  # start the search from another goal
    inverse = opt.Problem(other)
    inverse.add(opt.Equilibrium(np.zeros(9)))
    inverse.free("ctrl.reach.goal")
    inverse.add(opt.Cost(tip - target, name="reach"))
    r = inverse.solve()
    assert r.converged and r.costs["reach"] < 1e-8
    np.testing.assert_allclose(r.params["ctrl.reach.goal"], [0.12, 0.0, 0.55], atol=2e-3)

    controller = vmc.VMCController(vmc.compile(other))
    r.apply(controller)  # the goal that was found
    plant = vmc.sim.ModelPlant(arm)
    vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=20.0)
    assert np.linalg.norm(kin.position(plant.q, 1.0) - target) < 1e-3  # the tip rests on the target


def test_the_arguments_and_the_blocks_are_checked():
    system, _ = pushed_mass()
    with pytest.raises(ValueError, match="scale"):
        opt.Equilibrium([0.0], scale=0.0)
    wrong = opt.Problem(system)
    wrong.add(opt.Equilibrium([0.0, 0.0]))
    with pytest.raises(ValueError, match="q0 needs 1 entries"):
        wrong.build()
    twice = opt.Problem(system)
    twice.add(opt.Equilibrium([0.0]))
    with pytest.raises(ValueError, match="one Collocation or Equilibrium"):
        twice.add(opt.Collocation([0.0], 1.0, 5))
    with pytest.raises(ValueError, match="one Collocation or Equilibrium"):
        twice.add(opt.Equilibrium([0.0]))
    with pytest.raises(ValueError, match="Collocation or Equilibrium before the term"):
        opt.Problem(system).add(opt.Effort(1.0))
