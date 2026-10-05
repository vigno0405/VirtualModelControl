"""Structure optimization: gates on elements, and a sparsity term that keeps few of them."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from soft_arm_plan import HORIZON, NODES, TRANSITION
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import JointSpace
from virtualmodelcontrol.robots import helyx

STIFFNESS = (1.0, 2.0, 4.0)  # three springs to the same goal: the stiffest does the most per gate


def candidates():
    """A mass pulled to x = 1 by three gated springs, all open halfway."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, 1.0))
    robot.add("friction", vmc.LinearDamper(x, 1.0))
    ctrl = vmc.Mechanism("ctrl")
    for i, k in enumerate(STIFFNESS):
        ctrl.add(f"spring{i}", vmc.Gated(vmc.LinearSpring(x - 1.0, k), 0.5))
    ctrl.add("damper", vmc.LinearDamper(x, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def plan(weight=None):
    system, x = candidates()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 5.0, 41))
    problem.free("ctrl.*.gate")
    problem.add(opt.Effort(1e-2))
    problem.add(opt.Cost(x - 1.0, name="reach"))
    if weight is not None:
        problem.add(opt.Sparsity(weight, "ctrl.*.gate"))
    return problem.solve()


def gates(result):
    return np.array([v.item() for k, v in result.params.items() if k.endswith(".gate")])


def test_without_the_sparsity_term_the_gates_stay_spread():
    r = plan()
    assert r.converged and (gates(r) > 0.1).all()
    assert "sparsity" not in r.costs


def test_with_it_the_gates_of_the_weak_springs_close():
    r = plan(0.1)
    assert r.converged
    np.testing.assert_allclose(gates(r), [0.0, 0.0, 1.0], atol=1e-6)  # the stiffest does it all


def test_a_lighter_weight_keeps_more():
    kept = [(gates(plan(w)) > 1e-4).sum() for w in (1e-3, 0.1)]
    assert kept == [2, 1]


def test_the_term_is_the_weighted_sum_of_the_gates():
    r = plan(1e-2)
    assert r.costs["sparsity"] == pytest.approx(1e-2 * gates(r).sum(), rel=1e-12)


def test_the_term_needs_free_params_that_cannot_be_negative():
    system, _ = candidates()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 2.0, 11))
    problem.add(opt.Sparsity(1.0, "ctrl.*.gate"))
    with pytest.raises(ValueError, match="no free Param matches"):
        problem.build()  # nothing was freed
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 2.0, 11))
    problem.free("ctrl.spring0.stiffness")
    problem.params["ctrl.spring0.stiffness"].bounds = (-1.0, 5.0)
    problem.add(opt.Sparsity(1.0, "ctrl.spring0.stiffness"))
    with pytest.raises(ValueError, match="can be negative"):
        problem.build()


def soft_arm_fields(weight=None):
    """The hanging arm reaches past a sphere with a tip spring and five gated repulsive fields."""
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    tip, goal, ball = arm.point(s=1.0), [0.20, 0.0, 0.66], [0.15, 0.0, 0.50]
    hold = vmc.Mechanism("hold")
    hold.add("drag", vmc.LinearSpring(tip - [0.01, 0.0, 0.70], 10.0))
    hold.add("damp", vmc.LinearDamper(tip, 2.0))
    hold.add("gravity", vmc.GravityCompensation(arm))
    held = vmc.VirtualMechanismSystem(arm, hold)
    new = vmc.Mechanism("new")
    k = vmc.Param("stiffness", 100.0, bounds=(1.0, 300.0), scope="stage")
    new.add("pull", vmc.TanhSpring(tip - goal, k, 2.0))
    new.add("damp", vmc.LinearDamper(tip, 2.0))
    for i, s in enumerate((0.3, 0.45, 0.6, 0.75, 0.9)):
        field = vmc.GaussianSpring(arm.point(s=s) - ball, 200.0, 0.06)
        new.add(f"field{i}", vmc.Gated(field, 0.5))
    new.add("gravity", vmc.GravityCompensation(arm))
    plant = vmc.sim.ModelPlant(arm)
    vmc.sim.run(plant, vmc.VMCController(vmc.compile(held)), vmc.sim.SimClock(1 / 330), T=12.0)
    problem = opt.Problem(vmc.VirtualMechanismSystem(arm, new))
    problem.add(opt.Collocation(plant.q, HORIZON, NODES, initial=held, transition=TRANSITION))
    problem.free("new.pull.stiffness", "new.field*.gate")
    problem.add(opt.Effort(0.2))
    problem.add(opt.Cost(tip - goal, t_from=TRANSITION, name="reach"))
    body = [vmc.SphereDistance(arm.point(s=s), ball, 0.02) for s in np.linspace(0.0, 1.0, 41)]
    problem.add(opt.Bound(vmc.Stack(*body), lower=0.02, name="clear"))
    if weight is not None:
        problem.add(opt.Sparsity(weight, "new.field*.gate"))
    return problem.solve()


def test_the_arm_keeps_few_fields_when_they_cost():
    free, sparse = soft_arm_fields(), soft_arm_fields(1e-3)
    assert free.converged and sparse.converged
    closed = [(g < 1e-4).sum() for g in map(gates, (free, sparse))]
    print(f"gates closed: {closed[0]} without the term, {closed[1]} with it")
    print(f"gates: {gates(free).round(3)} and {gates(sparse).round(3)}")
    assert closed[1] >= 3 and closed[1] > closed[0]
    assert (gates(sparse) > 1e-4).any()  # the obstacle is still avoided, by the ones that stay
