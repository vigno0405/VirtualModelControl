"""TankBudget: the plan keeps the tank's level the way the real Tank does."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import tanh_mass
from virtualmodelcontrol import optimization as opt

K, GOAL = "ctrl.spring.stiffness", "ctrl.spring.goal"
INTERVALS, SUBSTEPS, HORIZON = 8, 4, 1.6


def program(level, refill=True, budget=True):
    """A mass pulled to 1 by a spring whose stiffness and reference step; a tank pays for it."""
    system, x = tanh_mass(stiffness=2.0, max_force=5.0, goal=0.0, bounds=(0.5, 50.0))
    problem = opt.Problem(system, solver="ipopt-exact")
    problem.add(opt.Shooting([0.0], HORIZON, INTERVALS + 1, steps=[K, GOAL], substeps=SUBSTEPS))
    problem.add(opt.Effort(0.01))
    problem.add(opt.Cost(x - 1.0, 1.0, name="reach"))
    if budget:
        problem.add(opt.TankBudget(level, refill=refill))
    return system, problem


def execute(system, plan, level, refill=True):
    """The plan on a plant, the controller's Params set through a Tank at the start of each
    interval; returns the tank's level after each change and the fraction it applied."""
    h = HORIZON / INTERVALS / SUBSTEPS
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=h)
    controller = vmc.VMCController(vmc.compile(system))
    tank = vmc.control.Tank(controller, level=level)
    controller.reset(0.0, plant.read())
    levels, fractions = [], []
    for i in range(INTERVALS * SUBSTEPS):
        meas = plant.read()
        cmd = tank.step(plant.t, meas) if refill else controller.step(plant.t, meas)
        if i % SUBSTEPS == 0:
            plan.apply(tank, interval=i // SUBSTEPS)
            levels.append(tank.level)
            fractions.append(tank.fraction)
            cmd = tank.step(plant.t, meas) if refill else controller.step(plant.t, meas)
        plant.write(cmd)
        plant.advance(h)
    return np.array(levels), np.array(fractions)


@pytest.mark.parametrize("refill", [True, False])
def test_the_planned_level_is_the_level_of_the_real_tank_executing_the_plan(refill):
    system, problem = program(0.05, refill)
    plan = problem.solve()
    levels, fractions = execute(system, plan, 0.05, refill)
    nlp = problem.build()
    x = problem.initial_guess(plan)
    x[nlp.variables.slices["level:tank"]] = levels  # the tank's own, in the plan's equations
    g = np.array(nlp.functions()[1](x, np.zeros(0))).ravel()
    assert plan.converged and plan.violation < 1e-6
    assert np.abs(g[nlp.constraints["tank"]]).max() < 1e-7
    assert np.abs(g[nlp.constraints["continuity"]]).max() < 1e-7
    assert (fractions == 1.0).all()  # the tank never had to cut a step
    assert levels.min() > -1e-9 and levels.min() < 1e-4  # and the budget was used up


def test_without_the_term_the_same_tank_cuts_the_steps_of_the_plan():
    system, problem = program(0.05, budget=False)
    plan = problem.solve()
    levels, fractions = execute(system, plan, 0.05)
    assert fractions.min() < 0.9
    assert levels.min() > -1e-9  # the tank keeps its budget by cutting


def test_the_refill_from_the_dampers_lets_a_plan_do_more():
    cost = {}
    for refill in (True, False):
        _, problem = program(0.05, refill)
        cost[refill] = problem.solve().cost
    assert cost[True] < cost[False] - 1e-3


def test_the_level_now_is_a_parameter_and_more_budget_buys_a_better_plan():
    _, problem = program(0.05)
    problem.parameter("tank.level")
    poor, rich = problem.solve({"tank.level": 0.02}), problem.solve({"tank.level": 2.0})
    assert poor.references["tank.level"].item() == 0.02
    assert rich.cost < poor.cost - 1e-2
    assert problem.build() is problem.build()


def test_a_negative_level_has_no_plan():
    _, problem = program(-1.0)
    assert not problem.solve().converged


def test_the_term_needs_steps_of_a_shooting():
    system, x = tanh_mass()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 1.0, 5))
    problem.add(opt.TankBudget(1.0))
    with pytest.raises(ValueError, match="needs a Shooting"):
        problem.build()
