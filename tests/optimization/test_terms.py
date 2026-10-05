"""The terms: the quadratures they promise, bounds, and writing your own."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import Rod
from small_plan import mass_spring, tanh_mass
from virtualmodelcontrol import optimization as opt

GOAL, SPRING = "ctrl.spring.goal", "ctrl.spring.stiffness"


def run(build, nodes=41, horizon=4.0, q0=0.0, free=()):
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([q0], horizon, nodes))
    for term in build(x):
        problem.add(term)
    problem.free(*free)
    return problem.solve()


def test_effort_is_the_trapezoid_integral_of_the_squared_torques():
    r = run(lambda x: [opt.Effort(0.3)])
    dt = r.t[1] - r.t[0]
    squared = r.u[:, 0] ** 2
    expected = 0.3 * dt * (0.5 * squared[0] + squared[1:-1].sum() + 0.5 * squared[-1])
    assert r.costs["effort"] == pytest.approx(expected, rel=1e-12)


def test_cost_counts_the_window_fully_and_the_last_node_by_half():
    r = run(lambda x: [opt.Cost(x - 1.0, 2.0, t_from=1.0, name="reach")])
    dt = r.t[1] - r.t[0]
    window = r.t >= 1.0 - 1e-9
    y2 = (r.q[window, 0] - 1.0) ** 2
    expected = 2.0 * dt * (y2[:-1].sum() + 0.5 * y2[-1])
    assert r.costs["reach"] == pytest.approx(expected, rel=1e-12)


def test_cost_window_ends_where_it_is_told_to():
    r = run(lambda x: [opt.Cost(x - 1.0, t_from=1.0, t_to=3.0, name="reach")])
    dt = r.t[1] - r.t[0]
    window = (r.t >= 1.0 - 1e-9) & (r.t <= 3.0 + 1e-9)
    y2 = (r.q[window, 0] - 1.0) ** 2
    assert r.costs["reach"] == pytest.approx(dt * (y2[:-1].sum() + 0.5 * y2[-1]), rel=1e-12)


def test_cost_weights_each_entry_of_the_coordinate():
    def terms(x):
        return [opt.Cost(vmc.Stack(x, x - 1.0), [1.0, 10.0], name="both")]

    r = run(terms)
    dt = r.t[1] - r.t[0]
    both = r.q[:, 0] ** 2 + 10.0 * (r.q[:, 0] - 1.0) ** 2
    assert r.costs["both"] == pytest.approx(dt * (both[:-1].sum() + 0.5 * both[-1]), rel=1e-12)


def test_a_window_without_nodes_is_an_error():
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 4.0, 41))
    problem.add(opt.Cost(x - 1.0, t_from=5.0))
    with pytest.raises(ValueError, match="no node lies in the window of 'cost'"):
        problem.build()


def test_a_bound_holds_at_every_node_but_the_first():
    # The mass starts above the bound; node 0 is the fixed start, so the bound skips it. The free
    # spring reference has to push the mass under the bound at once.
    def terms(x):
        return [opt.Cost(x - 1.0, name="reach"), opt.Bound(x, upper=0.5, name="ceiling")]

    r = run(terms, q0=0.52, free=[GOAL])
    assert r.converged and r.q[0, 0] == 0.52
    assert r.q[1:, 0].max() <= 0.5 + 1e-6
    assert r.q[1:, 0].max() == pytest.approx(0.5, abs=1e-5)  # the cost pulls it up to the bound
    assert r.violation < 1e-6


def test_a_bound_can_have_a_window_and_a_lower_limit():
    def terms(x):
        return [opt.Cost(x - 1.0, name="reach"), opt.Bound(x, upper=0.4, t_from=2.0)]

    r = run(terms, free=[GOAL])
    late = r.t >= 2.0 - 1e-9
    assert r.q[late, 0].max() <= 0.4 + 1e-6
    assert r.q[~late, 0].max() > 0.4  # before the window the mass is free to go higher
    low = run(lambda x: [opt.Cost(x - 0.0, name="back"), opt.Bound(x, lower=0.3)], free=[GOAL])
    assert low.q[1:, 0].min() >= 0.3 - 1e-6


def test_a_bound_needs_a_limit():
    _, x, _ = mass_spring()
    with pytest.raises(ValueError, match="lower bound, an upper bound or both"):
        opt.Bound(x)


def test_a_bound_on_a_vector_takes_one_limit_per_entry():
    def terms(x):
        both = vmc.Stack(x, x)
        return [opt.Cost(x - 1.0, name="reach"), opt.Bound(both, upper=[0.5, 0.3])]

    r = run(terms, free=[GOAL])
    assert r.q[1:, 0].max() <= 0.3 + 1e-6  # the tighter entry rules


class FinalPosition(opt.Term):
    """A custom term: end at ``target``, and pay for the squared peak acceleration at the nodes."""

    name = "final"

    def __init__(self, coordinate, target):
        self.coordinate, self.target = coordinate, target

    def coordinates(self):
        return (self.coordinate,)

    def cost(self, trajectory):
        return sum((a**2 for a in trajectory.a), 0.0) * 1e-6

    def constraints(self, trajectory):
        y, _ = trajectory.coordinate(self.coordinate)
        return [(y[-1], self.target, self.target)]


def test_a_custom_term_adds_a_cost_and_a_constraint():
    r = run(lambda x: [FinalPosition(x, 0.25)], free=[GOAL])
    assert r.q[-1, 0] == pytest.approx(0.25, abs=1e-7)
    assert set(r.costs) == {"final"} and r.costs["final"] >= 0.0


def test_terms_see_the_rates_of_the_coordinates():
    seen = {}

    class Speed(opt.Term):
        name = "speed"

        def __init__(self, coordinate):
            self.coordinate = coordinate

        def coordinates(self):
            return (self.coordinate,)

        def cost(self, trajectory):
            seen["y"], seen["yd"] = trajectory.coordinate(self.coordinate)
            return sum((rate**2 for rate in seen["yd"]), 0.0)

    r = run(lambda x: [Speed(x)])
    assert len(seen["y"]) == len(seen["yd"]) == 41
    assert r.costs["speed"] == pytest.approx((r.v[:, 0] ** 2).sum(), rel=1e-10)


def test_the_term_names_are_the_groups_of_constraints_and_costs():
    system, x = tanh_mass()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 4.0, 11))
    problem.add(opt.Bound(x, upper=2.0, name="ceiling"))
    problem.add(opt.Effort(1.0))
    nlp = problem.build()
    assert list(nlp.constraints) == ["start", "dynamics", "position", "velocity", "ceiling"]
    assert nlp.cost_names == ["effort"]
    assert nlp.constraints["ceiling"] == slice(2 + 11 + 10 + 10, 2 + 11 + 10 + 10 + 10)
    assert np.all(nlp.ubg[nlp.constraints["ceiling"]] == 2.0)


def clearance_plan(bounded):
    """A point mass pulled along x through a sphere, with a free via-point spring to steer by."""
    robot = vmc.Mechanism("pt", model=Rod(length=1.0))
    tip = robot.point("tip")
    robot.add("mass", vmc.PointMass(tip, 1.0))
    robot.add("friction", vmc.LinearDamper(tip, 2.0))
    goal = vmc.Ref("goal", 3, [1.0, 0.0, 1.0])
    via = vmc.Ref("via", 3, [0.5, 0.4, 1.0])
    via.param.bounds = (-2.0, 2.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.TanhSpring(tip - goal, 6.0, 4.0))
    ctrl.add("steer", vmc.LinearSpring(tip - via, 3.0))
    ctrl.add("damper", vmc.LinearDamper(tip, 2.0))
    problem = opt.Problem(vmc.VirtualMechanismSystem(robot, ctrl))
    problem.add(opt.Collocation([0.0, 0.0, 0.0], 4.0, 41))
    problem.free("ctrl.steer.via")
    problem.add(opt.Effort(1e-3))
    problem.add(opt.Cost(tip - goal, 1.0, t_from=2.0, name="reach"))
    if bounded:
        sphere = vmc.SphereDistance(tip, SPHERE_CENTRE, SPHERE_RADIUS)
        problem.add(opt.Bound(sphere, lower=CLEARANCE, name="clearance"))
    result = problem.solve()
    kin = vmc.Kinematics(robot)
    path = np.array([kin.position(q, "tip") for q in result.q])
    return result, np.linalg.norm(path - SPHERE_CENTRE, axis=1) - SPHERE_RADIUS, path


SPHERE_CENTRE, SPHERE_RADIUS, CLEARANCE = np.array([0.5, 0.0, 1.0]), 0.15, 0.05


def test_a_clearance_bound_built_from_library_coordinates_steers_the_plan():
    free, free_gap, free_path = clearance_plan(bounded=False)
    assert free.converged and free_gap.min() < 0.0  # the straight path goes through the sphere
    np.testing.assert_allclose(free_path[:, 1], 0.0, atol=1e-5)
    kept, gap, path = clearance_plan(bounded=True)
    assert kept.converged and kept.violation < 1e-6
    assert gap[1:].min() >= CLEARANCE - 1e-6
    assert gap.min() == pytest.approx(CLEARANCE, abs=1e-5)  # the bound is active
    assert np.abs(path[:, 1]).max() > 0.1  # it swerves round the sphere


def test_the_arguments_of_a_term_are_checked_with_its_name():
    _, x, _ = mass_spring()
    with pytest.raises(ValueError, match="'wall': the lower bound is above the upper bound"):
        opt.Bound(x, lower=1.0, upper=0.5, name="wall")
    with pytest.raises(ValueError, match="'reach': the weight needs 1 or 1 values, got 2"):
        opt.Cost(x - 1.0, [1.0, 2.0], name="reach")
    with pytest.raises(ValueError, match="'both': the upper bound needs 1 or 2 values, got 3"):
        opt.Bound(vmc.Stack(x, x), upper=[1.0, 2.0, 3.0], name="both")


def test_an_empty_window_says_where_the_nodes_are():
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 4.0, 21))  # nodes every 0.2 s
    problem.add(opt.Cost(x - 1.0, t_from=0.05, t_to=0.15, name="reach"))
    with pytest.raises(
        ValueError, match=r"'reach', from 0\.05 s to 0\.15 s: the nodes are 0\.2 s apart"
    ):
        problem.build()
