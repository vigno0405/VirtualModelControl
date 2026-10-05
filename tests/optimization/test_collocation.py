"""The collocation block: accuracy, the start, the swap blend and its arguments."""

import itertools

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import DAMPING, exact_position, mass_spring
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import blend_weight
from virtualmodelcontrol.optimization.collocation import _flat


def plain(nodes, horizon=3.0, q0=0.0, **kwargs):
    """The mass-spring from rest at ``q0`` with nothing to optimize."""
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([q0], horizon, nodes, **kwargs))
    return problem, system, x


def hold_controller(system, x, stiffness, goal):
    mechanism = vmc.Mechanism("hold")
    mechanism.add("spring", vmc.LinearSpring(x - goal, stiffness))
    mechanism.add("damper", vmc.LinearDamper(x, DAMPING))
    return vmc.VirtualMechanismSystem(system.robot, mechanism)


def test_collocation_is_second_order_accurate_against_the_exact_solution():
    errors = []
    for nodes in (41, 81, 161):
        problem, _, _ = plain(nodes)
        result = problem.solve()
        assert result.converged
        errors.append(np.abs(result.q[:, 0] - exact_position(result.t)).max())
    assert errors[2] < 1e-4
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)  # halving the step quarters the error


def test_velocities_and_accelerations_satisfy_the_trapezoid_rule():
    problem, _, _ = plain(51)
    r = problem.solve()
    dt = r.t[1] - r.t[0]
    np.testing.assert_allclose(r.q[1:] - r.q[:-1], 0.5 * dt * (r.v[:-1] + r.v[1:]), atol=1e-10)
    np.testing.assert_allclose(r.v[1:] - r.v[:-1], 0.5 * dt * (r.a[:-1] + r.a[1:]), atol=1e-10)
    assert r.violation < 1e-8


def test_the_motion_starts_at_rest_or_at_the_given_velocity():
    problem, _, _ = plain(31)
    r = problem.solve()
    assert r.q[0, 0] == 0.0 and r.v[0, 0] == 0.0
    problem, _, _ = plain(31, v0=[0.5])
    assert problem.solve().v[0, 0] == pytest.approx(0.5)


def test_the_plan_starts_at_rest_where_the_robot_is_whatever_the_model_gets_wrong():
    # At q0 = 0.3 the spring pulls towards 1.0, so the robot is not in equilibrium under it.
    # With the controller in place at the start, the model carries the force that balances it.
    system, x, _ = mass_spring()
    hold = hold_controller(system, x, 4.0, 1.0)
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.3], 3.0, 31, initial=hold, transition=1.0))
    held = problem.solve()
    np.testing.assert_allclose(held.q, 0.3, atol=1e-9)
    np.testing.assert_allclose(held.v, 0.0, atol=1e-9)
    moving, _, _ = plain(31, q0=0.3)
    assert moving.solve().q[-1, 0] > 0.6  # without it, the spring pulls the mass away


def test_the_torques_are_the_blend_of_the_two_controllers():
    system, x, _ = mass_spring(goal=1.0, stiffness=9.0)  # the new controller: k = 9 to 1.0
    hold = hold_controller(system, x, 4.0, 0.0)  # the one in place: k = 4 to 0.0
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 31, initial=hold, transition=1.0))
    r = problem.solve()
    w = np.array([blend_weight(t, 1.0) for t in r.t])
    np.testing.assert_allclose(r.blend, w, atol=1e-15)
    new = -9.0 * (r.q[:, 0] - 1.0) - DAMPING * r.v[:, 0]
    old = -4.0 * (r.q[:, 0] - 0.0) - DAMPING * r.v[:, 0]
    np.testing.assert_allclose(r.u[:, 0], w * new + (1.0 - w) * old, atol=1e-10)
    assert r.blend[0] == 0.0 and r.blend[-1] == 1.0


def test_without_an_initial_controller_the_new_one_acts_from_the_start():
    problem, _, _ = plain(21)
    r = problem.solve()
    np.testing.assert_array_equal(r.blend, 1.0)
    assert r.u[0, 0] == pytest.approx(-4.0 * (0.0 - 1.0))  # the spring alone at the start


def test_a_running_controller_can_be_the_initial_one():
    system, x, _ = mass_spring()
    hold = hold_controller(system, x, 4.0, 1.0)
    running = vmc.VMCController(vmc.compile(hold))
    by_system, by_controller = (
        opt.Problem(system),
        opt.Problem(system),
    )
    by_system.add(opt.Collocation([0.3], 3.0, 21, initial=hold, transition=1.0))
    by_controller.add(opt.Collocation([0.3], 3.0, 21, initial=running, transition=1.0))
    np.testing.assert_allclose(by_system.solve().u, by_controller.solve().u, atol=1e-12)


def test_the_initial_controller_must_control_the_same_robot():
    system, _, _ = mass_spring()
    other, other_x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 11, initial=hold_controller(other, other_x, 4.0, 1.0)))
    with pytest.raises(ValueError, match="same robot"):
        problem.build()


def test_virtual_states_cannot_be_planned_yet():
    system, x, _ = mass_spring()
    ctrl = vmc.Mechanism("virtual")
    z = ctrl.add_state("z", 1, unit="m")
    ctrl.add("mass", vmc.Inertance(z, 1.0))
    ctrl.add("link", vmc.LinearSpring(x - z, 5.0))
    stateful = vmc.VirtualMechanismSystem(system.robot, ctrl)
    problem = opt.Problem(stateful)
    problem.add(opt.Collocation([0.0], 3.0, 11))
    with pytest.raises(NotImplementedError, match="virtual states"):
        problem.build()
    plan = opt.Problem(system)
    plan.add(opt.Collocation([0.0], 3.0, 11, initial=stateful, transition=1.0))
    with pytest.raises(NotImplementedError, match="virtual states"):
        plan.build()


def test_only_flat_spaces_are_supported():
    assert _flat(vmc.Euclidean(3))
    assert _flat(vmc.Product(vmc.Euclidean(2), vmc.Euclidean(1)))
    assert not _flat(vmc.SO2())
    assert not _flat(vmc.Product(vmc.Euclidean(2), vmc.SO2()))


def test_the_arguments_are_checked():
    with pytest.raises(ValueError, match="2 nodes"):
        opt.Collocation([0.0], 1.0, 1)
    with pytest.raises(ValueError, match="horizon"):
        opt.Collocation([0.0], 0.0, 10)
    with pytest.raises(ValueError, match="transition"):
        opt.Collocation([0.0], 1.0, 10, transition=-1.0)
    with pytest.raises(ValueError, match="scales"):
        opt.Collocation([0.0], 1.0, 10, scales=(1.0, 0.0, 1.0))
    problem, _, _ = plain(11, q0=0.0)
    problem._blocks[0].q0 = np.zeros(2)
    with pytest.raises(ValueError, match="q0 and v0 need 1 and 1"):
        problem.build()


def test_the_controller_in_place_is_the_one_there_when_the_collocation_is_made():
    # Applying a result to the system that is also the initial controller must not change what
    # the next plan starts from.
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.2], 4.0, 21, initial=system, transition=1.0))
    problem.free("ctrl.spring.stiffness")
    problem.add(opt.Cost(x - 1.0, name="reach"))
    first = problem.solve()
    first.apply(system)
    problem.add(opt.Bound(x, upper=5.0, name="loose"))  # rebuilds the program
    again = problem.solve()
    assert again.cost == pytest.approx(first.cost, rel=1e-6)
    np.testing.assert_allclose(again.q, first.q, atol=1e-4)  # solver noise, not a new start


def test_the_initial_controller_must_be_a_system_or_a_running_controller():
    system, x, _ = mass_spring()
    del system
    ctrl = vmc.Mechanism("hold")
    ctrl.add("spring", vmc.LinearSpring(x - 1.0, 4.0))
    with pytest.raises(TypeError, match="not a Mechanism"):
        opt.Collocation([0.0], 3.0, 11, initial=ctrl)


def test_the_scales_are_three_positive_numbers():
    with pytest.raises(ValueError, match="three positive numbers"):
        opt.Collocation([0.0], 1.0, 10, scales=(1.0, 1.0))
