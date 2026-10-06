from types import SimpleNamespace

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import Inversion
from virtualmodelcontrol.models import SerialChain
from virtualmodelcontrol.robots import bimanual, helyx

L1, L2 = 0.3, 0.2
ARC = [0.5, 0.75, 1.0]
SECTION_RADIUS = 0.030


def planar_arm():
    """Two links of 0.3 and 0.2 m turning about z, with sites at the elbow and the tip."""
    z = [0.0, 0.0, 1.0]
    return SerialChain(
        ["revolute", "revolute"],
        axes=[z, z],
        points=[[0.0, 0.0, 0.0], [L1, 0.0, 0.0]],
        sites={"elbow": (1, [L1, 0.0, 0.0]), "tip": (2, [L1 + L2, 0.0, 0.0])},
    )


def slider():
    """One prismatic joint along x, with sites a and b 0.5 m apart along y."""
    return SerialChain(
        ["prismatic"],
        axes=[[1.0, 0.0, 0.0]],
        points=[[0.0, 0.0, 0.0]],
        sites={"a": (1, [0.0, 0.0, 0.0]), "b": (1, [0.0, 0.5, 0.0])},
    )


def seen(robot, q, at):
    """What a position sensor sees of the robot at q: one row per point, (m, 3)."""
    kin = vmc.Kinematics(robot)
    return np.array([kin.position(q, a) for a in at])


def counting(inv):
    """The list that fills with a mark for every evaluation of the kinematics the fit makes."""
    calls, real = [], inv._f
    inv._f = lambda q: (calls.append(1), real(q))[1]
    return calls


def bend(theta, phi=0.0, dl=0.0):
    """Dx, Dy, Dl of one section bent by ``theta`` rad in the direction ``phi``."""
    D = theta * SECTION_RADIUS
    return [D * np.cos(phi), D * np.sin(phi), dl]


def test_a_planar_two_link_chain_is_recovered_from_its_points_to_the_tolerance():
    arm, at = planar_arm(), ["elbow", "tip"]
    inv = Inversion(arm, at, tol=1e-11)
    for q in ([0.7, -0.5], [-2.5, 2.0], [1.2, 0.4]):
        inv.reset()  # each from the straight arm, where the Jacobian has lost a rank
        found = inv(seen(arm, q, at))
        assert inv.converged
        np.testing.assert_allclose(found, q, atol=1e-10)
        assert np.abs(seen(arm, found, at) - seen(arm, q, at)).max() <= 1e-11


def test_the_soft_arm_is_recovered_from_its_markers_by_arc_parameter_by_site_and_by_part():
    arm = helyx.arm("290-145-145")
    q = np.concatenate([bend(0.3, 0.5, 0.004), bend(0.2, 2.0, -0.003), bend(0.4, -1.0, 0.002)])
    y = seen(arm, q, ARC)
    np.testing.assert_allclose(Inversion(arm, ARC)(y), q, atol=1e-8)
    np.testing.assert_allclose(Inversion(arm, ["seg1", "seg2", "tip"])(y), q, atol=1e-8)
    arms = bimanual.arms()  # two arms in one robot: 18 coordinates, points named (part, s)
    both = [(name, s) for name in ("right", "left") for s in ARC]
    q2 = np.concatenate([q, -q])
    np.testing.assert_allclose(Inversion(arms, both)(seen(arms, q2, both)), q2, atol=1e-8)


def test_a_warm_start_continues_from_the_last_result_and_needs_fewer_steps():
    arm, at = planar_arm(), ["elbow", "tip"]
    a, b = np.array([0.7, -0.5]), np.array([0.7005, -0.4995])
    inv = Inversion(arm, at, tol=1e-6, max_iter=1)  # one step per call
    inv(seen(arm, a, at), q0=a)
    assert inv.converged
    np.testing.assert_allclose(inv(seen(arm, b, at)), b, atol=1e-6)
    assert inv.converged  # one step from the last result was enough for the small move
    inv.reset()
    inv(seen(arm, b, at))
    assert not inv.converged  # one step from the neutral configuration is not
    calls = 1
    while not inv.converged and calls < 20:
        inv(seen(arm, b, at))  # every call goes on from where the last one stopped
        calls += 1
    assert 1 < calls < 20


def test_a_fit_that_cannot_converge_says_so_and_returns_the_best_configuration():
    inv = Inversion(slider(), ["a", "b"])
    q = inv([[1.0, 0.0, 0.0], [3.0, 0.5, 0.0]])  # a and b cannot both be where they were seen
    assert not inv.converged
    np.testing.assert_allclose(q, [2.0], atol=1e-9)  # the compromise of least squares
    arm = planar_arm()
    reach = Inversion(arm, ["tip"])
    q = reach([[0.6, 0.8, 0.0]])  # a metre away from a 0.5 m arm: stretched towards the point
    assert not reach.converged
    np.testing.assert_allclose(q, [np.arctan2(0.8, 0.6), 0.0], atol=1e-5)
    tip, start = seen(arm, q, ["tip"])[0, :2], seen(arm, [0.0, 0.0], ["tip"])[0, :2]
    assert np.linalg.norm(tip - [0.6, 0.8]) < np.linalg.norm(start - [0.6, 0.8])


def test_reset_forgets_the_warm_start_and_can_set_a_new_one():
    arm, at = planar_arm(), ["elbow", "tip"]
    y_a, y_b = seen(arm, [0.7, -0.5], at), seen(arm, [-0.4, 0.9], at)
    fresh, used = Inversion(arm, at, max_iter=2), Inversion(arm, at, max_iter=2)
    used(y_a)  # two steps on the way to a
    assert not np.array_equal(used(y_b), fresh(y_b))  # it goes on from there
    used.reset()
    fresh.reset()
    np.testing.assert_array_equal(used(y_b), fresh(y_b))  # as if from the neutral configuration
    used.reset([-0.4, 0.9])
    assert not used.converged
    np.testing.assert_array_equal(used(y_b), [-0.4, 0.9])  # it starts where it was told to
    assert used.converged
    used.reset()
    assert not used.converged


def test_q0_starts_this_fit_and_its_result_starts_the_next_one():
    inv = Inversion(slider(), ["a"], max_iter=0)  # no step: the result is the start
    y = [[2.0, 0.0, 0.0]]
    np.testing.assert_array_equal(inv(y), [0.0])  # the neutral configuration at first
    np.testing.assert_array_equal(inv(y, q0=[0.25]), [0.25])
    np.testing.assert_array_equal(inv(y), [0.25])
    np.testing.assert_array_equal(inv(y, q0=[0.75]), [0.75])
    np.testing.assert_array_equal(inv(y), [0.75])


def test_the_results_and_the_starts_are_copies_that_the_caller_may_change():
    inv = Inversion(slider(), ["a"], max_iter=0)
    start = np.array([0.5])
    inv.reset(start)
    start[:] = 7.0
    q = inv([[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(q, [0.5])
    q[:] = 99.0
    np.testing.assert_array_equal(inv([[2.0, 0.0, 0.0]]), [0.5])
    guess = np.array([0.25])
    inv([[2.0, 0.0, 0.0]], q0=guess)
    guess[:] = 5.0
    np.testing.assert_array_equal(inv([[2.0, 0.0, 0.0]]), [0.25])


def test_a_step_that_does_not_lower_the_error_is_tried_again_with_more_damping():
    arm = planar_arm()
    targets = [[0.1, 0.1], [-0.1, 0.05], [0.1 + 1e-6, 0.1], [-0.09, 0.069], [-0.205, 0.291]]
    for target in [*targets, [0.05, 0.2], [-0.348, -0.013]]:
        inv = Inversion(arm, ["tip"])  # from the straight arm the first full steps overshoot
        q = inv([[*target, 0.0]])  # each step raises the damping until the error drops
        assert inv.converged
        np.testing.assert_allclose(seen(arm, q, ["tip"])[0, :2], target, atol=1e-9)


def test_a_fit_that_no_damping_improves_gives_up_after_eight_retries():
    inv = Inversion(slider(), ["a", "b"], max_iter=1000)
    inv.reset([2.0])  # the compromise between the two points: no step can lower the error
    calls = counting(inv)
    inv([[1.0, 0.0, 0.0], [3.0, 0.5, 0.0]])
    assert not inv.converged
    assert len(calls) == 1 + 1 + 8  # the start, the first step and its eight retries, no more


def test_a_step_is_the_damped_gauss_newton_step_of_the_kinematics_jacobian():
    soft = np.concatenate([bend(0.3, 0.5, 0.004), bend(0.2, 2.0, -0.003), bend(0.4, -1.0, 0.002)])
    cases = [
        (planar_arm(), ["elbow", "tip"], [0.3, 0.2], [0.6, -0.1]),
        (helyx.arm("290-145-145"), ARC, 0.9 * soft, soft),
    ]
    for robot, at, q0, target in cases:
        kin, q0, lam = vmc.Kinematics(robot), np.asarray(q0), 0.05
        y = seen(robot, target, at)
        p = np.concatenate([kin.position(q0, a) for a in at])
        J = np.vstack([kin.jacobian(q0, a) for a in at])
        step = np.linalg.solve(J.T @ J + lam * np.eye(q0.size), J.T @ (p - y.ravel()))
        one = Inversion(robot, at, damping=lam, max_iter=1)
        np.testing.assert_allclose(one(y, q0=q0), q0 - step, rtol=1e-9, atol=1e-13)
        assert np.abs(step).max() > 0.5 * np.abs(q0 - target).max()  # a real step, not a zero


def test_the_tolerance_is_the_largest_position_error_in_metres():
    arm, tol = slider(), 1e-3
    three = ["a", "a", "a"]
    y = [[0.5, 0.9 * tol, 0.0]] * 3  # the joint cannot move them in y: 0.9 tol off, each
    inside = Inversion(arm, three, tol=tol)
    np.testing.assert_allclose(inside(y), [1.5 / (3.0 + 1e-6)], rtol=1e-12)  # one step
    assert inside.converged  # the largest error is 0.9 tol (its norm would be 1.6 tol)
    outside = Inversion(arm, three, tol=0.8 * tol)
    outside(y)
    assert not outside.converged


def test_a_start_inside_the_tolerance_is_kept_even_when_the_tolerance_is_zero():
    y = [[0.5, 0.0, 0.0]]
    exact = Inversion(slider(), ["a"], tol=0.0)
    np.testing.assert_array_equal(exact(y, q0=[0.5]), [0.5])
    assert exact.converged
    edge = Inversion(slider(), ["a"], tol=0.25)  # 0.75 is exactly the tolerance from 0.5
    np.testing.assert_array_equal(edge(y, q0=[0.75]), [0.75])
    assert edge.converged
    loose = Inversion(slider(), ["a"], tol=1e-3)
    close = np.array([0.5 + 5e-4])
    np.testing.assert_array_equal(loose(y, q0=close), close)  # converged: no step is taken
    assert loose.converged
    np.testing.assert_allclose(loose(y, q0=[0.5 + 2e-3]), [0.5], atol=1e-3)  # outside: one is
    assert loose.converged


def test_the_defaults_are_a_damping_of_a_millionth_a_nanometre_and_fifty_steps():
    one = Inversion(slider(), ["a"], max_iter=1)
    np.testing.assert_allclose(one([[2.0, 0.0, 0.0]]), [2.0 / (1.0 + 1e-6)], rtol=1e-12)
    inv = Inversion(slider(), ["a"])
    np.testing.assert_array_equal(inv([[0.5 + 5e-10, 0.0, 0.0]], q0=[0.5]), [0.5])  # inside
    np.testing.assert_allclose(inv([[0.5 + 5e-9, 0.0, 0.0]], q0=[0.5]), [0.5 + 5e-9], atol=1e-13)
    assert inv.converged
    slow = Inversion(slider(), ["a"], damping=1e3)  # a step is a thousandth of the way: count them
    q = slow([[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(q, 2.0 * (1.0 - (1.0 - 1.0 / 1001.0) ** 50), rtol=1e-9)
    assert not slow.converged


def test_lost_markers_do_not_raise_and_leave_the_warm_start_where_it_was():
    arm, at = planar_arm(), ["elbow", "tip"]
    inv, good = Inversion(arm, at), seen(arm, [0.7, -0.5], at)
    q = inv(good)
    np.testing.assert_array_equal(inv(np.full((2, 3), np.nan)), q)
    assert not inv.converged
    np.testing.assert_allclose(inv(good), [0.7, -0.5], atol=1e-8)
    assert inv.converged


def test_a_start_far_from_the_answer_may_end_in_a_local_minimum_and_says_so():
    arm, rng = helyx.arm("290-145-145"), np.random.default_rng(8)
    inv = Inversion(arm, ARC)
    for _ in range(40):  # from the neutral configuration: every section bent by up to 0.5 rad
        low, high = [0.0, 0.0, -0.01], [0.5, 2.0 * np.pi, 0.01]
        q = np.concatenate([bend(*rng.uniform(low, high)) for _ in range(3)])
        inv.reset()
        np.testing.assert_allclose(inv(seen(arm, q, ARC)), q, atol=1e-8)
        assert inv.converged
    q = np.concatenate([bend(1.5), np.zeros(6)])  # the first section alone bent by 1.5 rad
    inv.reset()
    inv(seen(arm, q, ARC))
    assert not inv.converged  # the later sections wandered off to compensate
    near = q + 5e-3 * rng.standard_normal(9)
    np.testing.assert_allclose(inv(seen(arm, q, ARC), q0=near), q, atol=1e-8)
    assert inv.converged


def test_the_positions_need_one_row_per_point_and_the_robot_a_flat_space():
    inv = Inversion(planar_arm(), ["elbow", "tip"])
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        inv([[0.3, 0.0, 0.0]])
    space = vmc.Product(vmc.Euclidean(2), vmc.SO2())  # planar body: nq = 4, nv = 3
    with pytest.raises(ValueError, match="nq=4 and nv=3"):
        Inversion(SimpleNamespace(space=space), [])
