"""The grid search over references: the grid, and the search's rules on a stand-in problem."""

from types import SimpleNamespace

import numpy as np
import pytest

from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.optimization.search import sphere_points


def test_the_grid_is_the_points_inside_the_ball_centre_first():
    points = sphere_points([1.0, 2.0, 3.0], 0.1, 0.1)
    assert len(points) == 7 and np.allclose(points[0], [1.0, 2.0, 3.0])
    offsets = points[1:] - [1.0, 2.0, 3.0]
    assert np.allclose(np.sort(np.abs(offsets).sum(axis=1)), 0.1)  # the six axis neighbours
    assert len(sphere_points([0, 0, 0], 0.1, 0.05)) == 33
    flat = sphere_points([0.0, 0.0], 0.2, 0.1)  # works in any dimension
    assert len(flat) == 13 and np.linalg.norm(flat, axis=1).max() <= 0.2 + 1e-12
    dist = np.linalg.norm(points - points[0], axis=1)
    assert np.all(np.diff(dist) >= -1e-12)  # ordered by distance


class Stand:
    """A problem whose cost is a function of the references; ``crash`` and ``stuck`` mark points."""

    def __init__(self, cost, crash=lambda r: False, stuck=lambda r: False, names=("a", "b")):
        self.cost, self.crash, self.stuck = cost, crash, stuck
        self.calls = []
        self.names = names

    def build(self):
        return SimpleNamespace(parameters={n: slice(i, i + 1) for i, n in enumerate(self.names)})

    def solve(self, references, warm_start=None):
        refs = {n: np.asarray(v, dtype=float) for n, v in references.items()}
        self.calls.append((refs, warm_start))
        if self.crash(refs):
            raise RuntimeError("the solver crashed")
        status = "Infeasible_Problem_Detected" if self.stuck(refs) else "Solve_Succeeded"
        return opt.Result(
            t=np.zeros(1), q=np.zeros((1, 1)), v=np.zeros((1, 1)), a=np.zeros((1, 1)),
            u=np.zeros((1, 1)), blend=np.ones(1), params={}, references=refs,
            cost=float(self.cost(refs)), costs={}, status=status, iterations=3, seconds=0.0,
            violation=0.0,
        )  # fmt: skip


def bowl(centre_a, centre_b):
    return lambda r: np.sum((r["a"] - centre_a) ** 2) + np.sum((r["b"] - centre_b) ** 2)


ANCHORS = {"a": [0.0, 0.0, 0.0], "b": [1.0, 1.0, 1.0]}


def test_each_reference_in_turn_takes_its_best_point_with_the_others_at_their_best():
    problem = Stand(bowl([0.1, 0.0, 0.0], [1.0, 1.0, 0.9]))
    found = opt.search_references(problem, ANCHORS, 0.1, 0.1)
    np.testing.assert_allclose(found.references["a"], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(found.references["b"], [1.0, 1.0, 0.9])
    assert found.result.cost == pytest.approx(0.0, abs=1e-12)
    # 7 points for a, then 7 for b, but b's first point (at its anchor) is already solved
    assert len(problem.calls) == 7 + 6 and len(found.candidates) == 13
    first_b = problem.calls[7][0]  # b's search starts with a at its best point
    np.testing.assert_allclose(first_b["a"], [0.1, 0.0, 0.0])


def test_every_solve_after_the_first_is_warm_started_from_the_best_so_far():
    cost = bowl([0.0, 0.0, 0.1], [1.0, 1.0, 1.0])
    problem = Stand(cost)
    opt.search_references(problem, ANCHORS, 0.1, 0.1)
    assert problem.calls[0][1] is None
    solved = []
    for k, (refs, warm) in enumerate(problem.calls):
        if k:
            assert warm.cost == min(solved)  # the best plan found before this point
        solved.append(cost(refs))


def test_a_point_that_crashes_or_does_not_converge_just_loses():
    cheap_but_bad = lambda r: r["a"][0] > 0.05  # noqa: E731  (the lowest-cost point is cut off)
    problem = Stand(
        bowl([0.1, 0.0, 0.0], [1.0, 1.0, 1.0]),
        crash=lambda r: r["a"][1] > 0.05,
        stuck=cheap_but_bad,
    )
    found = opt.search_references(problem, ANCHORS, 0.1, 0.1)
    assert found.result.converged
    assert not np.allclose(found.references["a"], [0.1, 0.0, 0.0])
    statuses = [c["status"] for c in found.candidates]
    assert any("crashed" in s for s in statuses) and "Infeasible_Problem_Detected" in statuses
    # a point that did not converge still shows its cost, a point that crashed has none
    stuck = [c for c in found.candidates if "Infeasible" in c["status"]]
    assert len(stuck) == 1 and stuck[0]["cost"] == pytest.approx(0.0, abs=1e-12)
    assert all(c["cost"] is None for c in found.candidates if "crashed" in c["status"])


def test_when_nothing_converges_the_plan_at_the_anchors_comes_back():
    problem = Stand(bowl([0.0] * 3, [1.0] * 3), stuck=lambda r: True)
    found = opt.search_references(problem, ANCHORS, 0.1, 0.1)
    assert not found.result.converged
    np.testing.assert_allclose(found.references["a"], ANCHORS["a"])
    np.testing.assert_allclose(found.references["b"], ANCHORS["b"])


def test_when_everything_crashes_it_says_so():
    problem = Stand(bowl([0.0] * 3, [1.0] * 3), crash=lambda r: True)
    with pytest.raises(RuntimeError, match="no grid point could be solved"):
        opt.search_references(problem, ANCHORS, 0.1, 0.1)


def test_a_second_pass_corrects_a_choice_that_depended_on_the_other():
    # The cost couples the two references: the best b depends on a, and the best a on b.
    def coupled(r):
        return (r["a"][0] - r["b"][0] + 0.1) ** 2 + 2 * (r["b"][0] - 0.1) ** 2

    anchors = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}
    one = opt.search_references(Stand(coupled), anchors, 0.1, 0.1, passes=1)
    two = opt.search_references(Stand(coupled), anchors, 0.1, 0.1, passes=3)
    assert one.result.cost == pytest.approx(0.01)  # a = -0.1 first, then b = 0.1
    assert two.result.cost == pytest.approx(0.0, abs=1e-12)  # the second pass moves a to 0.0


def test_progress_is_reported_before_and_after_each_point():
    problem = Stand(lambda r: np.sum(r["a"] ** 2), names=("a",))
    events = []
    opt.search_references(
        problem, {"a": [0.0] * 3}, 0.1, 0.1,
        progress=lambda n, best, stage, candidate: events.append((n, stage, candidate)),
    )  # fmt: skip
    assert len(events) == 14 and events[0][2] is None and events[1][2] is not None
    assert events[0][1].endswith("solving") and events[1][1] == "a, point 1/7"
    assert events[-1][0] == 7


def test_an_unknown_anchor_is_refused():
    with pytest.raises(KeyError, match="not parameters"):
        opt.search_references(Stand(bowl([0.0] * 3, [0.0] * 3)), {"nope": [0.0] * 3}, 0.1, 0.1)


def test_an_empty_set_of_anchors_is_refused_and_a_wrong_one_names_the_right_ones():
    problem = Stand(bowl([0.0] * 3, [0.0] * 3))
    with pytest.raises(ValueError, match="at least one anchor"):
        opt.search_references(problem, {}, 0.1, 0.1)
    with pytest.raises(KeyError, match=r"they are \['a', 'b'\]"):
        opt.search_references(problem, {"nope": [0.0] * 3}, 0.1, 0.1)
