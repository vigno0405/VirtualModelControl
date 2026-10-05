"""Ask and tell: the searchers (grid, random, CMA-ES, extremum seeking, Bayes), `tune`, Params."""

from itertools import pairwise

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import mass_spring
from virtualmodelcontrol import optimization as opt


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def rosenbrock(x):
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def run_until(searcher, cost, tolerance, max_evaluations):
    """Ask, evaluate and tell until the best cost is below ``tolerance``; the evaluations used."""
    while len(searcher.history) < max_evaluations and searcher.best_cost > tolerance:
        candidates = searcher.ask()
        searcher.tell(candidates, [cost(x) for x in candidates])
    return len(searcher.history)


def test_grid_asks_every_point_once_and_the_best_is_the_nearest_to_the_optimum():
    grid = opt.Grid([-1.0, 0.0], [1.0, 1.0], [5, 3])
    points = grid.ask()
    expected = [[a, b] for a in np.linspace(-1, 1, 5) for b in np.linspace(0, 1, 3)]
    np.testing.assert_allclose(points, expected)
    assert grid.ask() == []
    centre = np.array([0.3, 0.4])
    grid.tell(points, [np.sum((x - centre) ** 2) for x in points])
    np.testing.assert_allclose(grid.best, [0.5, 0.5])
    assert len(grid.history) == 15 and grid.best_cost == pytest.approx(0.05)
    assert len(opt.Grid([0, 0], [1, 1], 4).ask()) == 16  # one int for every dimension


def test_random_stays_inside_the_bounds_repeats_with_the_seed_and_never_gets_worse():
    lower, upper = np.array([-1.0, 2.0, 0.0]), np.array([1.0, 3.0, 0.5])
    a, b = opt.Random(lower, upper, size=5, seed=3), opt.Random(lower, upper, size=5, seed=3)
    best = []
    for _ in range(6):
        candidates = a.ask()
        assert len(candidates) == 5
        assert all(np.all(x >= lower) and np.all(x <= upper) for x in candidates)
        np.testing.assert_array_equal(candidates, b.ask())
        a.tell(candidates, [sphere(x) for x in candidates])
        best.append(a.best_cost)
    assert all(later <= earlier for earlier, later in pairwise(best))
    assert best[-1] < best[0] and len(a.history) == 30
    assert not np.array_equal(
        opt.Random(lower, upper, seed=4).ask(), opt.Random(lower, upper).ask()
    )


def test_cmaes_converges_on_a_sphere_and_its_step_size_shrinks():
    search = opt.CMAES([3.0, -2.0], sigma=1.0)
    assert search.size == 6
    used = run_until(search, sphere, 1e-10, 500)
    assert used <= 400 and search.best_cost < 1e-10
    assert search.sigma < 1e-2  # from 1.0: the step size follows the distance to the optimum


def test_cmaes_converges_on_rosenbrock():
    search = opt.CMAES([-1.2, 1.0], sigma=0.5, seed=1)
    used = run_until(search, rosenbrock, 1e-8, 2000)
    assert used <= 1000 and search.best_cost < 1e-8
    np.testing.assert_allclose(search.best, [1.0, 1.0], atol=1e-3)


def test_cmaes_respects_the_bounds_and_finds_the_optimum_on_the_edge():
    lower, upper = np.array([1.0, 1.0]), np.array([5.0, 5.0])
    search = opt.CMAES([3.0, 4.0], sigma=2.0, lower=lower, upper=upper, size=8)
    for _ in range(40):
        candidates = search.ask()
        assert all(np.all(x >= lower) and np.all(x <= upper) for x in candidates)
        search.tell(candidates, [sphere(x) for x in candidates])
    np.testing.assert_allclose(search.best, [1.0, 1.0], atol=1e-6)


def test_cmaes_is_deterministic_with_a_seed():
    a, b = opt.CMAES([1.0, 1.0, 1.0], 0.3, seed=5), opt.CMAES([1.0, 1.0, 1.0], 0.3, seed=5)
    assert a.size == 7  # 4 + floor(3 ln 3)
    run_until(a, sphere, 0.0, 60)
    run_until(b, sphere, 0.0, 60)
    np.testing.assert_array_equal(a.best, b.best)


@pytest.mark.parametrize(
    "searcher",
    [
        lambda: opt.Grid([0.0], [1.0], 3),
        lambda: opt.Random([0.0], [1.0], size=3),
        lambda: opt.CMAES([0.5], 0.1, size=4),
    ],
)
def test_tell_takes_one_cost_per_candidate_in_the_order_asked(searcher):
    search = searcher()
    candidates = search.ask()
    with pytest.raises(ValueError):
        search.tell(candidates, [1.0] * (len(candidates) + 1))
    search.tell(candidates, [3.0, 1.0, 2.0, 4.0][: len(candidates)])
    assert [cost for _, cost in search.history] == [3.0, 1.0, 2.0, 4.0][: len(candidates)]
    np.testing.assert_array_equal([x for x, _ in search.history], candidates)
    np.testing.assert_array_equal(search.best, candidates[1])
    assert search.best_cost == 1.0


def test_bounds_of_and_set_vector_round_trip_a_vector_and_a_matrix_param():
    vector = vmc.Param("v", [1.0, 2.0, 3.0], bounds=([0.0, 0.0, -1.0], [2.0, 4.0, 5.0]))
    matrix = vmc.Param("m", [[1.0, 2.0], [3.0, 4.0]], bounds=(-3.0, 8.0))
    other = vmc.Param("other", 9.0)
    params = vmc.ParamSet([vector, matrix, other])
    lower, upper, x0 = opt.bounds_of(params, ["m", "v"])
    np.testing.assert_array_equal(x0, [1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0])  # columns of m, then v
    np.testing.assert_array_equal(lower, [-3.0] * 4 + [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(upper, [8.0] * 4 + [2.0, 4.0, 5.0])
    params.set_vector(np.arange(7.0), ["m", "v"])
    np.testing.assert_array_equal(matrix.value, [[0.0, 2.0], [1.0, 3.0]])
    np.testing.assert_array_equal(vector.value, [4.0, 5.0, 6.0])
    assert other.value == 9.0
    np.testing.assert_array_equal(opt.bounds_of(params, ["m", "v"])[2], np.arange(7.0))


def test_tune_finds_a_stiffness_that_beats_the_start_and_agrees_with_a_grid():
    system, _, _ = mass_spring()
    name, dt = "ctrl.spring.stiffness", 1 / 200
    system.params[name].bounds = (0.5, 100.0)
    lower, upper, x0 = opt.bounds_of(system.params, [name])

    def error_cost(x):
        """Integral of the squared position error over 4 s of a simulated episode."""
        system.params.set_vector(x, [name])
        controller = vmc.VMCController(vmc.compile(system))
        plant = vmc.sim.ModelPlant(system.robot)
        log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=4.0)
        return float(np.sum((log.arrays()["q"][:, 0] - 1.0) ** 2) * dt)

    start = error_cost(x0)
    grid = opt.tune(opt.Grid(lower, upper, 8), error_cost, rounds=2)  # the second ask is empty
    assert len(grid.history) == 8
    cmaes = opt.tune(opt.CMAES(x0, 10.0, lower, upper, seed=0), error_cost, rounds=6)
    assert len(cmaes.history) == 24
    assert cmaes.best_cost < 0.7 * start
    assert cmaes.best_cost <= grid.best_cost * 1.05
    assert lower[0] <= cmaes.best[0] <= upper[0]
    assert abs(cmaes.best[0] - grid.best[0]) < 30.0  # the cost is flat near its minimum


def bowl(x):
    return float(np.sum((np.asarray(x) - [1.0, -2.0]) ** 2))


def test_extremum_seeking_walks_downhill_on_a_bowl_from_two_runs_a_round():
    searcher = opt.ExtremumSeeking([3.0, 2.0], amplitude=0.1, gain=0.15, seed=2)
    assert len(searcher.ask()) == 2
    start = bowl(searcher.x)
    opt.tune(searcher, bowl, rounds=150)
    assert bowl(searcher.x) < 1e-2 * start and len(searcher.history) == 2 * 150
    np.testing.assert_allclose(searcher.x, [1.0, -2.0], atol=0.1)


def test_extremum_seeking_stays_in_the_bounds_and_is_reproducible():
    def run():
        searcher = opt.ExtremumSeeking(
            [0.5, 0.5], 0.2, 0.3, lower=[0.0, 0.0], upper=[1.0, 1.0], seed=1
        )
        opt.tune(
            searcher, lambda x: -float(np.sum(x)), rounds=60
        )  # downhill is up and to the right
        return searcher

    first, again = run(), run()
    assert all(np.all((c >= 0.0) & (c <= 1.0)) for c, _ in first.history)
    np.testing.assert_allclose(first.x, [1.0, 1.0], atol=1e-6)  # pressed against the bounds
    np.testing.assert_array_equal(first.x, again.x)


def test_extremum_seeking_takes_two_costs_per_round():
    searcher = opt.ExtremumSeeking([0.0, 0.0], 0.1, 0.1)
    with pytest.raises(ValueError, match="2 candidates but 1 costs"):
        searcher.tell(searcher.ask(), [1.0])


def branin(x):
    """Three minima of 0.3979 on [-5, 10] x [0, 15]."""
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return float((x[1] - b * x[0] ** 2 + c * x[0] - 6.0) ** 2 + 10 * (1 - t) * np.cos(x[0]) + 10)


def test_bayes_beats_random_search_on_the_same_few_episodes():
    lower, upper = [-5.0, 0.0], [10.0, 15.0]
    for seed in range(3):
        bayes = opt.tune(opt.Bayes(lower, upper, seed=seed), branin, 30)
        random = opt.tune(opt.Random(lower, upper, size=1, seed=seed), branin, 30)
        print(f"branin, seed {seed}: {bayes.best_cost:.3f}, {random.best_cost:.3f}")
        assert bayes.best_cost < 1.0 and bayes.best_cost < random.best_cost
        assert len(bayes.history) == 30


def test_bayes_starts_from_a_latin_hypercube_and_then_asks_one_candidate_at_a_time():
    lower, upper = np.array([-1.0, 2.0]), np.array([1.0, 3.0])
    bayes = opt.Bayes(lower, upper, initial=6, seed=1)
    first = []
    for _ in range(6):
        (x,) = bayes.ask()
        first.append(x)
        bayes.tell([x], [sphere(x)])
    unit = (np.array(first) - lower) / (upper - lower)
    for column in unit.T:  # one point in each sixth of every dimension
        assert sorted(np.floor(6 * column)) == list(range(6))
    for _ in range(4):
        candidates = bayes.ask()
        assert len(candidates) == 1 and np.all(candidates[0] >= lower)
        assert np.all(candidates[0] <= upper)
        bayes.tell(candidates, [sphere(candidates[0])])
    assert len(bayes.history) == 10


def test_bayes_closes_in_on_the_minimum_of_a_bowl_and_repeats_with_the_seed():
    def bowl(x):
        return float((x[0] - 0.3) ** 2 + 2 * (x[1] + 0.2) ** 2)

    a = opt.tune(opt.Bayes([-1.0, -1.0], [1.0, 1.0], seed=2), bowl, 20)
    b = opt.tune(opt.Bayes([-1.0, -1.0], [1.0, 1.0], seed=2), bowl, 20)
    np.testing.assert_array_equal(a.best, b.best)
    np.testing.assert_allclose(a.best, [0.3, -0.2], atol=0.05)
    other = opt.tune(opt.Bayes([-1.0, -1.0], [1.0, 1.0], seed=3), bowl, 20)
    assert not np.array_equal(other.best, a.best)


def next_after(xs, costs):
    """Where Bayes asks next on [0, 1] once told the cost at each of the points ``xs``."""
    bayes = opt.Bayes([0.0], [1.0], initial=len(xs), seed=0)
    bayes.tell([[x] for x in xs], costs)
    return float(bayes.ask()[0][0])


def test_bayes_tries_where_it_knows_least_and_where_the_costs_promise_most():
    # equal costs in a corner say nothing about the rest: it explores the far end
    assert next_after([0.0, 0.05, 0.1], [1.0, 1.0, 1.0]) > 0.8
    # costs falling steadily to the right: it goes on to the right edge
    assert next_after([0.0, 0.2, 0.4, 0.6, 0.8], [1.0, 0.8, 0.6, 0.4, 0.2]) > 0.9
    # a valley between 0.3 and 0.8, lowest at 0.5: the expected improvement is highest just beside
    # it, on the side that is less known
    assert 0.58 < next_after([0.0, 0.3, 0.5, 0.8, 1.0], [1.0, 0.2, 0.0, 0.4, 1.0]) < 0.62


def test_bayes_refines_around_the_best_point_in_six_dimensions():
    centre = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.0])
    for seed in (1, 2):
        bayes = opt.Bayes(-np.ones(6), np.ones(6), seed=seed)
        opt.tune(bayes, lambda x: float(np.sum((np.asarray(x) - centre) ** 2)), 40)
        print(f"6-D bowl, seed {seed}: {bayes.best_cost:.3f}")
        assert bayes.best_cost < 0.06  # random points alone stay near 0.1 here
