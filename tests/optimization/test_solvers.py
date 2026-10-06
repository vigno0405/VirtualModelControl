"""The solver presets: each one that this CasADi build has finds the same plan."""

import casadi as ca
import numpy as np
import pytest

from small_plan import tanh_mass
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.optimization import solver as solvers

STIFFNESS = "ctrl.spring.stiffness"


def program(solver, **options):
    """A mass pulled to a goal by a spring of stiffness that steps, over 10 intervals."""
    system, x = tanh_mass(stiffness=2.0, max_force=5.0, goal=1.0, bounds=(0.5, 50.0))
    problem = opt.Problem(system, solver=solver, options=options)
    problem.add(opt.Shooting([0.0], 2.0, 11, steps=[STIFFNESS], substeps=4))
    problem.add(opt.Effort(0.01))
    problem.add(opt.Cost(x - 1.0, 1.0, name="reach"))
    return problem


@pytest.fixture(scope="module")
def reference():
    return program("ipopt-exact").solve()


def need(solver):
    if not solvers.available(solver):
        plugin, defaults = opt.PRESETS[solver]
        pytest.skip(f"this CasADi build lacks {plugin} or the QP solver {defaults.get('qpsol')}")


@pytest.mark.parametrize("solver", ["ipopt-exact", "sqp", "fatrop"])
def test_every_preset_this_build_has_finds_the_same_cost(solver, reference):
    need(solver)
    plan = program(solver).solve()
    assert plan.converged and plan.violation < 1e-5
    assert plan.cost == pytest.approx(reference.cost, rel=2e-3)
    assert plan.iterations > 0


@pytest.mark.parametrize("qp", ["qrqp", "qpoases", "proxqp", "daqp"])
def test_the_sqp_preset_runs_on_the_qp_solvers_that_take_this_problem(qp, reference):
    if not ca.has_conic(qp):
        pytest.skip(f"this CasADi build has no {qp}")
    plan = program("sqp", qpsol=qp).solve()
    assert plan.converged
    assert plan.cost == pytest.approx(reference.cost, rel=2e-3)


def test_the_exact_hessian_needs_far_fewer_iterations_than_the_default_quasi_newton_one(reference):
    default = program("ipopt").solve()
    assert default.cost == pytest.approx(reference.cost, rel=2e-2)
    assert reference.iterations * 5 < default.iterations


def test_the_real_time_iteration_is_one_sqp_step_that_leaves_an_optimum_where_it_is(reference):
    need("rti")
    problem = program("rti")
    cold = problem.solve()
    assert cold.iterations == 1 and not cold.converged
    again = problem.solve(warm_start=reference)  # one step from the optimum
    assert again.iterations == 1
    assert again.cost == pytest.approx(reference.cost, rel=1e-4)
    np.testing.assert_allclose(again.steps[STIFFNESS], reference.steps[STIFFNESS], rtol=1e-2)


def test_changing_the_solver_of_a_problem_makes_its_next_solve_create_the_solver_again(reference):
    problem = program("ipopt-exact")
    assert problem.solve().iterations == reference.iterations
    problem.solver = "rti"
    assert problem.solve().iterations == 1
    problem.solver = "ipopt-exact"
    assert problem.solve().iterations == reference.iterations


def test_the_progress_callback_can_stop_an_sqp_solve():
    seen = []

    def stop(iteration, cost, params, q):
        seen.append(iteration)
        return iteration >= 2

    plan = program("sqp").solve(progress=stop)
    assert plan.status == "User_Requested_Stop" and seen == [0, 1, 2]


def test_a_preset_is_known_by_name_and_has_to_be_in_the_build():
    with pytest.raises(ValueError, match="solver takes one of"):
        solvers.create_solver({}, {}, None, "nothing")
    assert solvers.available("ipopt")
    original, original_conic = ca.has_nlpsol, ca.has_conic
    try:
        ca.has_nlpsol = lambda name: False
        assert not solvers.available("fatrop")
    finally:
        ca.has_nlpsol = original
    try:
        ca.has_conic = lambda name: False
        assert not solvers.available("sqp") and solvers.available("ipopt")
    finally:
        ca.has_conic = original_conic
    assert set(opt.PRESETS) == {"ipopt", "ipopt-exact", "sqp", "rti", "fatrop"}


def test_fatrop_reports_a_number_and_a_flag_which_read_as_ipopts_words():
    assert solvers.status_of({"return_status": 0, "success": True}) == "Solve_Succeeded"
    assert solvers.status_of({"return_status": 3, "success": False}) == "Failed_3"
    assert solvers.status_of({"return_status": "Solve_Succeeded"}) == "Solve_Succeeded"
