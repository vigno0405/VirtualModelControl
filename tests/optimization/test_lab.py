"""The trajectory optimization against the lab's own optimizer, on the lab's model.

The fixtures were made by running the lab's optimizer on these problems: the cost and constraints
at a point near the solution, and its solved plans. The program must agree to rounding error; the
solver paths to solver tolerance.
"""

import numpy as np
import pytest

from lab_optimization import ARRAYS, COMMON, INFO, SIGN, lab_problem
from virtualmodelcontrol import optimization as opt

SCENARIOS = list(INFO["scenarios"])
N = COMMON["nodes"]


def library_name(element, part):
    return f"ctrl.{element}.{part}"


def probe(name, problem, references):
    """The library's x and p at the lab's probe point."""
    sc = INFO["scenarios"][name]
    nlp = problem.build()
    warm = {
        "q": ARRAYS[f"{name}/probe_q"],
        "v": ARRAYS[f"{name}/probe_v"],
        "a": ARRAYS[f"{name}/probe_a"],
        "params": {
            library_name(element, part): value
            for element, values in sc["probe_theta"].items()
            for part, value in values.items()
            if library_name(element, part) in nlp.free
        },
    }
    _, p = problem._references(nlp, references)
    return nlp, problem.initial_guess(warm), p


@pytest.mark.parametrize("name", SCENARIOS)
def test_program_matches_the_lab_at_a_probe_point(name):
    problem, _, references = lab_problem(name)
    nlp, x, p = probe(name, problem, references)
    f, g = (np.array(fn(x, p)).ravel() for fn in nlp.functions())
    lab = {key.split("probe_g_")[1]: ARRAYS[key] for key in ARRAYS if f"{name}/probe_g_" in key}
    rows = {key: g[where] for key, where in nlp.constraints.items()}
    np.testing.assert_allclose(f, ARRAYS[f"{name}/probe_f"], rtol=1e-12)
    np.testing.assert_allclose(rows["start"], lab["start"], atol=1e-13)
    np.testing.assert_allclose(
        rows["dynamics"].reshape(N, 9), lab["dynamics"], rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(rows["position"].reshape(N - 1, 9), lab["position"], atol=1e-14)
    np.testing.assert_allclose(rows["velocity"].reshape(N - 1, 9), lab["velocity"], atol=1e-14)
    np.testing.assert_allclose(rows["bending"].reshape(N - 1, 3), lab["bending"], atol=1e-14)
    for j in range(len(INFO["scenarios"][name]["obstacles"])):
        rows_j = rows[f"obstacle{j + 1}"].reshape(N - 1, -1)
        np.testing.assert_allclose(rows_j, lab["obstacle"][:, j, :], atol=1e-12)
    u = np.array(nlp.outputs["u"](x, p)).T
    np.testing.assert_allclose(SIGN * u, ARRAYS[f"{name}/probe_tau_motor"], atol=1e-14)


@pytest.mark.parametrize("name", SCENARIOS)
def test_solution_matches_the_lab(name):
    problem, _, references = lab_problem(name)
    result = problem.solve(references)
    sc = INFO["scenarios"][name]
    assert result.converged and sc["sol"]["status"] in opt.CONVERGED
    assert result.cost == pytest.approx(sc["sol"]["cost"], rel=1e-6)
    np.testing.assert_allclose(result.t, ARRAYS[f"{name}/sol_t"], atol=1e-14)
    np.testing.assert_allclose(result.blend, ARRAYS[f"{name}/sol_blend"], atol=1e-14)
    np.testing.assert_allclose(result.q, ARRAYS[f"{name}/sol_q"], atol=1e-7)
    np.testing.assert_allclose(result.v, ARRAYS[f"{name}/sol_v"], atol=1e-7)
    np.testing.assert_allclose(SIGN * result.u, ARRAYS[f"{name}/sol_tau_motor"], atol=1e-7)
    # What the optimizer decided: the placement and, where the cost is not flat, the gains.
    for element, values in sc["solution_theta"].items():
        for part, lab_value in values.items():
            if library_name(element, part) in result.params and part == "s":
                assert float(result.params[library_name(element, part)]) == pytest.approx(
                    lab_value, abs=1e-4
                )
    if name == "reach_gain":
        k = float(result.params["ctrl.tanh.stiffness"])
        assert k == pytest.approx(sc["solution_theta"]["tanh"]["stiffness"], rel=1e-5)


def test_the_search_visits_the_labs_points_and_picks_its_winner():
    lab = INFO["search"]
    problem, _, _ = lab_problem(lab["scenario"])
    anchors = {"ctrl.tanh.goal": COMMON["target"]}
    found = opt.search_references(problem, anchors, lab["radius"], lab["step"])
    assert len(found.candidates) == len(lab["candidates"]) == 7
    for mine, theirs in zip(found.candidates, lab["candidates"], strict=True):
        np.testing.assert_allclose(mine["references"]["ctrl.tanh.goal"], theirs["references"][0])
        assert mine["status"] == theirs["status"]
        # an "acceptable" ending stops once the cost changes by less than 1e-4 per iteration
        assert mine["cost"] == pytest.approx(theirs["cost"], rel=1e-4)
    np.testing.assert_allclose(found.references["ctrl.tanh.goal"], lab["best"]["references"][0])
    assert found.result.cost == pytest.approx(lab["result_cost"], rel=1e-6)
    np.testing.assert_allclose(found.result.q, ARRAYS["search/sol_q"], atol=1e-6)
    assert found.result.cost < found.candidates[0]["cost"]  # better than staying at the target


def test_the_cost_and_constraint_derivatives_match_finite_differences():
    problem, _, references = lab_problem(
        "avoid_both"
    )  # free placement: if_else inside the kinematics
    nlp, x, p = probe("avoid_both", problem, references)
    import casadi as ca

    f, g = nlp.functions()
    grad = np.array(
        ca.Function("gradient_f", [nlp.x, nlp.p], [ca.gradient(nlp.f, nlp.x)])(x, p)
    ).ravel()
    jac = ca.Function("jacobian_g", [nlp.x, nlp.p], [ca.jacobian(nlp.g, nlp.x)])
    jac_x = np.array(jac(x, p))
    rng = np.random.default_rng(0)
    columns = np.concatenate(
        [rng.choice(x.size - 6, 12, replace=False), np.arange(x.size - 6, x.size)]
    )
    h = 1e-6
    for i in columns:
        dx = np.zeros_like(x)
        dx[i] = h
        fd_f = (float(f(x + dx, p)) - float(f(x - dx, p))) / (2 * h)
        fd_g = (np.array(g(x + dx, p)) - np.array(g(x - dx, p))).ravel() / (2 * h)
        assert grad[i] == pytest.approx(fd_f, rel=1e-5, abs=1e-8)
        np.testing.assert_allclose(jac_x[:, i], fd_g, rtol=1e-5, atol=1e-7)
