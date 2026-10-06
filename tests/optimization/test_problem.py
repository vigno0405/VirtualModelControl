"""The problem: which Params are optimized, the references of each solve, warm starts, apply."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import STIFFNESS, mass_spring, tanh_mass, two_masses
from virtualmodelcontrol import optimization as opt

GOAL, SPRING = "ctrl.spring.goal", "ctrl.spring.stiffness"


def tracking(system, x, nodes=41, horizon=4.0, target=1.0, effort=1e-2):
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], horizon, nodes))
    problem.add(opt.Effort(effort))
    problem.add(opt.Cost(x - target, name="reach"))
    return problem


def test_params_are_chosen_by_name_or_glob():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    assert problem.free("ctrl.damper.damping") == ["ctrl.damper.damping"]
    assert problem.parameter("ctrl.spring.*") == [SPRING, GOAL]
    assert list(problem.build().free) == ["ctrl.damper.damping"]
    assert list(problem.build().parameters) == [SPRING, GOAL]


def test_naming_errors_are_reported_when_the_names_are_given():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    with pytest.raises(KeyError, match=r"no Param matches 'ctrl\.nothing'"):
        problem.free("ctrl.nothing")
    problem.free(SPRING)
    with pytest.raises(ValueError, match="free already"):
        problem.parameter(SPRING)
    problem.parameter(GOAL)
    with pytest.raises(ValueError, match="parameter already"):
        problem.free(GOAL)


def test_references_default_to_the_current_value_and_change_with_every_solve():
    system, x, _ = mass_spring()
    problem = tracking(system, x, horizon=6.0)
    problem.parameter(GOAL)
    nlp = problem.build()
    default = problem.solve()
    assert default.references[GOAL] == pytest.approx([1.0])
    assert default.q[-1, 0] == pytest.approx(1.0, abs=1e-2)
    moved = problem.solve({GOAL: [0.5]})
    assert moved.q[-1, 0] == pytest.approx(0.5, abs=1e-2)
    assert problem.build() is nlp  # the program was built once
    system.params[GOAL].value = [0.2]  # a changed Param is the new default
    assert problem.solve().q[-1, 0] == pytest.approx(0.2, abs=1e-2)
    with pytest.raises(KeyError, match="not parameters"):
        problem.solve({SPRING: 3.0})


def test_params_not_declared_stay_at_their_value_when_the_program_is_built():
    system, x, _ = mass_spring()
    problem = tracking(system, x, horizon=6.0)
    problem.build()
    system.params[GOAL].value = [0.2]  # too late: the goal was folded in at 1.0
    assert problem.solve().q[-1, 0] == pytest.approx(1.0, abs=1e-2)


def test_a_term_can_bring_its_own_params_and_make_them_parameters():
    system, x, _ = mass_spring()
    problem = tracking(system, x, horizon=6.0)
    assert "reach.ref" in problem.params  # the target inside Cost(x - 1.0)
    problem.parameter("reach.ref")
    # the spring still pulls to 1.0, so the cost to the new target is larger than to the old
    assert problem.solve({"reach.ref": [0.8]}).costs["reach"] > problem.solve().costs["reach"]


def test_the_free_reference_is_a_minimum_of_the_cost_it_defines():
    system, x, _ = mass_spring()
    free = tracking(system, x)
    free.free(GOAL)
    best = free.solve()
    fixed = tracking(*mass_spring()[:2])
    fixed.parameter(GOAL)
    g = best.params[GOAL].item()

    def total(r):
        return fixed.solve({GOAL: [r]}).cost

    assert total(g) == pytest.approx(best.cost, rel=1e-8)
    assert total(g) < total(g - 0.05) and total(g) < total(g + 0.05)
    slope = (total(g + 1e-3) - total(g - 1e-3)) / 2e-3
    assert abs(slope) < 1e-4 * best.cost


def test_bounds_and_scale_come_from_the_param():
    system, x = tanh_mass(bounds=(0.5, 3.0))
    problem = tracking(system, x)
    problem.free(SPRING)
    capped = problem.solve()
    assert float(capped.params[SPRING]) == pytest.approx(3.0, abs=1e-6)  # wants it stiffer
    scaled, x = tanh_mass(bounds=(0.5, 50.0), scale=10.0)
    plain, y = tanh_mass(bounds=(0.5, 50.0))
    results = []
    for system, coord in ((scaled, x), (plain, y)):
        problem = tracking(system, coord)
        problem.free(SPRING)
        results.append(problem.solve())
    assert results[0].cost == pytest.approx(results[1].cost, rel=1e-6)
    assert float(results[0].params[SPRING]) == pytest.approx(
        float(results[1].params[SPRING]), rel=1e-3
    )


def test_a_robot_param_can_be_optimized_too():
    system, x, _ = mass_spring()
    system.params["robot.mass.inertance"].bounds = (0.1, 10.0)
    problem = tracking(system, x, effort=0.0)
    problem.free("robot.mass.inertance")
    result = problem.solve()
    assert float(result.params["robot.mass.inertance"]) == pytest.approx(0.1, abs=1e-5)  # lighter


def test_warm_start_from_a_result_or_a_dict_saves_iterations():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    cold = problem.solve()
    warm = problem.solve(warm_start=cold)
    assert warm.iterations < cold.iterations and warm.cost == pytest.approx(cold.cost, rel=1e-6)
    as_dict = {"q": cold.q, "v": cold.v, "a": cold.a, "params": cold.params}
    assert problem.solve(warm_start=as_dict).iterations == warm.iterations
    with pytest.raises(ValueError, match="warm start 'q'"):
        problem.solve(warm_start={"q": np.zeros((3, 1))})


def test_progress_is_called_at_every_iteration_with_the_iterate():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    calls = []
    result = problem.solve(
        progress=lambda i, cost, params, q: calls.append((i, cost, float(params[SPRING]), q.shape))
    )
    assert [c[0] for c in calls] == list(range(result.iterations + 1))
    assert calls[0][2] == pytest.approx(2.0)  # the starting stiffness
    assert calls[-1][1] == pytest.approx(result.cost) and calls[-1][3] == (41, 1)
    assert calls[0][1] > calls[-1][1]
    again = problem.solve()  # no callback now
    assert again.cost == pytest.approx(result.cost)


def test_options_override_the_solver_defaults():
    system, x = tanh_mass()
    problem = opt.Problem(system, options={"ipopt.max_iter": 2})
    problem.add(opt.Collocation([0.0], 4.0, 41))
    problem.free(SPRING)
    problem.add(opt.Effort(1e-2))
    problem.add(opt.Cost(x - 1.0))
    result = problem.solve()
    assert result.status == "Maximum_Iterations_Exceeded" and not result.converged
    assert opt.IPOPT["ipopt.max_iter"] == 500  # the defaults are not touched


def test_the_result_breaks_the_cost_down_by_term():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    result = problem.solve()
    assert set(result.costs) == {"effort", "reach"}
    assert sum(result.costs.values()) == pytest.approx(result.cost, rel=1e-12)


def test_apply_writes_the_free_params_and_the_references_into_a_system():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    problem.parameter(GOAL)
    result = problem.solve({GOAL: [0.7]})
    assert result.apply(system) is None
    assert system.params[SPRING].value == pytest.approx(result.params[SPRING])
    assert system.params[GOAL].value == pytest.approx([0.7])
    spare, _ = tanh_mass()
    result.apply(spare.params)  # a ParamSet works too
    assert spare.params[GOAL].value == pytest.approx([0.7])


def test_apply_to_a_running_controller_returns_the_energy_jump():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    problem.free(SPRING)
    problem.parameter(GOAL)
    result = problem.solve({GOAL: [0.6]})
    controller = vmc.VMCController(vmc.compile(system))
    position = 0.2
    controller.reset(0.0)
    controller.step(0.0, vmc.Signals(0.0, motor_position=[position], motor_velocity=[0.0]))
    k_new = float(result.params[SPRING])
    expected = 0.5 * k_new * (position - 0.6) ** 2 - 0.5 * STIFFNESS * (position - 1.0) ** 2
    assert result.apply(controller) == pytest.approx(expected, rel=1e-9)
    assert controller.live_params()[SPRING] == pytest.approx(k_new)


def test_apply_to_a_controller_needs_the_params_to_be_live_there():
    system, x, _ = mass_spring()
    system.params["robot.mass.inertance"].bounds = (0.1, 10.0)
    problem = tracking(system, x)
    problem.free("robot.mass.inertance")
    result = problem.solve()
    controller = vmc.VMCController(vmc.compile(system))
    with pytest.raises(KeyError, match="not a live Param"):
        result.apply(controller)


def test_blocks_are_added_in_order():
    system, _, _ = mass_spring()
    problem = opt.Problem(system)
    with pytest.raises(ValueError, match="Collocation or Equilibrium before the term"):
        problem.add(opt.Effort(1.0))
    problem.add(opt.Collocation([0.0], 1.0, 5))
    with pytest.raises(ValueError, match="one Collocation"):
        problem.add(opt.Collocation([0.0], 1.0, 5))
    first, second = problem.add(opt.Effort(1.0)), problem.add(opt.Effort(2.0))
    assert (first.name, second.name) == ("effort", "effort2")


def test_the_programs_variables_carry_the_params_scale_and_bounds():
    system, x = tanh_mass(bounds=(0.5, 50.0), scale=10.0)
    problem = tracking(system, x)
    problem.free(SPRING)
    nlp = problem.build()
    key = f"param:{SPRING}"
    where = nlp.variables.slices[key]
    assert nlp.variables.scales[key] == 10.0
    assert nlp.lbx[where] == pytest.approx(0.05) and nlp.ubx[where] == pytest.approx(5.0)
    assert nlp.x0[where] == pytest.approx(0.2)  # the stiffness 2.0 over the scale


def test_the_next_solve_starts_from_the_current_param_values():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    where = problem.build().variables.slices[f"param:{SPRING}"]
    assert problem.initial_guess()[where] == pytest.approx(2.0)
    system.params[SPRING].value = 20.0  # e.g. a result applied to the system
    assert problem.initial_guess()[where] == pytest.approx(20.0)


def test_a_matrix_param_comes_back_in_its_shape_and_round_trips():
    system, q = two_masses()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0, 0.0], 4.0, 31))
    problem.free("ctrl.damper.damping")
    problem.add(opt.Effort(1e-2))
    problem.add(opt.Cost(q - [1.0, 0.5], name="reach"))
    result = problem.solve()
    damping = result.params["ctrl.damper.damping"]
    assert result.converged and damping.shape == (2, 2)
    assert damping.min() >= -3.0 - 1e-6 and damping.max() <= 8.0 + 1e-6
    # the result, packed back into the program's variables, is the point the solver found
    nlp = problem.build()
    f, _ = nlp.functions()
    x = problem.initial_guess(result)
    assert float(f(x, np.zeros(0))) == pytest.approx(result.cost, rel=1e-9)
    result.apply(system)
    np.testing.assert_array_equal(system.params["ctrl.damper.damping"].value, damping)


def test_options_changed_after_a_solve_apply_to_the_next_one():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    assert problem.solve().converged
    problem.options["ipopt.max_iter"] = 2  # the solver is made again, the program is not
    nlp = problem.build()
    assert problem.solve().status == "Maximum_Iterations_Exceeded"
    assert problem.build() is nlp


def test_param_names_with_glob_characters_are_matched_as_they_are():
    system, x, _ = mass_spring()
    robot = system.robot
    del robot.components["mass"]
    robot.add("mass[1]", vmc.Inertance(x, 1.0))  # a name that is also a glob pattern
    name = "robot.mass[1].inertance"
    assert name in system.params
    system.params[name].bounds = (0.1, 10.0)
    problem = tracking(system, x, effort=0.0)
    assert problem.free("robot.mass[[]1].inertance") == [name]  # escaped by the caller
    assert float(problem.solve().params[name]) == pytest.approx(0.1, abs=1e-5)  # lighter is faster


def test_returning_true_from_the_progress_callback_stops_the_solve():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    seen = []

    def stop_at_two(iteration, cost, params, q):
        seen.append(iteration)
        return iteration == 2

    stopped = problem.solve(progress=stop_at_two)
    assert stopped.status == "User_Requested_Stop" and not stopped.converged
    assert stopped.iterations == 2 and seen == [0, 1, 2]
    assert problem.solve().converged  # the problem is fine afterwards


def test_an_error_in_the_progress_callback_is_raised_after_the_solver_stops():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)

    def buggy(iteration, cost, params, q):
        if iteration == 1:
            raise ZeroDivisionError("a bug in the callback")

    with pytest.raises(ZeroDivisionError, match="a bug in the callback"):
        problem.solve(progress=buggy)
    with pytest.raises(KeyboardInterrupt):
        problem.solve(progress=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert problem.solve().converged  # nothing is left over from the failed solves


def test_a_term_cannot_be_added_twice():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    reach = problem.add(opt.Cost(x - 1.0, name="again"))
    with pytest.raises(ValueError, match="'again' is in the problem already"):
        problem.add(reach)


def test_names_are_given_as_separate_strings():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    with pytest.raises(TypeError, match="separate strings, not a list"):
        problem.free([SPRING])
    with pytest.raises(TypeError, match="separate strings"):
        problem.parameter(system.params[SPRING])


def test_the_bounds_of_a_free_param_are_read_again_at_every_solve():
    system, x = tanh_mass(bounds=(0.5, 3.0))
    problem = tracking(system, x)
    problem.free(SPRING)
    nlp = problem.build()
    assert float(problem.solve().params[SPRING]) == pytest.approx(3.0, abs=1e-6)
    system.params[SPRING].bounds = (0.5, 100.0)  # widened: no rebuild needed
    wider = problem.solve()
    assert problem.build() is nlp and float(wider.params[SPRING]) > 5.0


def test_a_warm_start_with_a_misspelled_key_or_the_wrong_type_is_refused():
    system, x = tanh_mass()
    problem = tracking(system, x)
    problem.free(SPRING)
    cold = problem.solve()
    with pytest.raises(ValueError, match=r"warm_start has \['qq'\]"):
        problem.solve(warm_start={"qq": cold.q})
    with pytest.raises(TypeError, match="Result or a dict"):
        problem.solve(warm_start=[cold.q])


def test_a_result_keeps_its_own_copy_of_the_references():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    problem.parameter(GOAL)
    result = problem.solve()
    system.params[GOAL].value[0] = 0.25  # edited in place
    assert result.references[GOAL][0] == 1.0


def test_the_efficiency_of_a_robot_without_actuation_acts_on_the_plan():
    eta = "robot.efficiency.c1"
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    problem.parameter(eta)
    full, half = problem.solve({eta: 1.0}), problem.solve({eta: 0.5})
    assert not np.allclose(full.q, half.q, atol=1e-3)
    system.params[eta].value = 0.5  # a Param set on the system acts too, even if not declared
    kept = tracking(system, x)
    assert np.allclose(kept.solve().q, half.q, atol=1e-6)


def test_a_reference_of_the_wrong_size_is_refused():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    problem.parameter(GOAL)
    with pytest.raises(ValueError, match="needs 1 values, got 2"):
        problem.solve({GOAL: [1.0, 2.0]})


def test_ipopt_prints_nothing(tmp_path):
    script = tmp_path / "quiet.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parents[1])!r})\n"
        "from small_plan import mass_spring\n"
        "from virtualmodelcontrol import optimization as opt\n"
        "system, x, _ = mass_spring()\n"
        "problem = opt.Problem(system)\n"
        "problem.add(opt.Collocation([0.0], 2.0, 11))\n"
        "assert problem.solve().converged\n"
    )
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout == "" and "Ipopt" not in run.stderr


def test_apply_leaves_out_what_belongs_to_the_task_not_the_controller():
    system, x, _ = mass_spring()
    problem = tracking(system, x)
    problem.parameter("reach.ref")  # the target inside Cost: not a Param of the controller
    problem.free(GOAL)
    result = problem.solve({"reach.ref": [0.8]})
    assert "reach.ref" in result.references
    result.apply(system)
    assert system.params[GOAL].value == pytest.approx(result.params[GOAL])
    controller = vmc.VMCController(vmc.compile(system))
    result.apply(controller)
    assert controller.live_params()[GOAL] == pytest.approx(result.params[GOAL])
