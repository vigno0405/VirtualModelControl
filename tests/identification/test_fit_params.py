"""fit_params: the Params of a robot from logged motion, in a linear and in a nonlinear case."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.identification import fit_params
from virtualmodelcontrol.identification.params import _undetermined

TRUE = {"mass": 2.0, "stiffness": 40.0, "damping": 1.5}


def param(name, value, unit, lower=0.0):
    return vmc.Param(name, value, unit=unit, bounds=(lower, np.inf), scope="design")


def slider(mass=2.0, stiffness=40.0, damping=1.5, tanh=None, floor=0.0):
    """A mass on a spring and a damper (or, with ``tanh``, a saturating damper (D, F)). Its
    mass is bounded below by ``floor``."""
    robot = vmc.Mechanism(
        "slider", model=vmc.models.JointSpace(1, unit="m"), actuation=vmc.models.Direct(1.0)
    )
    x = robot.joint(0)
    robot.add("m", vmc.Inertance(x, param("mass", mass, "kg", floor)))
    robot.add("spring", vmc.LinearSpring(x, param("k", stiffness, "N/m")))
    if tanh is None:
        robot.add("damper", vmc.LinearDamper(x, param("d", damping, "N·s/m")))
    else:
        robot.add(
            "damper",
            vmc.TanhDamper(x, param("D", tanh[0], "N·s/m"), param("F", tanh[1], "N")),
        )
    return robot


def record(robot, seconds=6.0, dt=1 / 200, noise=0.0, exact_a=True):
    """A run under a chirp-like torque: the time, q, v, the torque, and the exact acceleration."""
    plant = vmc.sim.ModelPlant(robot, max_step=1e-3)
    rng = np.random.default_rng(1)
    rows = {key: [] for key in ("t", "q", "v", "u", "a")}
    for _ in range(int(seconds / dt)):
        t = plant.t
        u = np.array([3.0 * np.sin(2.0 * t) + 2.0 * np.sin(7.0 * t * (1 + 0.2 * t))])
        plant.write(vmc.Signals(t, motor_torque=u))
        a = np.array(plant.dynamics.forward(plant.q, plant.v, u, plant.p, t)).ravel()
        values = (t, plant.q.copy(), plant.v.copy(), u + rng.normal(0, noise, 1), a)
        for key, value in zip(rows, values, strict=True):
            rows[key].append(value)
        plant.advance(dt)
    rows = {key: np.array(value) for key, value in rows.items()}
    if not exact_a:
        del rows["a"]
    return rows


def test_a_residual_linear_in_the_params_is_solved_from_a_wrong_start():
    run = record(slider())
    fit = fit_params(slider(1.0, 10.0, 0.2), ["m.inertance", "spring.*", "damper.damping"], [run])
    got = {k: float(v) for k, v in fit.values.items()}
    assert got["m.inertance"] == pytest.approx(TRUE["mass"], rel=1e-6)
    assert got["spring.stiffness"] == pytest.approx(TRUE["stiffness"], rel=1e-6)
    assert got["damper.damping"] == pytest.approx(TRUE["damping"], rel=1e-6)
    assert fit.rms < 1e-6  # [N]: nothing is left over


def test_a_param_that_enters_nonlinearly_converges_from_a_wrong_start():
    run = record(slider(tanh=(8.0, 2.0)))
    fit = fit_params(slider(tanh=(3.0, 1.0)), ["m.inertance", "spring.*", "damper.*"], [run])
    assert float(fit.values["damper.damping"]) == pytest.approx(8.0, rel=1e-5)
    assert float(fit.values["damper.max_force"]) == pytest.approx(2.0, rel=1e-5)
    assert float(fit.values["spring.stiffness"]) == pytest.approx(40.0, rel=1e-5)


def test_without_accelerations_in_the_run_they_come_from_the_velocity():
    run = record(slider(), exact_a=False)
    fit = fit_params(
        slider(1.0, 10.0, 0.2), ["m.inertance", "spring.*", "damper.*"], [run], smoothing=11
    )
    assert float(fit.values["m.inertance"]) == pytest.approx(2.0, rel=0.02)
    assert float(fit.values["spring.stiffness"]) == pytest.approx(40.0, rel=0.02)
    assert float(fit.values["damper.damping"]) == pytest.approx(1.5, rel=0.02)


def test_the_params_bounds_hold():
    run = record(slider())
    fit = fit_params(slider(1.0, floor=3.0), ["m.inertance"], [run])  # starts under its bound
    assert float(fit.values["m.inertance"]) == pytest.approx(3.0, abs=1e-6)  # the bound, not 2


def test_the_standard_errors_say_how_far_the_values_may_be_off():
    run = record(slider(), noise=0.3)
    fit = fit_params(slider(1.0, 10.0, 0.2), ["m.inertance", "spring.*", "damper.*"], [run])
    for name, true in (("m.inertance", 2.0), ("spring.stiffness", 40.0), ("damper.damping", 1.5)):
        error, std = abs(float(fit.values[name]) - true), float(fit.std[name])
        assert 0.0 < std < 0.2 * true  # a good fit
        assert error < 4 * std  # and its error is in its own error bars
    assert fit.rms == pytest.approx(0.3, rel=0.15)  # the noise on the torque is what is left


def test_the_samples_not_in_the_train_mask_are_left_out():
    run = record(slider())
    broken = {key: value.copy() for key, value in run.items()}
    broken["u"][::2] += 50.0  # every other torque is wrong...
    broken["train"] = np.arange(len(run["t"])) % 2 == 1  # ...and not used
    fit = fit_params(slider(1.0, 10.0, 0.2), ["m.inertance", "spring.*", "damper.*"], [broken])
    assert float(fit.values["m.inertance"]) == pytest.approx(2.0, rel=1e-6)


def pair_run():
    """A robot of two masses on a spring with a stiffness matrix, and a run of it."""
    robot = vmc.Mechanism(
        "pair", model=vmc.models.JointSpace(2, unit="m"), actuation=vmc.models.Direct(1.0)
    )
    both = robot.joint(slice(0, 2))
    for i in range(2):
        robot.add(f"m{i}", vmc.Inertance(robot.joint(i), 1.0 + i))
    stiffness = vmc.Param("K", [[30.0, 5.0], [5.0, 20.0]], unit="N/m", scope="design")
    robot.add("spring", vmc.LinearSpring(both, stiffness))
    robot.add("damper", vmc.LinearDamper(both, 1.0))
    plant = vmc.sim.ModelPlant(robot, max_step=1e-3)
    rows = {key: [] for key in ("t", "q", "v", "u", "a")}
    for _ in range(1200):
        t = plant.t
        u = np.array([2.0 * np.sin(3.0 * t), 2.0 * np.sin(5.0 * t + 1.0)])
        plant.write(vmc.Signals(t, motor_torque=u))
        a = np.array(plant.dynamics.forward(plant.q, plant.v, u, plant.p, t)).ravel()
        for key, value in zip(rows, (t, plant.q.copy(), plant.v.copy(), u, a), strict=True):
            rows[key].append(value)
        plant.advance(1 / 200)
    return robot, {key: np.array(value) for key, value in rows.items()}


def test_a_matrix_param_comes_back_in_its_own_shape():
    robot, rows = pair_run()
    start = np.array([[10.0, 4.0], [0.0, 10.0]])  # a wrong stiffness, not symmetric
    robot.components["spring"].stiffness.value = start
    fit = fit_params(robot, ["spring.stiffness"], [rows])
    found, std = fit.values["spring.stiffness"], fit.std["spring.stiffness"]
    assert found.shape == (2, 2) and std.shape == (2, 2)
    np.testing.assert_allclose(0.5 * (found + found.T), [[30.0, 5.0], [5.0, 20.0]], rtol=1e-5)
    # a spring only feels the symmetric part: the data fix K[0, 1] + K[1, 0] but not the entries on
    # their own, so those have no standard error, and the fit leaves their difference where it
    # started (entry [0, 1] above [1, 0] by 4, which a transposed layout would flip), on every
    # machine
    assert np.isfinite(np.diag(std)).all() and np.isinf([std[0, 1], std[1, 0]]).all()
    assert found[0, 1] - found[1, 0] == pytest.approx(4.0, abs=1e-6)


def test_an_unseen_direction_stays_where_it_was_unless_the_bounds_forbid_it():
    robot, rows = pair_run()
    spring = robot.components["spring"].stiffness
    # K[0, 1] <= 7 and K[1, 0] <= 3 leave only the corner (7, 3) for the sum of 10: the unseen
    # difference cannot go back to the start's (7), and the fit stays at the corner
    spring.bounds = ([[0.0, 0.0], [0.0, 0.0]], [[100.0, 7.0], [3.0, 100.0]])
    spring.value = np.array([[10.0, 8.0], [0.0, 10.0]])
    found = fit_params(robot, ["spring.stiffness"], [rows]).values["spring.stiffness"]
    assert found[0, 1] == pytest.approx(7.0, abs=1e-4) and found[1, 0] == pytest.approx(
        3.0, abs=1e-4
    )


def test_a_move_the_residual_feels_is_not_taken_back():
    # at one speed only, a saturating damper's two Params show in one number, F tanh(D v / F): the
    # pairs that fit equally lie on a curve, and a straight step back towards the start leaves it
    t = np.arange(0.0, 2.0, 1 / 200)
    q, v, a = np.zeros((t.size, 1)), np.full((t.size, 1), 0.5), np.zeros((t.size, 1))
    force = 0.8 * np.tanh(1.5 * 0.5 / 0.8)  # what the damper gives at that speed
    run = {"t": t, "q": q, "v": v, "u": np.full_like(q, force), "a": a}
    fit = fit_params(slider(tanh=(0.2, 3.0)), ["damper.*"], [run])
    assert fit.rms < 1e-6  # the fit explains the data, so the move was not undone
    assert np.isinf([float(fit.std["damper.damping"]), float(fit.std["damper.max_force"])]).all()


def test_a_param_that_the_run_never_excites_has_an_infinite_standard_error():
    t = np.arange(0.0, 6.0, 1 / 200)
    q = 0.1 * np.sin(3.0 * t)[:, None]
    zero = np.zeros_like(q)
    # the mass sees only the acceleration and the damper only the velocity, both zero here
    still = {"t": t, "q": q, "v": zero, "u": 40.0 * q, "a": zero}
    fit = fit_params(slider(1.0, 10.0, 0.2), ["m.inertance", "spring.*", "damper.damping"], [still])
    assert float(fit.values["spring.stiffness"]) == pytest.approx(40.0, rel=1e-6)
    assert 0.0 <= float(fit.std["spring.stiffness"]) < 1e-6
    for name, start in (("m.inertance", 1.0), ("damper.damping", 0.2)):
        assert float(fit.values[name]) == start  # nothing moved it
        assert float(fit.std[name]) == np.inf  # and nothing says how well it is known


def test_columns_that_the_data_cannot_tell_apart_are_undetermined():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 50))
    assert not _undetermined(np.column_stack([a, b, c])).any()
    assert not _undetermined(np.column_stack([a, a + 1e-3 * b, c])).any()  # close, not the same
    assert _undetermined(np.column_stack([a, a + 1e-7 * b, c])).tolist() == [True, True, False]
    assert _undetermined(np.column_stack([a, 1e6 * a, c])).tolist() == [True, True, False]  # units
    assert _undetermined(np.column_stack([a, 0.0 * b, c])).tolist() == [False, True, False]


def test_a_name_that_matches_nothing_is_an_error():
    with pytest.raises(ValueError, match="no Param"):
        fit_params(slider(), ["nothing.*"], [record(slider(), seconds=1.0)])


def test_a_stride_keeps_every_nth_sample():
    run = record(slider())
    broken = {key: value.copy() for key, value in run.items()}
    off_grid = np.arange(len(run["t"])) % 3 != 0
    broken["u"][off_grid] += 50.0  # wrong everywhere the stride does not look
    names = ["m.inertance", "spring.*", "damper.*"]
    fit = fit_params(slider(1.0, 10.0, 0.2), names, [broken], stride=3)
    assert float(fit.values["m.inertance"]) == pytest.approx(2.0, rel=1e-6)


def test_values_and_standard_errors_are_those_of_ordinary_least_squares():
    """For a residual linear in the Params the fit is a regression, so numpy's gives the same:
    m a + k q + d v = u, its estimates and their covariance s² (XᵀX)⁻¹ with N − 3 degrees."""
    run = record(slider(), noise=0.3)
    X = np.column_stack([run["a"][:, 0], run["q"][:, 0], run["v"][:, 0]])
    y = run["u"][:, 0]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = np.sum((y - X @ theta) ** 2) / (len(y) - 3)
    std = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    names = ["m.inertance", "spring.stiffness", "damper.damping"]
    fit = fit_params(slider(1.0, 10.0, 0.2), names, [run])
    np.testing.assert_allclose([float(fit.values[n]) for n in names], theta, rtol=1e-6)
    np.testing.assert_allclose([float(fit.std[n]) for n in names], std, rtol=1e-4)
    assert fit.rms == pytest.approx(np.sqrt(np.mean((y - X @ theta) ** 2)), rel=1e-6)
