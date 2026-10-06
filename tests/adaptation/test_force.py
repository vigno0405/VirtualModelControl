from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "adaptation.npz")
T_WB = DATA["t_wb"]  # the arm's base in the lab's world frame; the library's arm sits at the origin
SITES = {"tip_normal": 1.0, "tip_free": 1.0, "mid_normal": 0.5}
CASES = [f"{name}/{i}" for name in SITES for i in range(3)]


@pytest.fixture(scope="module")
def setup():
    arm = helyx.arm("290-145-145", efficiency=float(DATA["eta"]))
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.5, 1.0), 1):
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - vmc.Ref(f"goal{i}", 3), np.eye(3)))
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    return vmc.compile(system, runtime=["ctrl.*"])


def controller_at(compiled, key):
    """A controller that has just stepped at the case's motor angles, stiffnesses and references."""
    theta = DATA[f"{key}/q"]
    controller = vmc.VMCController(compiled)
    values = {}
    for i in (0, 1):
        values[f"ctrl.s{i + 1}.stiffness"] = DATA[f"{key}/Kd"][i]
        values[f"ctrl.s{i + 1}.goal{i + 1}"] = DATA[f"{key}/d_ref"][i] - T_WB
    controller.set(values)
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    return controller


def law(compiled, key, params, **kwargs):
    normal = DATA[f"{key}/normal"]
    site = SITES[key.split("/")[0]]
    return ForceTracking(
        controller_at(compiled, key), site, params, None if not normal.any() else normal, **kwargs
    )


@pytest.mark.parametrize("key", CASES)
def test_reference_directions_agree_with_the_labs_gradient(setup, key):
    force = law(setup, key, "ctrl.*.goal*")
    g = force.direction(controller_at(setup, key), DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    got = np.array([g["ctrl.s1.goal1"], g["ctrl.s2.goal2"]])
    np.testing.assert_allclose(got, DATA[f"{key}/gref"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("key", CASES)
def test_stiffness_directions_agree_with_the_symmetric_part_of_the_labs_gradient(setup, key):
    force = law(setup, key, "ctrl.*.stiffness")
    g = force.direction(controller_at(setup, key), DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    for i, lab in enumerate(DATA[f"{key}/gK"]):
        np.testing.assert_allclose(
            g[f"ctrl.s{i + 1}.stiffness"], 0.5 * (lab + lab.T), rtol=1e-8, atol=1e-9
        )


@pytest.mark.parametrize("key", [c for c in CASES if not c.startswith("mid")])
@pytest.mark.parametrize("cap, max_step", [("cap", None), ("geo", 1e-5)])
def test_steps_agree_with_the_labs_step_without_a_tank(setup, key, cap, max_step):
    controller = controller_at(setup, key)
    force = law(setup, key, "ctrl.*.goal*", max_step=max_step)
    alpha, _ = force.step(controller, DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    assert alpha == pytest.approx(float(DATA[f"{key}/{cap}/alpha"]), rel=1e-5)
    live = controller.live_params()
    new = np.array([live["ctrl.s1.goal1"], live["ctrl.s2.goal2"]]) + T_WB
    np.testing.assert_allclose(new, DATA[f"{key}/{cap}/new"], rtol=1e-9, atol=1e-10)


def test_the_law_reads_the_controllers_params_as_they_are_now(setup):
    key = "tip_normal/0"
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    controller = controller_at(setup, key)
    force = law(setup, key, "ctrl.*.goal*")
    before = force.direction(controller, f_meas, f_des)
    controller.set({"ctrl.s1.stiffness": 3.0 * DATA[f"{key}/Kd"][0]})  # after its last step
    after = force.direction(controller, f_meas, f_des)
    np.testing.assert_allclose(after["ctrl.s2.goal2"], before["ctrl.s2.goal2"], rtol=1e-12)
    np.testing.assert_allclose(after["ctrl.s1.goal1"], 3.0 * before["ctrl.s1.goal1"], rtol=1e-10)


def test_no_step_where_the_force_is_already_right(setup):
    key = "tip_normal/0"
    controller = controller_at(setup, key)
    before = controller.live_params()
    force = law(setup, key, "ctrl.*.goal*")
    assert force.step(controller, DATA[f"{key}/f_des"], DATA[f"{key}/f_des"]) == (0.0, 0.0)
    for name, value in controller.live_params().items():
        np.testing.assert_array_equal(value, before[name])


def test_a_tank_applies_the_part_of_the_step_it_pays_for(setup):
    key = "tip_normal/0"
    f_meas, f_des = DATA[f"{key}/f_des"], DATA[f"{key}/f_meas"]  # the opposite of the lab's case
    free = controller_at(setup, key)
    alpha, jump = law(setup, key, "ctrl.*.goal*").step(free, f_meas, f_des)
    assert jump > 0  # this step stores energy in the springs

    rich = vmc.control.Tank(controller_at(setup, key), level=10.0 * jump)
    assert law(setup, key, "ctrl.*.goal*").step(rich, f_meas, f_des) == (alpha, jump)
    assert rich.fraction == 1.0 and rich.level == pytest.approx(9.0 * jump)

    poor = vmc.control.Tank(controller_at(setup, key), level=0.25 * jump)
    start = poor.live_params()
    _, paid = law(setup, key, "ctrl.*.goal*").step(poor, f_meas, f_des)
    assert 0.0 < poor.fraction < 1.0 and paid == pytest.approx(0.25 * jump, rel=1e-6)
    for name in ("ctrl.s1.goal1", "ctrl.s2.goal2"):
        moved = poor.live_params()[name] - start[name]
        full = free.live_params()[name] - start[name]
        np.testing.assert_allclose(moved, poor.fraction * full, rtol=1e-6, atol=1e-12)


def test_a_param_that_is_not_live_is_refused(setup):
    with pytest.raises(ValueError, match="no live Param matches"):
        ForceTracking(controller_at(setup, "tip_normal/0"), 1.0, "ctrl.*.nothing")


def test_a_step_that_releases_energy_goes_through_a_tank_and_refills_it(setup):
    key = "tip_normal/0"
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    empty = vmc.control.Tank(controller_at(setup, key), level=0.0)
    _, jump = law(setup, key, "ctrl.*.goal*").step(empty, f_meas, f_des)
    assert jump < 0 and empty.fraction == 1.0
    assert empty.level == pytest.approx(-jump)


def test_the_normal_may_have_any_length(setup):
    key = "tip_normal/1"
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    controller = controller_at(setup, key)
    unit = ForceTracking(controller, 1.0, "ctrl.*.goal*", DATA[f"{key}/normal"])
    long = ForceTracking(controller, 1.0, "ctrl.*.goal*", 7.0 * DATA[f"{key}/normal"])
    a, b = unit.direction(controller, f_meas, f_des), long.direction(controller, f_meas, f_des)
    for name in a:
        np.testing.assert_allclose(b[name], a[name], rtol=1e-12)


@pytest.fixture(scope="module")
def scalar_setup():
    """The same two springs with one stiffness each, the isotropic gains the lab adapts."""
    arm = helyx.arm("290-145-145", efficiency=float(DATA["eta"]))
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.5, 1.0), 1):
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - vmc.Ref(f"goal{i}", 3), 1.0))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.*"])


def stiff_controller(compiled, key):
    controller = vmc.VMCController(compiled)
    values = {}
    for i in (0, 1):
        values[f"ctrl.s{i + 1}.stiffness"] = DATA[f"{key}/k"][i]
        values[f"ctrl.s{i + 1}.goal{i + 1}"] = DATA[f"{key}/d_ref"][i] - T_WB
    controller.set(values)
    meas = vmc.Signals(0.0, motor_position=DATA[f"{key}/q"], motor_velocity=np.zeros(9))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    return controller


@pytest.mark.parametrize("key", ["stiff/0", "stiff/1", "stiff/2"])
@pytest.mark.parametrize("cap, max_step", [("cap", None), ("geo", 0.5)])
def test_stiffness_steps_agree_with_the_labs_step_without_a_tank(scalar_setup, key, cap, max_step):
    controller = stiff_controller(scalar_setup, key)
    force = ForceTracking(
        controller, 1.0, "ctrl.*.stiffness", DATA[f"{key}/normal"], max_step=max_step
    )
    alpha, _ = force.step(controller, DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    assert alpha == pytest.approx(float(DATA[f"{key}/{cap}/alpha"]), rel=1e-5)
    live = controller.live_params()
    new = [float(live["ctrl.s1.stiffness"]), float(live["ctrl.s2.stiffness"])]
    np.testing.assert_allclose(new, DATA[f"{key}/{cap}/new"], rtol=1e-9, atol=1e-12)


def test_a_stiffness_never_goes_negative(scalar_setup):
    key = "stiff/2"  # the second spring is at 0.2 N/m and the step takes more than that
    controller = stiff_controller(scalar_setup, key)
    force = ForceTracking(controller, 1.0, "ctrl.*.stiffness", DATA[f"{key}/normal"])
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    g = force.direction(controller, f_meas, f_des)
    alpha, _ = force.step(controller, f_meas, f_des)
    assert 0.2 + alpha * float(g["ctrl.s2.stiffness"]) < 0.0  # the raw step goes below zero
    assert float(controller.live_params()["ctrl.s2.stiffness"]) == 0.0


def test_a_stiffness_matrix_stays_symmetric_and_positive_semidefinite(setup):
    key = "tip_normal/0"
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    controller = controller_at(setup, key)
    force = law(setup, key, "ctrl.*.stiffness", max_force_step=1e4)  # a step far too large
    start, g = controller.live_params(), force.direction(controller, f_meas, f_des)
    alpha, _ = force.step(controller, f_meas, f_des)
    for name in ("ctrl.s1.stiffness", "ctrl.s2.stiffness"):
        raw = start[name] + alpha * g[name]
        assert np.linalg.eigvalsh(0.5 * (raw + raw.T)).min() < 0.0  # the raw step breaks it
        new = controller.live_params()[name]
        np.testing.assert_allclose(new, new.T, atol=1e-12)
        assert np.linalg.eigvalsh(new).min() >= -1e-9
        np.testing.assert_allclose(new, vmc.control.project_psd(raw), rtol=1e-9, atol=1e-9)
