from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import StiffnessTracking
from virtualmodelcontrol.core.params import Param
from virtualmodelcontrol.estimation import TaskStiffness
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "adaptation.npz")
T_WB = DATA["t_wb"]  # the arm's base in the lab's world frame; the library's arm sits at the origin
CASES = [f"inv/{i}" for i in range(3)]


@pytest.fixture(scope="module")
def compiled():
    arm = helyx.arm("290-145-145", efficiency=float(DATA["eta"]))
    stiffness = Param("stiffness", DATA["k_struct"], unit="N/m", scope="design")
    arm.add("structure", vmc.LinearSpring(arm.joint(slice(0, arm.model.space.nq)), stiffness))
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.5, 1.0), 1):
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - vmc.Ref(f"goal{i}", 3), np.eye(3)))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.*"])


def controller_at(compiled, key, goals=None, stiffness=10.0):
    controller = vmc.VMCController(compiled)
    goals = DATA[f"{key}/d_ref"] - T_WB if goals is None else goals
    controller.set(
        {
            "ctrl.s1.goal1": goals[0],
            "ctrl.s2.goal2": goals[1],
            "ctrl.s1.stiffness": stiffness * np.eye(3),
            "ctrl.s2.stiffness": 2 * stiffness * np.eye(3),
        }
    )
    meas = vmc.Signals(0.0, motor_position=DATA[f"{key}/q"], motor_velocity=np.zeros(9))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    return controller


def law(controller, key, **kwargs):
    normal = DATA[f"{key}/normal"]
    return StiffnessTracking(
        controller, 1.0, "ctrl.*.stiffness", None if not normal.any() else normal, **kwargs
    )


@pytest.mark.parametrize("key", CASES)
def test_the_springs_agree_with_the_labs_inversion_where_no_spring_pulls(compiled, key):
    controller = controller_at(compiled, key)  # the goals are on the springs' points
    target = law(controller, key).target(controller, DATA[f"{key}/K_des"], f_ext=np.zeros(3))
    lab = DATA[f"{key}/Kd"]
    for i, name in enumerate(("ctrl.s1.stiffness", "ctrl.s2.stiffness")):
        np.testing.assert_allclose(
            target[name], np.ravel(lab[i], order="F"), rtol=1e-6, atol=1e-6 * np.abs(lab).max()
        )


@pytest.mark.parametrize("key, tolerance", [("inv/0", 1e-6), ("inv/1", 2e-3), ("inv/2", 1e-6)])
def test_a_step_all_the_way_gives_the_wanted_stiffness(compiled, key, tolerance):
    """With the springs pulling and a contact force, where the geometric terms count. Along a
    normal the solution has a slightly negative stiffness, which the projection takes away."""
    rng = np.random.default_rng(3)
    goals = DATA[f"{key}/d_ref"] - T_WB + rng.normal(0, 0.02, (2, 3))
    controller = controller_at(compiled, key, goals)
    normal = DATA[f"{key}/normal"]
    tracking = law(controller, key)
    stiffness = TaskStiffness(controller, 1.0, None if not normal.any() else normal)
    f_ext = np.zeros(3) if not normal.any() else 0.4 * normal
    K_des = DATA[f"{key}/K_des"]
    tracking.step(controller, K_des, f_ext)
    scale = np.abs(K_des).max()
    np.testing.assert_allclose(stiffness(controller, f_ext), K_des, rtol=0, atol=tolerance * scale)
    for name in ("ctrl.s1.stiffness", "ctrl.s2.stiffness"):  # what is set can be a spring
        new = controller.live_params()[name]
        assert np.linalg.eigvalsh(new).min() >= -1e-9 and np.allclose(new, new.T)


def test_a_step_moves_the_part_of_the_way_the_rate_says(compiled):
    key = "inv/0"
    controller = controller_at(compiled, key)
    tracking = law(controller, key, rate=0.25)
    before = controller.live_params()
    aim = tracking.target(controller, DATA[f"{key}/K_des"], np.zeros(3))
    tracking.step(controller, DATA[f"{key}/K_des"], np.zeros(3))
    for name, value in aim.items():
        here = np.ravel(before[name], order="F")
        np.testing.assert_allclose(
            np.ravel(controller.live_params()[name], order="F"),
            here + 0.25 * (value - here),
            rtol=1e-9,
            atol=1e-9,
        )


def test_the_target_is_a_fixed_point_and_leaves_out_what_does_not_show(compiled):
    """The smallest-norm solution has nothing in the null space: asking again changes nothing,
    and a starting stiffness that does not show at the site is gone from the target."""
    key = "inv/0"
    controller = controller_at(compiled, key, stiffness=50.0)
    tracking = law(controller, key)
    aim = tracking.target(controller, DATA[f"{key}/K_des"], np.zeros(3))
    controller.set({n: np.reshape(v, (3, 3), order="F") for n, v in aim.items()})
    meas = vmc.Signals(0.0, motor_position=DATA[f"{key}/q"], motor_velocity=np.zeros(9))
    controller.step(0.0, meas)
    again = tracking.target(controller, DATA[f"{key}/K_des"], np.zeros(3))
    for name, value in aim.items():
        np.testing.assert_allclose(again[name], value, rtol=1e-6, atol=1e-6)
    other = controller_at(compiled, key, stiffness=500.0)  # another start, the same target
    for name, value in aim.items():
        np.testing.assert_allclose(
            law(other, key).target(other, DATA[f"{key}/K_des"], np.zeros(3))[name],
            value,
            rtol=1e-6,
            atol=1e-6,
        )


def test_without_a_force_the_one_of_the_model_is_used(compiled):
    key = "inv/0"
    rng = np.random.default_rng(5)
    goals = DATA[f"{key}/d_ref"] - T_WB + rng.normal(0, 0.02, (2, 3))  # the springs pull
    controller = controller_at(compiled, key, goals)
    tracking = law(controller, key)
    model_force = TaskStiffness(controller, 1.0, None).force(controller)
    assert np.abs(model_force).max() > 0.1
    by_default = tracking.target(controller, DATA[f"{key}/K_des"])
    given = tracking.target(controller, DATA[f"{key}/K_des"], model_force)
    for name in given:
        np.testing.assert_allclose(by_default[name], given[name], rtol=1e-12)
    without = tracking.target(controller, DATA[f"{key}/K_des"], np.zeros(3))
    assert any(np.abs(without[n] - given[n]).max() > 1e-6 for n in given)
