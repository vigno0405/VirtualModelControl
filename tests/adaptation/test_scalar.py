"""The model-free laws of the hand and the finger, against the lab's own lines."""

from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceRatio, Stiffening
from virtualmodelcontrol.core.params import Param

DATA = np.load(Path(__file__).parents[1] / "data" / "heuristics.npz")


def controller(stiffness):
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(3))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("tip", vmc.LinearSpring(robot.joint(slice(0, 3)) - vmc.Ref("goal", 3), stiffness))
    return vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


@pytest.mark.parametrize("case", DATA["ratio_cases"])
def test_the_ratio_law_agrees_with_the_labs_update(case):
    f_meas, level, before, after = case[:3], case[3], case[4:7], case[7:10]
    c = controller(np.diag(before))
    law = ForceRatio(c, "ctrl.*.stiffness", max_change=float(DATA["max_change"]))
    law.step(c, f_meas, [level, 0.0, 0.0])
    np.testing.assert_allclose(c.live_params()["ctrl.tip.stiffness"], np.diag(after), rtol=1e-12)


def test_the_ratio_moves_towards_the_wanted_force_by_the_step_times_the_relative_error():
    c = controller(np.diag([40.0, 40.0, 40.0]))
    law = ForceRatio(c, "ctrl.*.stiffness", max_change=0.5)

    def ratio(measured, wanted):
        law.step(c, [0.0, 0.0, measured], [0.0, 0.0, wanted])
        return law.ratio

    # the relative error is |measured - wanted| / the larger of the two
    assert ratio(0.5, 0.4) == pytest.approx(0.9)  # too much force: softer by 0.5 * 0.2
    assert ratio(1.0, 0.5) == pytest.approx(0.75)  # by 0.5 * 0.5
    assert ratio(0.25, 0.5) == pytest.approx(1.25)  # too little: stiffer by 0.5 * 0.5
    assert ratio(0.49, 0.5) == pytest.approx(1.01)  # by 0.5 * 0.02


def test_the_wanted_over_the_measured_force_counts_only_where_the_step_allows_more():
    """For a step up to 1 the cap is always the one that binds; above it, the ratio can."""
    c = controller(np.diag([40.0, 40.0, 40.0]))
    law = ForceRatio(c, "ctrl.*.stiffness", max_change=2.0)
    law.step(c, [0.0, 0.0, 0.4], [0.0, 0.0, 0.5])  # relative error 0.2, cap 1 +- 0.4
    assert law.ratio == pytest.approx(1.25)  # wanted over measured, inside the cap
    law.step(c, [0.0, 0.0, 0.1], [0.0, 0.0, 0.5])  # relative error 0.8, cap 1 +- 1.6
    assert law.ratio == pytest.approx(2.6)  # the cap, below 0.5 / 0.1 = 5


def test_nothing_happens_while_the_force_is_near_zero():
    c = controller(np.diag([40.0, 40.0, 40.0]))
    before = c.live_params()["ctrl.tip.stiffness"].copy()
    law = ForceRatio(c, "ctrl.*.stiffness")
    assert law.step(c, [0.0, 0.0, 1e-9], [0.0, 0.0, 0.5]) == 0.0 and law.ratio == 1.0
    np.testing.assert_array_equal(c.live_params()["ctrl.tip.stiffness"], before)


def test_a_stiffness_stays_within_its_bounds_and_a_matrix_stays_positive_semidefinite():
    c = controller(2.0)  # a scalar stiffness, bounded below by zero
    ForceRatio(c, "ctrl.*.stiffness", max_change=1.0).step(c, [0.0, 0.0, 1.0], [0.0, 0.0, 1e-9])
    assert 0.0 <= float(c.live_params()["ctrl.tip.stiffness"]) < 2.0
    m = controller(np.array([[40.0, 5.0, 0.0], [5.0, 40.0, 0.0], [0.0, 0.0, 40.0]]))
    ForceRatio(m, "ctrl.*.stiffness").step(m, [0.0, 0.0, 0.3], [0.0, 0.0, 0.5])
    new = m.live_params()["ctrl.tip.stiffness"]
    np.testing.assert_allclose(new, new.T, atol=1e-12)
    assert np.linalg.eigvalsh(new).min() > 0.0


def test_through_a_tank_the_step_is_paid_for():
    c = controller(np.diag([40.0, 40.0, 40.0]))
    c.set({"ctrl.tip.goal": [0.1, 0.0, 0.0]})
    meas = vmc.Signals(0.0, motor_position=np.zeros(3), motor_velocity=np.zeros(3))
    c.reset(0.0, meas)
    c.step(0.0, meas)
    tank = vmc.control.Tank(c, level=0.0)
    law = ForceRatio(tank, "ctrl.*.stiffness")
    law.step(tank, [0.0, 0.0, 0.25], [0.0, 0.0, 0.5])  # more stiffness stores energy
    assert tank.fraction < 1e-9
    np.testing.assert_array_equal(c.live_params()["ctrl.tip.stiffness"], np.diag([40.0] * 3))
    law.step(tank, [0.0, 0.0, 1.0], [0.0, 0.0, 0.5])  # less stiffness gives energy back
    assert tank.fraction == 1.0 and tank.level > 0.0


@pytest.mark.parametrize("alpha, force, k", DATA["stiffening_cases"])
def test_the_stiffening_agrees_with_the_labs_function(alpha, force, k):
    c = controller(np.full(3, 0.1))  # one stiffness per axis, as in the lab's finger
    law = Stiffening(c, "ctrl.*.stiffness", float(DATA["k_min"]), float(DATA["k_max"]), alpha)
    law.step(c, force)
    assert law.k == pytest.approx(k, rel=1e-12)
    np.testing.assert_allclose(c.live_params()["ctrl.tip.stiffness"], np.full(3, k), rtol=1e-12)


def test_stiffening_fills_a_scalar_and_a_matrix_the_way_it_says():
    scalar = controller(1.0)
    Stiffening(scalar, "ctrl.*.stiffness", 1.0, 5.0, 2.0).step(scalar, 0.5)
    k = 1.0 + 4.0 * (1.0 - np.exp(-1.0))
    assert float(scalar.live_params()["ctrl.tip.stiffness"]) == pytest.approx(k)
    matrix = controller(np.eye(3))
    Stiffening(matrix, "ctrl.*.stiffness", 1.0, 5.0, 2.0).step(matrix, 0.5)
    np.testing.assert_allclose(matrix.live_params()["ctrl.tip.stiffness"], k * np.eye(3))


def bounded(upper):
    return Param("stiffness", 40.0, unit="N/m", scope="stage", bounds=(0.0, upper))


def test_both_laws_keep_a_stiffness_below_its_upper_bound():
    c = controller(bounded(45.0))
    ForceRatio(c, "ctrl.*.stiffness", max_change=1.0).step(c, [0.0, 0.0, 0.25], [0.0, 0.0, 0.5])
    assert float(c.live_params()["ctrl.tip.stiffness"]) == 45.0  # 40 * 1.5 is clipped
    c = controller(bounded(45.0))
    Stiffening(c, "ctrl.*.stiffness", 1.0, 100.0, 50.0).step(c, 1.0)  # k is about 100
    assert float(c.live_params()["ctrl.tip.stiffness"]) == 45.0


def test_a_matrix_that_is_not_symmetric_comes_out_as_its_symmetric_part_scaled():
    start = np.array([[40.0, 10.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]])
    c = controller(start)
    law = ForceRatio(c, "ctrl.*.stiffness")
    law.step(c, [0.0, 0.0, 0.4], [0.0, 0.0, 0.5])
    ratio = law.ratio
    assert ratio == pytest.approx(1.01)  # 0.05 times the relative error 0.2
    np.testing.assert_allclose(
        c.live_params()["ctrl.tip.stiffness"], ratio * 0.5 * (start + start.T), rtol=1e-12
    )


def test_the_ratio_law_scales_a_vector_of_stiffnesses_as_it_does_a_matrix():
    c = controller(np.array([10.0, 20.0, 30.0]))  # one stiffness per axis
    law = ForceRatio(c, "ctrl.*.stiffness")
    law.step(c, [0.0, 0.0, 0.4], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(c.live_params()["ctrl.tip.stiffness"], 1.01 * np.array([10, 20, 30]))


def test_the_stiffening_through_a_tank_applies_what_the_tank_pays_for():
    c = controller(np.full(3, 10.0))
    c.set({"ctrl.tip.goal": [0.1, 0.0, 0.0]})  # the spring is stretched
    meas = vmc.Signals(0.0, motor_position=np.zeros(3), motor_velocity=np.zeros(3))
    c.reset(0.0, meas)
    c.step(0.0, meas)
    tank = vmc.control.Tank(c, level=0.0)
    law = Stiffening(tank, "ctrl.*.stiffness", 10.0, 100.0, 1.0)
    jump = law.step(tank, 1.0)  # stiffer: stores energy, which the empty tank cannot pay
    assert law.k > 10.0 and tank.fraction < 1e-9 and abs(jump) < 1e-9
    np.testing.assert_allclose(c.live_params()["ctrl.tip.stiffness"], np.full(3, 10.0))
    tank.level = 10.0
    jump = law.step(tank, 1.0)
    assert jump > 0 and tank.fraction == 1.0
    np.testing.assert_allclose(c.live_params()["ctrl.tip.stiffness"], np.full(3, law.k))
