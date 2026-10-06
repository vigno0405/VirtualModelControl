"""The finger's fixed-rate laws against the lab's: joint-space stiffness and reference descent."""

from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.estimation import ContactForce
from virtualmodelcontrol.robots import adapt

DATA = np.load(Path(__file__).parents[1] / "data" / "fingertip.npz")
CASES = range(6)
NORMAL = DATA["normal"]
RATES = {
    "stiffness": float(DATA["lr_stiffness"]),
    "reference": float(DATA["lr_reference"]),
    "big": 5e-2,
}


@pytest.fixture(scope="module")
def compiled():
    finger = adapt.finger(efficiency=adapt.MOTOR_EFFICIENCY)
    ctrl = vmc.Mechanism("ctrl")
    spring = vmc.LinearSpring(adapt.joint_angles(finger) - vmc.Ref("theta_ref", 3), np.eye(3))
    ctrl.add("hold", spring)
    return vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl))


def controller_at(compiled, i):
    key = f"case/{i}"
    controller = vmc.VMCController(compiled)
    controller.set(
        {
            "ctrl.hold.stiffness": DATA[f"{key}/K"],
            "ctrl.hold.theta_ref": np.radians(DATA[f"{key}/theta_ref"]),
        }
    )
    meas = vmc.Signals(0.0, motor_position=np.radians(DATA[f"{key}/q"]), motor_velocity=np.zeros(2))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    return controller


def test_the_efficiencies_are_the_labs():
    np.testing.assert_allclose(DATA["eta"], adapt.MOTOR_EFFICIENCY, atol=0)


@pytest.mark.parametrize("i", CASES)
def test_the_force_agrees_with_the_labs_tip_force(compiled, i):
    controller = controller_at(compiled, i)
    bare = adapt.finger(efficiency=adapt.MOTOR_EFFICIENCY)  # the finger alone holds nothing
    force = ContactForce(controller, "tip", NORMAL, robot=bare)(controller)
    np.testing.assert_allclose(force, DATA[f"case/{i}/force"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("tag", RATES)
@pytest.mark.parametrize("i", CASES)
def test_stiffness_descent_agrees_with_the_labs_at_its_learning_rate(compiled, i, tag):
    key = f"case/{i}"
    controller = controller_at(compiled, i)
    law = ForceTracking(controller, "tip", "ctrl.hold.stiffness", NORMAL, rate=RATES[tag])
    law.step(controller, DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    assert law.alpha == RATES[tag]
    lab = DATA[f"{key}/K_{tag}"]  # the lab does not symmetrize; the spring acts through the part
    np.testing.assert_allclose(
        controller.live_params()["ctrl.hold.stiffness"], 0.5 * (lab + lab.T), rtol=1e-9, atol=1e-14
    )


@pytest.mark.parametrize("tag", RATES)
@pytest.mark.parametrize("i", CASES)
def test_reference_descent_agrees_with_the_labs_at_its_learning_rate(compiled, i, tag):
    key = f"case/{i}"
    controller = controller_at(compiled, i)
    law = ForceTracking(controller, "tip", "ctrl.hold.theta_ref", NORMAL, rate=RATES[tag])
    law.step(controller, DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"])
    new = np.degrees(controller.live_params()["ctrl.hold.theta_ref"])
    np.testing.assert_allclose(new, DATA[f"{key}/ref_{tag}"], rtol=1e-9, atol=1e-12)


def test_a_rate_replaces_the_force_cap_but_not_the_step_cap(compiled):
    key = "case/0"
    f_meas, f_des = DATA[f"{key}/f_meas"], DATA[f"{key}/f_des"]
    capped = controller_at(compiled, 0)
    law = ForceTracking(capped, "tip", "ctrl.hold.theta_ref", NORMAL, max_force_step=1e-9, rate=0.5)
    law.step(capped, f_meas, f_des)
    assert law.alpha == 0.5  # not 1e-9 divided by the force change
    small = controller_at(compiled, 0)
    law = ForceTracking(small, "tip", "ctrl.hold.theta_ref", NORMAL, rate=0.5, max_step=1e-3)
    law.step(small, f_meas, f_des)
    assert law.alpha < 0.5
    moved = small.live_params()["ctrl.hold.theta_ref"] - np.radians(DATA[f"{key}/theta_ref"])
    assert np.linalg.norm(moved) == pytest.approx(1e-3, rel=1e-9)


def test_on_the_hand_a_step_changes_the_estimate_by_the_force_cap_towards_the_wanted_force():
    hand = adapt.hand()
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("index", vmc.LinearSpring(hand.point("index/tip") - vmc.Ref("goal", 3), 50.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(hand, ctrl)))
    n = hand.model.space.nq
    meas = vmc.Signals(0.0, motor_position=np.full(n, 0.2), motor_velocity=np.zeros(n))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    estimate = ContactForce(controller, "index/tip", NORMAL)
    law = ForceTracking(controller, "index/tip", "ctrl.index.goal", NORMAL, max_force_step=0.01)
    wanted = np.array([0.0, 0.0, 1.0])
    before = estimate(controller)
    law.step(controller, before, wanted)  # the force is linear in the goal: the cap is exact
    after = estimate(controller)
    assert np.linalg.norm(after - before) == pytest.approx(0.01, rel=1e-6)
    assert abs(after[2] - 1.0) < abs(before[2] - 1.0)


@pytest.mark.parametrize("i", CASES)
def test_without_a_normal_the_finger_laws_agree_with_the_labs(compiled, i):
    """The two motors move the tip in a plane: the Jacobian has rank 2 and the map needs pinv."""
    key = f"case/{i}"
    f_meas, f_des = DATA[f"{key}/free/f_meas"], DATA[f"{key}/free/f_des"]
    controller = controller_at(compiled, i)
    bare = adapt.finger(efficiency=adapt.MOTOR_EFFICIENCY)
    force = ContactForce(controller, "tip", robot=bare)(controller)
    np.testing.assert_allclose(force, DATA[f"{key}/free/force"], rtol=1e-9, atol=1e-12)
    stiffness = ForceTracking(controller, "tip", "ctrl.hold.stiffness", rate=5e-2)
    stiffness.step(controller, f_meas, f_des)
    lab = DATA[f"{key}/free/K_big"]  # the lab does not project; the library keeps K a spring
    np.testing.assert_allclose(
        controller.live_params()["ctrl.hold.stiffness"],
        vmc.control.project_psd(lab),
        rtol=1e-9,
        atol=1e-12,
    )
    other = controller_at(compiled, i)
    ForceTracking(other, "tip", "ctrl.hold.theta_ref", rate=5e-2).step(other, f_meas, f_des)
    new = np.degrees(other.live_params()["ctrl.hold.theta_ref"])
    np.testing.assert_allclose(new, DATA[f"{key}/free/ref_big"], rtol=1e-9, atol=1e-12)
