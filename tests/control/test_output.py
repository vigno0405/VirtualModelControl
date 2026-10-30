"""Output stages against the original controllers' output formulas and friction compensation."""

from pathlib import Path

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import output
from virtualmodelcontrol.core.registry import get
from virtualmodelcontrol.robots import adapt, bimanual, helyx

HAND = np.load(Path(__file__).parents[1] / "data" / "hand.npz")
rng = np.random.default_rng(19)


def meas(theta, rate=None):
    rate = np.zeros_like(theta) if rate is None else rate
    return vmc.Signals(0.0, motor_position=theta, motor_velocity=rate)


def test_hand_and_finger_friction_match_the_recorded_controllers():
    hand, finger = adapt.hand_output_stage()[0], adapt.output_stage()[0]
    for v, tau, f_hand, f_finger in zip(
        HAND["friction_v"],
        HAND["friction_tau"],
        HAND["hand_friction"],
        HAND["finger_friction"],
        strict=True,
    ):
        np.testing.assert_allclose(hand(tau, meas(np.zeros(13), v)) - tau, f_hand, atol=1e-15)
        np.testing.assert_allclose(
            finger(tau[:2], meas(np.zeros(2), v[:2])) - tau[:2], f_finger, atol=1e-15
        )


def test_bimanual_output_reproduces_its_controller_formula():
    # original: clip(tau + 0.03 - where(q < -30°, 0.03 q, 0), -0.5, 0.5), sign +1 (q > 0 pulls)
    stages = bimanual.output_stage()
    for _ in range(20):
        tau, q = rng.uniform(-0.6, 0.6, 18), rng.uniform(-1.5, 1.5, 18)
        expected = np.clip(tau + 0.03 - np.where(q < -np.radians(30), 0.03 * q, 0.0), -0.5, 0.5)
        np.testing.assert_allclose(output.apply(stages, tau, meas(q)), expected, atol=1e-15)


def test_single_arm_pretension_reproduces_its_controller_formula():
    # original, in raw motor angles (q > 0 lengthens): tau_cmd = tau - 0.010 q_raw
    sign = helyx.ENCODER_SIGN["145-290-290"]
    stages = helyx.output_stage()
    for _ in range(20):
        tau_raw, q_raw = rng.uniform(-0.2, 0.2, 9), rng.uniform(-3, 3, 9)
        u = output.apply(stages, sign * tau_raw, meas(sign * q_raw))
        np.testing.assert_allclose(sign * u, tau_raw - 0.010 * q_raw, atol=1e-15)


def test_controller_applies_stages_after_the_law():
    robot, ctrl = adapt.finger(), vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)), 1.0))
    c = vmc.VMCController(
        vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)),
        output=[output.TorqueLimit(0.1), output.TorqueOffset(0.05)],
    )
    out = c.step(0.0, meas(np.array([1.0, -0.05])))
    np.testing.assert_allclose(out["law_torque"], [-1.0, 0.05])
    np.testing.assert_allclose(out["motor_torque"], [-0.05, 0.1])  # clipped, then offset


def exact_floor(u, floor):
    """The floor without the rounded corner: each group lifted until its lowest is on it."""
    out = u.copy()
    for k in range(0, u.size, 3):
        out[k : k + 3] += max(0.0, floor - u[k : k + 3].min())
    return out


def test_tension_floor_keeps_every_tendon_up_without_changing_the_bend():
    stage = output.TensionFloor(0.03)
    for _ in range(50):
        u = rng.uniform(-0.5, 0.5, 9)
        out = stage(u, meas(np.zeros(9)))
        assert (out >= 0.03 - 1e-12).all()
        # the lift is equal on the three tendons of a segment: their differences are the command's
        np.testing.assert_allclose(
            np.diff(out.reshape(3, 3), axis=1), np.diff(u.reshape(3, 3), axis=1), atol=1e-12
        )
        assert np.abs(out - exact_floor(u, 0.03)).max() <= (np.log(3) + np.log(2)) * stage.width


def test_tension_floor_leaves_tendons_far_above_it_and_survives_big_commands():
    stage = output.TensionFloor(0.03)
    far = np.full(9, 0.4)
    np.testing.assert_allclose(stage(far, meas(np.zeros(9))), far, atol=1e-12)
    big = stage(np.array([-80.0, 0.3, 0.2] * 3), meas(np.zeros(9)))
    assert np.isfinite(big).all() and big.min() >= 0.03 - 1e-12


def test_tension_floor_symbolic_is_the_same_map():
    stage = output.TensionFloor(0.03, width=0.002)
    u = ca.SX.sym("u", 9)
    f = ca.Function("f", [u], [stage.symbolic(u)])
    for _ in range(20):
        x = rng.uniform(-0.5, 0.5, 9)
        np.testing.assert_allclose(np.array(f(x)).ravel(), stage(x, meas(np.zeros(9))), atol=1e-14)
    slope = ca.Function("slope", [u], [ca.jacobian(stage.symbolic(u), u)])
    assert np.isfinite(np.array(slope(rng.uniform(-0.5, 0.5, 9)))).all()


def test_tension_floor_refuses_a_width_that_gives_nan_and_a_group_that_is_not_a_group():
    for width in (0.0, -0.001):
        with pytest.raises(ValueError, match=r"width must be positive.*1e-4"):
            output.TensionFloor(0.03, width=width)
    for group in (0, -3):
        with pytest.raises(ValueError, match="group must be at least 1"):
            output.TensionFloor(0.03, group=group)
    nearly_exact = output.TensionFloor(0.03, width=1e-4)(
        np.array([0.1, -0.2, 0.0]), meas(np.zeros(3))
    )
    assert np.isfinite(nearly_exact).all() and nearly_exact.min() == pytest.approx(0.03, abs=1e-3)


def test_tension_floor_groups_and_registry():
    with pytest.raises(ValueError, match="groups of 3"):
        output.TensionFloor(0.03)(np.zeros(8), meas(np.zeros(8)))
    pair = output.TensionFloor(0.0, group=2)
    np.testing.assert_allclose(
        pair(np.array([-0.1, 0.3]), meas(np.zeros(2))), [0.0, 0.4], atol=2e-3
    )
    assert get("output", "tension_floor") is output.TensionFloor
    assert helyx.tension_stage()[0].floor == helyx.TENSION_FLOOR
    assert helyx.tension_stage(0.0)[0].floor == 0.0
