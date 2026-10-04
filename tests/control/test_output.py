"""Output stages against the original controllers' output formulas and friction compensation."""

from pathlib import Path

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import output
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
