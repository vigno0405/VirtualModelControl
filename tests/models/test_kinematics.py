"""Jacobians and Hessians by automatic differentiation, against the original hand-derived ones."""

from pathlib import Path

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt, helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "derivatives.npz")


def test_hand_tip_jacobians_and_hessians():
    kin = vmc.Kinematics(adapt.hand())
    for i, q in enumerate(DATA["hand/q"]):
        for digit in adapt.HAND_DIGITS:
            np.testing.assert_allclose(
                kin.jacobian(q, f"{digit}/tip"), DATA[f"hand/{digit}/J"][i], atol=1e-14
            )
            np.testing.assert_allclose(
                kin.hessian(q, f"{digit}/tip"), DATA[f"hand/{digit}/H"][i], atol=1e-14
            )


def test_finger_tip_jacobian_and_hessian():
    kin = vmc.Kinematics(adapt.finger())
    for i, q in enumerate(DATA["finger/q"]):
        np.testing.assert_allclose(kin.jacobian(q, "tip"), DATA["finger/J"][i], atol=1e-15)
        np.testing.assert_allclose(kin.hessian(q, "tip"), DATA["finger/H"][i], atol=1e-15)


def test_soft_arm_jacobian_and_hessian_in_motor_angles():
    kin = vmc.Kinematics(helyx.arm("290-145-145"), coordinates="motors")
    for i, theta in enumerate(DATA["bimanual/q"]):  # the arm's encoder sign is +1
        for s in (0.6, 1.0):
            np.testing.assert_allclose(
                kin.jacobian(theta, s), DATA[f"bimanual/{s}/J"][i], atol=1e-14
            )
            np.testing.assert_allclose(
                kin.hessian(theta, s), DATA[f"bimanual/{s}/H"][i], atol=1e-13
            )


def test_angular_jacobian_and_offsets():
    arm = helyx.arm()
    kin = vmc.Kinematics(arm)
    rng = np.random.default_rng(17)
    q, v, h = rng.uniform(-0.01, 0.01, 9), rng.normal(size=9), 1e-7
    R0, R1 = kin.rotation(q - h * v, 0.8), kin.rotation(q + h * v, 0.8)
    S = (R1 - R0) / (2 * h) @ kin.rotation(q, 0.8).T  # skew-symmetric: [ω]×
    np.testing.assert_allclose(
        kin.angular_jacobian(q, 0.8) @ v, [S[2, 1], S[0, 2], S[1, 0]], atol=1e-6
    )
    off = [0.01, 0.0, 0.02]
    np.testing.assert_allclose(
        kin.position(q, 0.8, off), kin.position(q, 0.8) + kin.rotation(q, 0.8) @ off, atol=1e-15
    )
