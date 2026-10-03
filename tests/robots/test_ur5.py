"""UR5 from its DH table, and the hand on its flange."""

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import adapt, ur5

rng = np.random.default_rng(18)


def dh_forward(q):
    """Direct product of standard DH transforms (independent of the library)."""
    T = np.eye(4)
    for qi, d, a, al in zip(q, ur5.DH_D, ur5.DH_A, ur5.DH_ALPHA, strict=True):
        ct, st, ca_, sa = np.cos(qi), np.sin(qi), np.cos(al), np.sin(al)
        T = T @ np.array(
            [
                [ct, -st * ca_, st * sa, a * ct],
                [st, ct * ca_, -ct * sa, a * st],
                [0, sa, ca_, d],
                [0, 0, 0, 1],
            ]
        )
    return T


def test_tool_frame_matches_the_dh_product():
    model = ur5.model()
    for q in rng.uniform(-np.pi, np.pi, (20, 6)):
        R, p = evaluate_frame(model, q, "tool")
        T = dh_forward(q)
        np.testing.assert_allclose(p, T[:3, 3], atol=1e-14)
        np.testing.assert_allclose(R, T[:3, :3], atol=1e-14)


def test_hand_rides_on_the_flange():
    robot = ur5.with_hand()
    hand = adapt.hand_model()
    for _ in range(5):
        qa, qh = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-1, 1, 13)
        T = dh_forward(qa)
        R_h, p_h = evaluate_frame(hand, qh, "index/tip")
        R, p = evaluate_frame(robot.model, np.concatenate([qa, qh]), "hand/index/tip")
        np.testing.assert_allclose(p, T[:3, 3] + T[:3, :3] @ p_h, atol=1e-14)
        np.testing.assert_allclose(R, T[:3, :3] @ R_h, atol=1e-13)


def test_gravity_on_the_arm_equals_the_hand_alone_with_rotated_gravity():
    with_arm, ctrl = ur5.with_hand(), vmc.Mechanism("ctrl")
    ctrl.add("gravity", vmc.GravityCompensation(with_arm))
    a = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(with_arm, ctrl)))
    hand, ctrl2 = adapt.hand(), vmc.Mechanism("ctrl")
    ctrl2.add("gravity", vmc.GravityCompensation(hand))
    b = vmc.VMCController(
        vmc.compile(vmc.VirtualMechanismSystem(hand, ctrl2), runtime=["*.gravity"])
    )
    for _ in range(5):
        qa, qh = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-1, 1, 13)
        meas = vmc.Signals(
            0.0, motor_position=np.concatenate([qa, qh]), motor_velocity=np.zeros(19)
        )
        u_arm = a.step(0.0, meas)["motor_torque"][6:]
        b.set({"hand.gravity": adapt.hand_gravity(dh_forward(qa)[:3, :3])})
        u_hand = b.step(0.0, vmc.Signals(0.0, motor_position=qh, motor_velocity=np.zeros(13)))[
            "motor_torque"
        ]
        np.testing.assert_allclose(u_arm, u_hand, atol=1e-14)
