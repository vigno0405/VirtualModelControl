"""Assemblies and couplings: mounted parts, stacked actuation, controllers across parts."""

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import (
    Assembly,
    LinearCoupling,
    SerialChain,
    evaluate_frame,
    from_dict,
)
from virtualmodelcontrol.robots import adapt, helyx

rng = np.random.default_rng(13)


def two_arms():
    left, right = helyx.arm("290-145-145").model, helyx.arm("290-145-145").model
    rot = [np.pi, 0.0, 0.0]
    return Assembly({"left": (left, [-0.125, 0, 0], rot), "right": (right, [0.125, 0, 0], rot)})


def test_parts_sit_at_their_mounts():
    body = two_arms()
    q = rng.uniform(-0.01, 0.01, 18)
    R_m = np.diag([1.0, -1.0, -1.0])  # half turn about x
    for name, sl, x in (("left", slice(0, 9), -0.125), ("right", slice(9, 18), 0.125)):
        R, p = evaluate_frame(helyx.arm("290-145-145").model, q[sl], 0.7)
        R_a, p_a = evaluate_frame(body, q, (name, 0.7))
        np.testing.assert_allclose(p_a, [x, 0, 0] + R_m @ p, atol=1e-14)
        np.testing.assert_allclose(R_a, R_m @ R, atol=1e-12)
        np.testing.assert_allclose(
            evaluate_frame(body, q, f"{name}/tip")[1],
            evaluate_frame(body, q, (name, 1.0))[1],
            atol=1e-14,
        )
    assert "left.seg1.L0" in body.params and "right.mount.position" in body.params


def test_controller_across_two_tendon_arms():
    body = two_arms()
    tendons = helyx.arm("290-145-145").actuation
    robot = vmc.Mechanism(
        "arms",
        model=body,
        actuation=body.stacked_actuation(
            {"left": tendons, "right": helyx.arm("290-145-145").actuation}
        ),
    )
    ctrl = vmc.Mechanism("ctrl")
    # a spring between the two tips: pulls them together (a grasp)
    ctrl.add(
        "squeeze", vmc.LinearSpring(robot.point("left", s=1.0) - robot.point("right", s=1.0), 20.0)
    )
    law = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    theta = rng.uniform(-1, 1, 18)
    u = np.array(law.fast(np.concatenate([theta, np.zeros(18), law.live_values(), [0.0]]))).ravel()
    assert u.shape == (18,) and np.all(np.isfinite(u)) and np.abs(u).max() > 0
    # the same torques, part by part, from each arm's own transmission
    p = constants(robot.actuation.params)
    q = np.array(robot.actuation.config_from_motors(ca.DM(theta), p)).ravel()
    tau = np.array(law.tau(q, np.zeros(18), [], law.live_values(), 0.0)).ravel()
    pa = constants(tendons.params)
    for sl in (slice(0, 9), slice(9, 18)):
        np.testing.assert_allclose(
            u[sl], np.array(tendons.allocate(ca.DM(tau[sl]), None, pa)).ravel(), atol=1e-12
        )


def test_coupled_finger_and_dict_round_trips():
    finger = adapt.finger().model
    assert isinstance(finger, LinearCoupling) and finger.space.nq == 2
    copy = from_dict(finger.to_dict())
    q = rng.uniform(-2, 2, 2)
    np.testing.assert_allclose(
        evaluate_frame(copy, q, "tip")[1], evaluate_frame(finger, q, "tip")[1]
    )
    body = two_arms()
    q = rng.uniform(-0.01, 0.01, 18)
    np.testing.assert_allclose(
        evaluate_frame(from_dict(body.to_dict()), q, "right/tip")[1],
        evaluate_frame(body, q, "right/tip")[1],
        atol=1e-15,
    )


def test_shared_part_model_keeps_one_set_of_params():
    chain = SerialChain(["revolute"], [[0, 0, 1]], [[0, 0, 0]], {"tip": (1, [0.1, 0, 0])})
    body = Assembly({"a": (chain, [0, 0, 0], [0, 0, 0]), "b": (chain, [0, 0.2, 0], [0, 0, 0])})
    np.testing.assert_allclose(
        evaluate_frame(body, [0.0, np.pi / 2], "b/tip")[1], [0, 0.3, 0], atol=1e-15
    )
