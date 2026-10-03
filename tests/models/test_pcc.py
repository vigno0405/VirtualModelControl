"""PCC kinematics: recorded lab values, arc geometry, smoothness and the model contract."""

from pathlib import Path

import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import PCC, evaluate_frame, from_dict
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "pcc.npz")
GEOMETRIES = list(helyx.GEOMETRIES)
rng = np.random.default_rng(4)
# D = √(Dx² + Dy² + ε) shortens the bending axis slightly, so rotations are orthonormal and arcs
# are exact only up to about ε / d² (ε = 1e-12, d = 30 mm), per segment.
TOL = 3 * 1e-12 / helyx.SECTION_RADIUS**2


def motor_function(geometry):
    """(raw motor angles, s) -> (position, R, ∂position/∂q_raw) for one arm."""
    robot = helyx.arm(geometry)
    p = {**constants(robot.model.params), **constants(robot.actuation.params)}
    q_raw, s = ca.SX.sym("q", 9), ca.SX.sym("s")
    delta = robot.actuation.config_from_motors(helyx.ENCODER_SIGN[geometry] * q_raw, p)
    R, pos = robot.model.frame(delta, s, p)
    return ca.Function("f", [q_raw, s], [pos, R, ca.jacobian(pos, q_raw)])


def config_function(model):
    """(Δ, s) -> (position, R, ∂position/∂Δ, ∂position/∂s)."""
    q, s = ca.SX.sym("q", model.space.nq), ca.SX.sym("s")
    R, pos = model.frame(q, s, constants(model.params))
    return ca.Function("f", [q, s], [pos, R, ca.jacobian(pos, q), ca.jacobian(pos, s)])


def euler_to_matrix(yaw_pitch_roll):
    return Rotation.from_euler("ZYX", yaw_pitch_roll).as_matrix()


def random_config(n=9, scale=0.02):
    return rng.uniform(-scale, scale, n)


def test_matches_recorded_145_290_290_with_jacobians():
    f, key = motor_function("145-290-290"), "145-290-290"
    for q, s, pose, J in zip(
        DATA[f"{key}/q"], DATA[f"{key}/s"], DATA[f"{key}/pose"], DATA[f"{key}/jac"], strict=True
    ):
        pos, R, jac = (np.array(x) for x in f(q, s))
        np.testing.assert_allclose(pos.ravel(), pose[:3], atol=1e-14)
        np.testing.assert_allclose(R, euler_to_matrix(pose[3:]), atol=1e-9)
        np.testing.assert_allclose(jac, J, atol=1e-13)


def test_matches_recorded_145_145_145():
    f, key = motor_function("145-145-145"), "145-145-145"
    for q, s, pose in zip(DATA[f"{key}/q"], DATA[f"{key}/s"], DATA[f"{key}/pose"], strict=True):
        pos, R, _ = (np.array(x) for x in f(q, s))
        np.testing.assert_allclose(pos.ravel(), pose[:3], atol=1e-14)
        np.testing.assert_allclose(R, euler_to_matrix(pose[3:]), atol=1e-9)


def test_matches_recorded_290_145_145():
    f, key = motor_function("290-145-145"), "290-145-145"
    for q, s, position in zip(
        DATA[f"{key}/q"], DATA[f"{key}/s"], DATA[f"{key}/position"], strict=True
    ):
        np.testing.assert_allclose(np.array(f(q, s)[0]).ravel(), position, atol=1e-14)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_points_lie_on_true_arcs_at_constant_speed(geometry):
    model = helyx.arm(geometry).model
    f, b = config_function(model), model.breakpoints()
    d = helyx.SECTION_RADIUS
    for _ in range(5):
        q = random_config()
        for i in range(3):
            dx, dy, dl = q[3 * i : 3 * i + 3]
            L0 = helyx.GEOMETRIES[geometry]["L0"][i]
            D = np.sqrt(dx**2 + dy**2 + 1e-12)
            start = np.array(f(q, b[i])[0]).ravel()
            for sl in (0.25, 0.5, 0.9):
                pos, R, _, dpds = (np.array(x) for x in f(q, b[i] + sl * (b[i + 1] - b[i])))
                # chord of an arc of radius ρ = d (L0 + Dl) / D turned by θ = sl D / d
                chord = 2 * d * (L0 + dl) / D * np.sin(sl * D / d / 2)
                assert abs(np.linalg.norm(pos.ravel() - start) - chord) < TOL * chord
                speed = np.linalg.norm(dpds)
                assert abs(speed / ((L0 + dl) / (b[i + 1] - b[i])) - 1) < TOL
                np.testing.assert_allclose(dpds.ravel() / speed, R[:, 2], atol=TOL)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_frames_are_continuous_and_rotations_orthonormal(geometry):
    model = helyx.arm(geometry).model
    f, b = config_function(model), model.breakpoints()
    for _ in range(5):
        q = random_config()
        for s in b[1:-1]:
            p0, R0 = (np.array(x) for x in f(q, s)[:2])
            p1, R1 = (np.array(x) for x in f(q, s + 1e-9)[:2])
            assert np.abs(p1 - p0).max() < 1e-8 and np.abs(R1 - R0).max() < 1e-7
        for s in rng.uniform(0, 1, 5):
            R = np.array(f(q, s)[1])
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=TOL)
            assert abs(np.linalg.det(R) - 1.0) < TOL


def test_jacobian_matches_finite_differences():
    f = config_function(helyx.arm().model)
    h = 1e-7
    for _ in range(5):
        q, s = random_config(), rng.uniform(0, 1)
        J = np.array(f(q, s)[2])
        fd = np.column_stack(
            [
                (np.array(f(q + h * e, s)[0]) - np.array(f(q - h * e, s)[0])).ravel() / (2 * h)
                for e in np.eye(9)
            ]
        )
        np.testing.assert_allclose(J, fd, atol=1e-7)


def test_no_nan_at_straight_pose_bounds_and_large_bends():
    model = helyx.arm().model
    f, b = config_function(model), model.breakpoints()
    for q in (np.zeros(9), np.array([0.05, 0.0, 0.0] * 3), np.array([0.0, -0.06, 0.01] * 3)):
        for s in (0.0, *b[1:-1], 0.5, 1.0, -0.2, 1.3):
            assert all(np.all(np.isfinite(np.array(x))) for x in f(q, s))


def test_straight_arm_and_sites():
    model = helyx.arm("145-290-290").model
    R, p = evaluate_frame(model, np.zeros(9), "tip")
    np.testing.assert_allclose(p, [0, 0, 0.725], atol=1e-15)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)
    q, b = random_config(), model.breakpoints()
    for site, s in (("seg1", b[1]), ("seg2", b[2]), ("tip", 1.0), ("base", 0.0)):
        for a, c in zip(evaluate_frame(model, q, site), evaluate_frame(model, q, s), strict=True):
            np.testing.assert_allclose(a, c, atol=1e-15)
    with pytest.raises(KeyError, match="unknown site"):
        evaluate_frame(model, q, "elbow")


def test_symbolic_and_numeric_arc_parameter_agree():
    model = helyx.arm().model
    f = config_function(model)
    q = random_config()
    for s in rng.uniform(0, 1, 10):
        np.testing.assert_allclose(
            np.array(f(q, s)[0]).ravel(), evaluate_frame(model, q, float(s))[1], atol=1e-15
        )


def test_dict_round_trip():
    model = PCC([0.1, 0.2], [0.02, 0.025])
    copy = from_dict(model.to_dict())
    assert isinstance(copy, PCC) and copy.to_dict() == model.to_dict()
    q = random_config(6)
    np.testing.assert_allclose(evaluate_frame(copy, q, 0.7)[1], evaluate_frame(model, q, 0.7)[1])
