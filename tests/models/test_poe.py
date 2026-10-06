import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import SerialChain, evaluate_frame, from_dict

rng = np.random.default_rng(6)
L1, L2 = 0.3, 0.2


def two_link():
    z = [0.0, 0.0, 1.0]
    return SerialChain(
        ["revolute", "revolute"],
        axes=[z, z],
        points=[[0.0, 0.0, 0.0], [L1, 0.0, 0.0]],
        sites={"elbow": (1, [L1, 0.0, 0.0]), "tip": (2, [L1 + L2, 0.0, 0.0])},
    )


def test_two_link_arm_matches_the_textbook():
    arm = two_link()
    for q in rng.uniform(-np.pi, np.pi, (10, 2)):
        R, p = evaluate_frame(arm, q, "tip")
        c1, s1, c12, s12 = np.cos(q[0]), np.sin(q[0]), np.cos(q.sum()), np.sin(q.sum())
        np.testing.assert_allclose(p, [L1 * c1 + L2 * c12, L1 * s1 + L2 * s12, 0.0], atol=1e-15)
        np.testing.assert_allclose(R[:2, :2], [[c12, -s12], [s12, c12]], atol=1e-15)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)


def test_jacobian_matches_finite_differences_and_is_finite_when_straight():
    arm = two_link()
    q = ca.SX.sym("q", 2)
    _, p = arm.frame(q, "tip", constants(arm.params))
    J = ca.Function("J", [q], [ca.jacobian(p, q)])
    h = 1e-7
    for q0 in [np.zeros(2), *rng.uniform(-3, 3, (5, 2))]:
        fd = np.column_stack(
            [
                (
                    evaluate_frame(arm, q0 + h * e, "tip")[1]
                    - evaluate_frame(arm, q0 - h * e, "tip")[1]
                )
                / (2 * h)
                for e in np.eye(2)
            ]
        )
        Jq = np.array(J(q0))
        assert np.all(np.isfinite(Jq))
        np.testing.assert_allclose(Jq, fd, atol=1e-8)


def test_prismatic_joint_translates_along_its_axis():
    slider = SerialChain(
        ["prismatic"],
        axes=[[0.0, 2.0, 0.0]],
        points=[[0, 0, 0]],
        sites={"cart": (1, [1.0, 0.0, 0.0])},
    )
    R, p = evaluate_frame(slider, [0.5], "cart")
    np.testing.assert_allclose(p, [1.0, 0.5, 0.0])  # axis normalized
    np.testing.assert_allclose(R, np.eye(3))
    assert slider.q_unit == "m" and two_link().q_unit == "rad"


def test_dict_round_trip_and_errors():
    arm = two_link()
    copy = from_dict(arm.to_dict())
    q = rng.normal(size=2)
    np.testing.assert_allclose(evaluate_frame(copy, q, "tip")[1], evaluate_frame(arm, q, "tip")[1])
    with pytest.raises(KeyError, match="unknown site"):
        evaluate_frame(arm, q, "wrist")
    with pytest.raises(ValueError, match="joint types"):
        SerialChain(["ball"], [[0, 0, 1]], [[0, 0, 0]], {})


def test_helical_joint_turns_and_slides_by_its_pitch():
    screw = SerialChain(
        [("helical", 0.004)],
        axes=[[0.0, 0.0, 2.0]],
        points=[[0, 0, 0]],
        sites={"nut": (1, [1.0, 0.0, 0.0])},
    )
    for q in (-2.0, 0.3, 5.0):
        R, p = evaluate_frame(screw, [q], "nut")
        np.testing.assert_allclose(p, [np.cos(q), np.sin(q), 0.004 * q], atol=1e-15)
        np.testing.assert_allclose(R, Rotation.from_euler("z", q).as_matrix(), atol=1e-15)
    assert screw.params["j1.pitch"].unit == "m/rad"


def test_spherical_joint_turns_about_its_point_by_a_rotation_vector():
    centre, site = np.array([0.1, -0.2, 0.3]), np.array([0.5, 0.4, 0.3])
    ball = SerialChain(["spherical"], axes=[None], points=[centre], sites={"tip": (1, site)})
    assert ball.space.nq == 3
    for w in rng.normal(size=(5, 3)):
        R, p = evaluate_frame(ball, w, "tip")
        expected = Rotation.from_rotvec(w).as_matrix()
        np.testing.assert_allclose(R, expected, atol=1e-14)
        np.testing.assert_allclose(p, centre + expected @ (site - centre), atol=1e-14)


def test_free_joint_moves_a_body_by_a_translation_and_a_rotation_vector():
    centre, site = np.array([0.1, 0.0, 0.2]), np.array([0.3, 0.4, 0.5])
    body = SerialChain(["free"], axes=[None], points=[centre], sites={"corner": (1, site)})
    assert body.space.nq == 6
    for q in rng.normal(size=(5, 6)):
        R, p = evaluate_frame(body, q, "corner")
        expected = Rotation.from_rotvec(q[3:]).as_matrix()
        np.testing.assert_allclose(R, expected, atol=1e-14)
        np.testing.assert_allclose(p, q[:3] + centre + expected @ (site - centre), atol=1e-14)
    R, p = evaluate_frame(body, np.zeros(6), "corner")  # the neutral configuration is q = 0
    np.testing.assert_allclose(p, site, atol=1e-15)


def floating_arm():
    """A free base carrying a two-link arm: the arm's joint axes are given at q = 0."""
    z = [0.0, 0.0, 1.0]
    return SerialChain(
        ["free", "revolute", "revolute"],
        axes=[None, z, z],
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [L1, 0.0, 0.0]],
        sites={"base": (1, [0.0, 0.0, 0.0]), "tip": (3, [L1 + L2, 0.0, 0.0])},
    )


def test_a_chain_on_a_floating_base_moves_with_the_base():
    arm = floating_arm()
    assert arm.space.nq == 8
    base, joints = rng.normal(size=6), rng.normal(size=2)
    R, p = evaluate_frame(arm, np.concatenate([base, joints]), "tip")
    R0, p0 = evaluate_frame(arm, np.concatenate([np.zeros(6), joints]), "tip")
    Rb = Rotation.from_rotvec(base[3:]).as_matrix()
    np.testing.assert_allclose(R, Rb @ R0, atol=1e-14)
    np.testing.assert_allclose(p, base[:3] + Rb @ p0, atol=1e-14)


def test_every_joint_type_passes_the_model_contract():
    from virtualmodelcontrol.testing import check_model

    check_model(floating_arm(), samples=3, energy=False)
    screw = SerialChain(
        [("helical", 0.004), "spherical"],
        axes=[[0.0, 1.0, 0.0], None],
        points=[[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]],
        sites={"tip": (2, [0.0, 0.0, 0.5])},
    )
    check_model(screw, samples=3, energy=False)


def test_the_new_joint_types_survive_a_dict_round_trip_in_json():
    import json

    screw = SerialChain(
        [("helical", 0.004), "free", "prismatic"],
        axes=[[0.0, 1.0, 0.0], None, [1.0, 0.0, 0.0]],
        points=[[0.1, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]],
        sites={"tip": (3, [0.0, 0.0, 0.5])},
    )
    copy = from_dict(json.loads(json.dumps(screw.to_dict())))
    q = rng.normal(size=screw.space.nq)
    for a, b in zip(evaluate_frame(copy, q, "tip"), evaluate_frame(screw, q, "tip"), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-15)
    assert copy.joints == screw.joints and copy.q_unit == "rad"
