import casadi as ca
import numpy as np
import pytest

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
