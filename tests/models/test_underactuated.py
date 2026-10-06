import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import Underactuated, from_dict

B_TENDON = np.array([[-2.0, -2.0], [-2.0, 0.0], [-2.0, 0.0]])  # not orthonormal


def numbers(actuation):
    return constants(actuation.params)


def num(x):
    return np.array(ca.evalf(x))


def test_joints_builds_the_selection_matrix():
    act = Underactuated.joints(3, [0, 2])
    np.testing.assert_array_equal(act.params["B"].value, [[1, 0], [0, 0], [0, 1]])
    assert act.motor_sizes(None) == (2, 2)
    q = ca.DM([0.1, 0.2, 0.3])
    np.testing.assert_allclose(num(act.motor_angles(q, numbers(act))).ravel(), [0.1, 0.3])
    np.testing.assert_allclose(num(act.motor_rates(q, q, numbers(act))).ravel(), [0.1, 0.3])
    np.testing.assert_allclose(num(act.generalized_force(ca.DM([1, 2]), q, numbers(act))).ravel(),
                               [1, 0, 2])  # fmt: skip


def test_the_projector_splits_a_torque_into_what_the_motors_give_and_the_rest():
    act = Underactuated(B_TENDON)
    p, tau = numbers(act), ca.DM([1.0, -2.0, 0.5])
    E = num(act.projector(p))
    np.testing.assert_allclose(E, E.T, atol=1e-12)
    np.testing.assert_allclose(E @ E, E, atol=1e-12)
    np.testing.assert_allclose(E @ B_TENDON, 0, atol=1e-12)
    u = num(act.allocate(tau, None, p)).ravel()
    np.testing.assert_allclose(u, np.linalg.pinv(B_TENDON) @ num(tau).ravel(), atol=1e-12)
    assert not np.allclose(np.linalg.pinv(B_TENDON), B_TENDON.T)  # B+ is not B transposed
    delivered = num(act.generalized_force(ca.DM(u), None, p)).ravel()
    defect = num(act.defect(tau, p)).ravel()
    np.testing.assert_allclose(delivered + defect, num(tau).ravel(), atol=1e-12)
    np.testing.assert_allclose(defect, E @ num(tau).ravel(), atol=1e-12)


def test_the_motors_give_the_frozen_configuration():
    rest = [0.0, 0.7, -0.4]
    act = Underactuated(B_TENDON, rest)
    p = numbers(act)
    q, v = np.array([0.3, -0.2, 0.5]), np.array([0.1, 0.4, -0.3])
    E, B = num(act.projector(p)), B_TENDON
    theta = num(act.motor_angles(ca.DM(q), p))
    frozen = num(act.config_from_motors(ca.DM(theta), p)).ravel()
    np.testing.assert_allclose(frozen, (np.eye(3) - E) @ q + E @ rest, atol=1e-12)
    rate = num(act.motor_rates(ca.DM(q), ca.DM(v), p))
    np.testing.assert_allclose(
        num(act.velocity_from_motors(ca.DM(frozen), ca.DM(rate), p)).ravel(),
        (np.eye(3) - E) @ v, atol=1e-12)  # fmt: skip
    # the same motors, whatever the unmeasured part of q is
    other = q + E @ [1.0, 2.0, 3.0]
    np.testing.assert_allclose(num(act.motor_angles(ca.DM(other), p)), theta, atol=1e-12)
    assert np.linalg.matrix_rank(B) == 2


def test_efficiency_acts_on_the_delivered_torque_only():
    act = Underactuated.joints(3, [0, 2], efficiency=0.5)
    p, u = numbers(act), ca.DM([2.0, 4.0])
    np.testing.assert_allclose(num(act.generalized_force(u, None, p)).ravel(), [1, 0, 2])
    tau = ca.DM([1.0, 5.0, 2.0])
    np.testing.assert_allclose(num(act.allocate(tau, None, p)).ravel(), [1, 2])  # no 1/eta


def test_serialization_and_shape():
    act = Underactuated(B_TENDON, [0.0, 0.1, 0.2], 0.9)
    clone = from_dict(act.to_dict(), "actuation")
    assert isinstance(clone, Underactuated)
    np.testing.assert_array_equal(clone.params.vector(), act.params.vector())
    for bad in (np.eye(3), np.ones(3), np.ones((2, 3))):
        with pytest.raises(ValueError, match="m < n"):
            Underactuated(bad)
