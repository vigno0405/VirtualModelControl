import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.models.kinematics import contact_map

rng = np.random.default_rng(11)
J = rng.normal(size=(3, 9))


def test_without_a_normal_the_map_is_the_pseudo_inverse_of_the_transpose():
    A = np.array(contact_map(J))
    np.testing.assert_allclose(A, np.linalg.pinv(J).T, atol=1e-12)
    np.testing.assert_allclose(A @ J.T, np.eye(3), atol=1e-12)


def test_with_a_normal_it_is_the_pseudo_inverse_of_the_projected_jacobian():
    n = np.array([0.2, -0.4, 0.9])
    unit = n / np.linalg.norm(n)
    A = np.array(contact_map(J, 7.0 * n))  # any length
    np.testing.assert_allclose(A, np.linalg.pinv(np.outer(unit, unit) @ J).T, atol=1e-12)
    np.testing.assert_allclose(A @ J.T @ (3.0 * unit), 3.0 * unit, atol=1e-12)
    assert np.linalg.matrix_rank(A) == 1  # the force is along the normal


@pytest.mark.parametrize("normal", [None, [0.0, 0.3, 0.95]])
def test_it_works_on_symbols_too(normal):
    x = ca.SX.sym("x", 3, 9)
    A = ca.Function("A", [x], [contact_map(x, normal)])
    np.testing.assert_allclose(np.array(A(J)), np.array(contact_map(J, normal)), atol=1e-13)
