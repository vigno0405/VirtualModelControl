import numpy as np
import pytest

from virtualmodelcontrol.models.kinematics import contact_map, contact_projection

rng = np.random.default_rng(11)
J = rng.normal(size=(3, 9))


def test_without_a_normal_the_map_is_the_pseudo_inverse_of_the_transpose():
    A = contact_map(J)
    np.testing.assert_allclose(A, np.linalg.pinv(J).T, atol=1e-12)
    np.testing.assert_allclose(A @ J.T, np.eye(3), atol=1e-12)


def test_with_a_normal_it_is_the_pseudo_inverse_of_the_projected_jacobian():
    n = np.array([0.2, -0.4, 0.9])
    unit = n / np.linalg.norm(n)
    A = contact_map(J, 7.0 * n)  # any length
    np.testing.assert_allclose(A, np.linalg.pinv(np.outer(unit, unit) @ J).T, atol=1e-12)
    np.testing.assert_allclose(A @ J.T @ (3.0 * unit), 3.0 * unit, atol=1e-12)
    assert np.linalg.matrix_rank(A) == 1  # the force is along the normal


@pytest.mark.parametrize("normal", [None, [0.0, 0.3, 0.95]])
def test_a_jacobian_of_rank_below_3_has_a_finite_map(normal):
    """Two motors move a point in a plane: the force along the missing direction is not seen."""
    J2 = rng.normal(size=(3, 2))
    A = contact_map(J2, normal)
    assert np.all(np.isfinite(A)) and A.shape == (3, 2)
    f = contact_projection(normal) @ J2 @ np.array([1.0, 0.5])  # a force the motors can make
    np.testing.assert_allclose(A @ (J2.T @ f), contact_projection(normal) @ f, atol=1e-10)


def test_the_projection_keeps_the_part_along_the_normal():
    n = np.array([0.0, 3.0, 4.0])
    np.testing.assert_allclose(contact_projection(n) @ np.array([1.0, 1.0, 1.0]), [0.0, 0.84, 1.12])
    np.testing.assert_array_equal(contact_projection(None), np.eye(3))
