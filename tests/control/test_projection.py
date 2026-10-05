"""The projection of a stiffness onto the positive semidefinite matrices."""

import numpy as np
import pytest

from virtualmodelcontrol.control import project_psd

rng = np.random.default_rng(7)


def test_a_symmetric_positive_semidefinite_matrix_is_unchanged():
    A = rng.normal(size=(4, 4))
    K = A @ A.T
    np.testing.assert_allclose(project_psd(K), K, atol=1e-12)


def test_a_non_symmetric_matrix_loses_its_antisymmetric_part():
    K = np.array([[2.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(project_psd(K), [[2.0, 0.5], [0.5, 2.0]], atol=1e-12)


def test_a_negative_eigenvalue_is_raised_to_zero():
    # eigenvalues 3 along (1, 1) and -1 along (1, -1)
    np.testing.assert_allclose(project_psd([[1.0, 2.0], [2.0, 1.0]]), [[1.5, 1.5], [1.5, 1.5]])
    assert project_psd([[-3.0]]).tolist() == [[0.0]]
    np.testing.assert_allclose(project_psd(np.diag([2.0, -1.0, 0.5])), np.diag([2.0, 0.0, 0.5]))


def test_the_result_is_positive_semidefinite_and_nearest_in_the_frobenius_norm():
    for _ in range(20):
        K = rng.normal(size=(3, 3))
        P = project_psd(K)
        assert np.linalg.eigvalsh(P).min() > -1e-12
        np.testing.assert_allclose(P, P.T, atol=1e-15)
        np.testing.assert_allclose(
            project_psd(P), P, atol=1e-12
        )  # a second projection changes nothing
        for _ in range(10):
            B = rng.normal(size=(3, 3))
            other = B @ B.T  # some other positive semidefinite matrix
            assert np.linalg.norm(K - P) <= np.linalg.norm(K - other) + 1e-12


def test_a_scalar_stiffness_is_a_one_by_one_matrix():
    assert project_psd([[2.5]]) == pytest.approx(2.5)
