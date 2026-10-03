import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation

from virtualmodelcontrol.core.symbolic import exp_so3, gauss_legendre, quad, skew, smooth_norm


def test_gauss_legendre_is_exact_up_to_degree_2n_minus_1():
    t, w = gauss_legendre(4)
    for k in range(8):
        assert abs(np.sum(w * t**k) - 1.0 / (k + 1)) < 1e-14


def test_quad_works_on_symbols():
    b = ca.SX.sym("b")
    f = ca.Function("f", [b], [quad(lambda x: x**3, 0.0, b, n=2)])
    assert abs(float(f(2.0)) - 4.0) < 1e-13


def test_smooth_norm_is_differentiable_at_zero():
    x = ca.SX.sym("x", 3)
    grad = ca.Function("g", [x], [ca.gradient(smooth_norm(x), x)])
    assert np.all(np.isfinite(np.array(grad([0.0, 0.0, 0.0]))))
    assert abs(float(ca.Function("n", [x], [smooth_norm(x)])([3.0, 4.0, 0.0])) - 5.0) < 1e-12


def test_skew_and_rodrigues():
    w, x = np.array([0.3, -1.2, 0.5]), np.array([1.0, 2.0, -0.5])
    np.testing.assert_allclose(np.array(skew(w)) @ x, np.cross(w, x), atol=1e-15)
    axis, angle = w / np.linalg.norm(w), 1.1
    R = np.array(exp_so3(axis, angle))
    np.testing.assert_allclose(R, Rotation.from_rotvec(axis * angle).as_matrix(), atol=1e-14)
