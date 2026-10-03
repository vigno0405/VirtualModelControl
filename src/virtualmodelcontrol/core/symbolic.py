"""Symbolic helpers shared by the models: type checks, smooth norms, quadrature, rotations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np


def is_casadi(x: Any) -> bool:
    """True for CasADi symbols (SX, MX) and numeric matrices (DM)."""
    return isinstance(x, (ca.SX, ca.MX, ca.DM))


def is_symbolic(x: Any) -> bool:
    """True for CasADi symbols (SX, MX)."""
    return isinstance(x, (ca.SX, ca.MX))


def smooth_norm(x: Any, eps: float = 1e-18) -> Any:
    """√(xᵀx + ε): the Euclidean norm, differentiable at zero."""
    return ca.sqrt(ca.sumsqr(x) + eps)


def logcosh(x: Any) -> Any:
    """log(cosh x), written to avoid overflow for large arguments."""
    a = ca.fabs(x)
    return a + ca.log1p(ca.exp(-2.0 * a)) - np.log(2.0)


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weights on [0, 1], exact for polynomials of degree 2n − 1."""
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (t + 1.0), 0.5 * w


def quad(f: Callable[[Any], Any], a: Any, b: Any, n: int = 16) -> Any:
    """∫ f over [a, b] by n-point Gauss–Legendre quadrature; a, b and f may be symbolic."""
    t, w = gauss_legendre(n)
    h = b - a
    return h * sum(float(wi) * f(a + h * float(ti)) for ti, wi in zip(t, w, strict=True))


def skew(w: Any) -> Any:
    """Skew-symmetric matrix [w]× with [w]× x = w × x."""
    return ca.vertcat(
        ca.horzcat(0, -w[2], w[1]),
        ca.horzcat(w[2], 0, -w[0]),
        ca.horzcat(-w[1], w[0], 0),
    )


def exp_so3(axis: Any, angle: Any) -> Any:
    """Rotation by ``angle`` (rad) about the unit vector ``axis`` (Rodrigues)."""
    k = skew(axis)
    return ca.DM.eye(3) + ca.sin(angle) * k + (1 - ca.cos(angle)) * ca.mtimes(k, k)


def rotation_from_vector(w: Any) -> Any:
    """Rotation matrix of a rotation vector w (axis · angle [rad]); smooth at w = 0."""
    theta = ca.sqrt(ca.sumsqr(w) + 1e-24)
    k = skew(w)
    return (
        ca.DM.eye(3) + ca.sin(theta) / theta * k + (1 - ca.cos(theta)) / theta**2 * ca.mtimes(k, k)
    )
