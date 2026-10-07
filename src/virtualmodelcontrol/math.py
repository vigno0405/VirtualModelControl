"""Rotations and rigid transforms that run on numpy arrays and CasADi symbols alike.

Write a model's ``frame`` with these and it works on numbers, to look at it, and on symbols, to
differentiate it. A rotation vector is the axis times the angle [rad]; a twist is ``(ω, v)``.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .core.symbolic import is_casadi

__all__ = [
    "EPS",
    "SMALL",
    "adjoint",
    "exp_se3",
    "exp_so3",
    "invert",
    "log_se3",
    "log_so3",
    "quat_rot",
    "rot",
    "rot_x",
    "rot_y",
    "rot_z",
    "skew",
    "transform",
    "vee",
]

EPS = 1e-24
"""Added under the square root of a squared angle, so that every function is smooth at zero."""
SMALL = 1e-2
"""Angles below this [rad] use a series where the exact form loses its digits."""


class _Numpy:
    sin, cos, sqrt, atan2 = np.sin, np.cos, np.sqrt, np.arctan2
    dot, eye, where = staticmethod(np.dot), staticmethod(np.eye), staticmethod(np.where)

    @staticmethod
    def col(x: Any) -> Any:
        return np.ravel(np.asarray(x, dtype=float))

    @staticmethod
    def mat(rows: list[list[Any]]) -> Any:
        return np.array(rows, dtype=float)

    @staticmethod
    def vec(items: list[Any]) -> Any:
        return np.array(items, dtype=float)

    @staticmethod
    def block(rows: list[list[Any]]) -> Any:
        return np.block(rows)


class _Casadi:
    sin, cos, sqrt, atan2 = ca.sin, ca.cos, ca.sqrt, ca.atan2
    dot, where = staticmethod(ca.dot), staticmethod(ca.if_else)

    @staticmethod
    def eye(n: int) -> Any:
        return ca.DM.eye(n)

    @staticmethod
    def col(x: Any) -> Any:
        return ca.reshape(x, -1, 1)

    @staticmethod
    def mat(rows: list[list[Any]]) -> Any:
        return ca.vertcat(*[ca.horzcat(*row) for row in rows])

    @staticmethod
    def vec(items: list[Any]) -> Any:
        return ca.vertcat(*items)

    @staticmethod
    def block(rows: list[list[Any]]) -> Any:
        return ca.vertcat(*[ca.horzcat(*row) for row in rows])


def _ops(*xs: Any) -> Any:
    return _Casadi if any(is_casadi(x) for x in xs) else _Numpy


def _angle(ops: Any, w: Any) -> Any:
    return ops.sqrt(ops.dot(w, w) + EPS)


def rot_x(angle: Any) -> Any:
    """Rotation by ``angle`` [rad] about x."""
    o = _ops(angle)
    c, s = o.cos(angle), o.sin(angle)
    return o.mat([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(angle: Any) -> Any:
    """Rotation by ``angle`` [rad] about y."""
    o = _ops(angle)
    c, s = o.cos(angle), o.sin(angle)
    return o.mat([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(angle: Any) -> Any:
    """Rotation by ``angle`` [rad] about z."""
    o = _ops(angle)
    c, s = o.cos(angle), o.sin(angle)
    return o.mat([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skew(w: Any) -> Any:
    """The matrix [w]× with [w]× x = w × x."""
    o = _ops(w)
    w = o.col(w)
    return o.mat([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(S: Any) -> Any:
    """The vector w of a skew-symmetric matrix [w]×."""
    return _ops(S).vec([S[2, 1], S[0, 2], S[1, 0]])


def rot(axis: Any, angle: Any) -> Any:
    """Rotation by ``angle`` [rad] about the unit vector ``axis``."""
    o = _ops(axis, angle)
    k = skew(axis)
    return o.eye(3) + o.sin(angle) * k + (1 - o.cos(angle)) * (k @ k)


def quat_rot(q: Any) -> Any:
    """The rotation matrix of the unit quaternion ``q`` = (w, x, y, z)."""
    o = _ops(q)
    q = o.col(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return o.mat([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def exp_so3(w: Any) -> Any:
    """The rotation matrix of the rotation vector ``w``, smooth at 0."""
    o = _ops(w)
    w = o.col(w)
    theta, k = _angle(o, w), skew(w)
    half = o.sin(theta / 2) / (theta / 2)  # (1 - cos θ) / θ² is half² / 2: exact for a tiny θ
    return o.eye(3) + o.sin(theta) / theta * k + 0.5 * half**2 * (k @ k)


def log_so3(R: Any) -> Any:
    """The rotation vector of ``R``, whose angle is below π; smooth at the identity."""
    o = _ops(R)
    v = vee(R - R.T) / 2  # sin θ times the unit axis
    s = _angle(o, v)
    return o.atan2(s, (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2) / s * v


def transform(R: Any, p: Any) -> Any:
    """The 4 by 4 matrix of the rotation ``R`` followed by the translation ``p``."""
    o = _ops(R, p)
    p = o.col(p)
    return o.mat([[R[i, j] for j in range(3)] + [p[i]] for i in range(3)] + [[0, 0, 0, 1]])


def invert(T: Any) -> Any:
    """The inverse of a 4 by 4 rigid transform."""
    R, p = T[:3, :3], _ops(T).col(T[:3, 3])
    return transform(R.T, -(R.T @ p))


def exp_se3(twist: Any) -> Any:
    """The transform of the twist ``(ω, v)`` run for one unit of time, smooth at 0."""
    o = _ops(twist)
    twist = o.col(twist)
    w, v = twist[:3], twist[3:]
    theta, k = _angle(o, w), skew(w)
    half = o.sin(theta / 2) / (theta / 2)
    t2 = theta**2
    b = o.where(theta < SMALL, 1 / 6 - t2 / 120 + t2**2 / 5040, (1 - o.sin(theta) / theta) / t2)
    V = o.eye(3) + 0.5 * half**2 * k + b * (k @ k)  # (1 - cos θ) / θ² and (θ - sin θ) / θ³
    return transform(exp_so3(w), V @ v)


def log_se3(T: Any) -> Any:
    """The twist ``(ω, v)`` of a rigid transform whose rotation angle is below π."""
    o = _ops(T)
    w = log_so3(T[:3, :3])
    theta, k = _angle(o, w), skew(w)
    t2, x = theta**2, theta / 2
    c = o.where(
        theta < SMALL, 1 / 12 + t2 / 720 + t2**2 / 30240, (1 - x * o.cos(x) / o.sin(x)) / t2
    )
    v = (o.eye(3) - 0.5 * k + c * (k @ k)) @ o.col(T[:3, 3])  # V⁻¹ p
    return o.vec([w[0], w[1], w[2], v[0], v[1], v[2]])


def adjoint(T: Any) -> Any:
    """The 6 by 6 matrix that carries a twist ``(ω, v)`` across the transform ``T``."""
    o = _ops(T)
    R, p = T[:3, :3], skew(T[:3, 3])
    return o.block([[R, 0 * R], [p @ R, R]])
