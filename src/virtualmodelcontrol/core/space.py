"""Configuration spaces: q lives on the space (nq numbers), v in its tangent space (nv numbers)."""

from __future__ import annotations

from typing import Any, Protocol

import casadi as ca
import numpy as np

from .symbolic import is_casadi

EPS = 1e-24
"""Added under the square root of a squared angle, so that every function is smooth at zero."""


class Space(Protocol):
    """Where a configuration lives. ``integrate`` and ``difference`` accept numpy or CasADi."""

    nq: int
    nv: int

    def neutral(self) -> np.ndarray:
        """Reference configuration (zero displacement)."""
        ...

    def integrate(self, q: Any, v: Any) -> Any:
        """Configuration reached from ``q`` by the tangent step ``v``."""
        ...

    def difference(self, q1: Any, q0: Any) -> Any:
        """Tangent step ``v`` with ``integrate(q0, v) = q1``."""
        ...

    def velocity_map(self, q: Any) -> Any:
        """G(q) with q̇ = G(q) v, shape (nq, nv)."""
        ...

    def coadjoint(self, v: Any, mu: Any) -> Any:
        """The term that velocities that do not commute add to the momentum's rate: ad*_v μ."""
        ...


def _numeric_call(fn: Any, *args: Any) -> Any:
    """Run a CasADi-written ``fn`` on numpy inputs and return numpy; pass CasADi through."""
    if any(is_casadi(a) for a in args):
        return fn(*args)
    out = np.array(fn(*[ca.DM(np.asarray(a, dtype=float).reshape(-1)) for a in args]))
    return out.ravel() if out.shape[1] == 1 else out


class Euclidean:
    """Flat space: q and v are both n-vectors."""

    def __init__(self, n: int) -> None:
        self.nq = self.nv = int(n)

    def neutral(self) -> np.ndarray:
        """Origin."""
        return np.zeros(self.nq)

    def integrate(self, q: Any, v: Any) -> Any:
        """``q + v``."""
        if is_casadi(q) or is_casadi(v):
            return q + v
        return np.asarray(q, dtype=float) + np.asarray(v, dtype=float)

    def difference(self, q1: Any, q0: Any) -> Any:
        """``q1 - q0``."""
        if is_casadi(q1) or is_casadi(q0):
            return q1 - q0
        return np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)

    def velocity_map(self, q: Any) -> Any:
        """Identity."""
        return type(q).eye(self.nq) if is_casadi(q) else np.eye(self.nq)

    def coadjoint(self, v: Any, mu: Any) -> Any:
        """Zero: the velocities commute."""
        return 0 * v

    def __repr__(self) -> str:
        return f"Euclidean({self.nq})"


class SO2:
    """Planar rotations as unit complex numbers: q = (cos θ, sin θ), v = angle step (rad)."""

    nq = 2
    nv = 1

    def neutral(self) -> np.ndarray:
        """Zero angle."""
        return np.array([1.0, 0.0])

    def integrate(self, q: Any, v: Any) -> Any:
        """Rotate ``q`` by the angle ``v``."""

        def fn(q: Any, v: Any) -> Any:
            c, s, w = q[0], q[1], v[0]
            return ca.vertcat(c * ca.cos(w) - s * ca.sin(w), s * ca.cos(w) + c * ca.sin(w))

        return _numeric_call(fn, q, v)

    def difference(self, q1: Any, q0: Any) -> Any:
        """Angle from ``q0`` to ``q1``, in (-π, π]."""

        def fn(q1: Any, q0: Any) -> Any:
            return ca.atan2(q1[1] * q0[0] - q1[0] * q0[1], q1[0] * q0[0] + q1[1] * q0[1])

        return _numeric_call(fn, q1, q0)

    def velocity_map(self, q: Any) -> Any:
        """(-sin θ, cos θ) as a column."""
        if is_casadi(q):
            return ca.vertcat(-q[1], q[0])
        c, s = np.asarray(q, dtype=float).ravel()
        return np.array([[-s], [c]])

    def coadjoint(self, v: Any, mu: Any) -> Any:
        """Zero: there is one angular velocity."""
        return 0 * v

    def __repr__(self) -> str:
        return "SO2()"


class Quaternion:
    """Rotations of a body as unit quaternions (w, x, y, z): q has 4 entries, and v is the body's
    angular velocity in its own frame [rad/s], 3 entries. A body can turn any number of times."""

    nq = 4
    nv = 3

    def neutral(self) -> np.ndarray:
        """No rotation."""
        return np.array([1.0, 0.0, 0.0, 0.0])

    def integrate(self, q: Any, v: Any) -> Any:
        """Turn ``q`` by the rotation vector ``v`` about the body's own axes; stays a unit."""

        def fn(q: Any, v: Any) -> Any:
            angle = ca.sqrt(ca.dot(v, v) + EPS)
            k, a = ca.sin(angle / 2) / angle, ca.cos(angle / 2)
            w, x, y, z = (q[i] for i in range(4))
            bx, by, bz = (k * v[i] for i in range(3))
            out = ca.vertcat(
                w * a - x * bx - y * by - z * bz,
                w * bx + x * a + y * bz - z * by,
                w * by - x * bz + y * a + z * bx,
                w * bz + x * by - y * bx + z * a,
            )
            return out / ca.norm_2(out)

        return _numeric_call(fn, q, v)

    def difference(self, q1: Any, q0: Any) -> Any:
        """The rotation vector, of angle at most π, that takes ``q0`` to ``q1``."""

        def fn(q1: Any, q0: Any) -> Any:
            w0, x0, y0, z0 = (q0[i] for i in range(4))
            w1, x1, y1, z1 = (q1[i] for i in range(4))
            w = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1  # conj(q0) times q1
            x = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
            y = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
            z = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
            sign = ca.if_else(w < 0, -1.0, 1.0)  # the short way round
            w, x, y, z = sign * w, sign * x, sign * y, sign * z
            s = ca.sqrt(x**2 + y**2 + z**2 + EPS)
            return 2 * ca.atan2(s, w) / s * ca.vertcat(x, y, z)

        return _numeric_call(fn, q1, q0)

    def velocity_map(self, q: Any) -> Any:
        """G(q) with q̇ = G(q) v, the rows of ½ q ⊗ (0, v)."""
        if is_casadi(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            rows = [[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]]
            return 0.5 * ca.vertcat(*[ca.horzcat(*row) for row in rows])
        w, x, y, z = np.asarray(q, dtype=float).ravel()
        return 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])

    def coadjoint(self, v: Any, mu: Any) -> Any:
        """μ × v: what turns a spinning body's angular momentum away from its angular velocity."""
        return ca.cross(mu, v) if is_casadi(v) or is_casadi(mu) else np.cross(mu, v)

    def __repr__(self) -> str:
        return "Quaternion()"


class Product:
    """Cartesian product of spaces; q and v are the blocks stacked in order."""

    def __init__(self, *spaces: Space) -> None:
        self.spaces = spaces
        self.nq = sum(s.nq for s in spaces)
        self.nv = sum(s.nv for s in spaces)

    def _blocks(self, x: Any, attr: str) -> list[Any]:
        out, offset = [], 0
        for space in self.spaces:
            n = getattr(space, attr)
            out.append(x[offset : offset + n])
            offset += n
        return out

    @staticmethod
    def _cat(parts: list[Any]) -> Any:
        if any(is_casadi(p) for p in parts):
            return ca.vertcat(*parts)
        return np.concatenate([np.ravel(p) for p in parts])

    def neutral(self) -> np.ndarray:
        """Each block's neutral configuration."""
        return np.concatenate([s.neutral() for s in self.spaces])

    def integrate(self, q: Any, v: Any) -> Any:
        """Integrate block by block."""
        qs, vs = self._blocks(q, "nq"), self._blocks(v, "nv")
        return self._cat([s.integrate(a, b) for s, a, b in zip(self.spaces, qs, vs, strict=True)])

    def difference(self, q1: Any, q0: Any) -> Any:
        """Difference block by block."""
        a, b = self._blocks(q1, "nq"), self._blocks(q0, "nq")
        return self._cat([s.difference(x, y) for s, x, y in zip(self.spaces, a, b, strict=True)])

    def velocity_map(self, q: Any) -> Any:
        """Block-diagonal G(q)."""
        qs = self._blocks(q, "nq")
        blocks = [s.velocity_map(x) for s, x in zip(self.spaces, qs, strict=True)]
        if any(is_casadi(b) for b in blocks):
            return ca.diagcat(*blocks)
        out, i, j = np.zeros((self.nq, self.nv)), 0, 0
        for block in blocks:
            r, c = np.shape(block)
            out[i : i + r, j : j + c] = block
            i, j = i + r, j + c
        return out

    def coadjoint(self, v: Any, mu: Any) -> Any:
        """Block by block."""
        a, b = self._blocks(v, "nv"), self._blocks(mu, "nv")
        return self._cat([s.coadjoint(x, y) for s, x, y in zip(self.spaces, a, b, strict=True)])

    def __repr__(self) -> str:
        return f"Product({', '.join(map(repr, self.spaces))})"
