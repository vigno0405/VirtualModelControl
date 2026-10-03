"""Configuration spaces: q lives on the space (nq numbers), v in its tangent space (nv numbers)."""

from __future__ import annotations

from typing import Any, Protocol

import casadi as ca
import numpy as np

from .symbolic import is_casadi


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

    def __repr__(self) -> str:
        return "SO2()"


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

    def __repr__(self) -> str:
        return f"Product({', '.join(map(repr, self.spaces))})"
