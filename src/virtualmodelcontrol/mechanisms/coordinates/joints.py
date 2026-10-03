"""Joint coordinates of the robot, and states of virtual degrees of freedom."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from .base import Context, Coordinate


class Joint(Coordinate):
    """Entries of the robot's generalized coordinates q: an index, a slice or a list of indices."""

    def __init__(self, index: int | slice | Sequence[int], unit: str = "") -> None:
        if isinstance(index, slice):
            if index.stop is None:
                raise ValueError("a Joint slice needs an explicit stop")
            indices = list(range(index.stop))[index]
        elif isinstance(index, int):
            indices = [index]
        else:
            indices = [int(i) for i in index]
        super().__init__(len(indices), unit)
        self.indices = indices

    def value(self, ctx: Context) -> Any:
        """Selected entries of q."""
        return ca.vertcat(*[ctx.q[i] for i in self.indices])

    def __repr__(self) -> str:
        return f"Joint({self.indices}, unit={self.unit!r})"


class State(Coordinate):
    """Position of a virtual degree of freedom owned by a controller (part of its state z)."""

    def __init__(self, name: str, dim: int = 1, unit: str = "", initial: ArrayLike = 0.0) -> None:
        super().__init__(dim, unit)
        self.name = name
        self.initial = np.broadcast_to(np.asarray(initial, dtype=float), (dim,)).copy()

    def value(self, ctx: Context) -> Any:
        """Slice of z."""
        return ctx.state(self)

    def __repr__(self) -> str:
        return f"State({self.name!r}, dim={self.dim}, unit={self.unit!r})"
