"""Joint coordinates of the robot, and states of virtual degrees of freedom."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from .base import Context, Coordinate


def as_indices(index: Any) -> list[int]:
    """An int (a numpy integer too) or a sequence of them, as a list of ints."""
    try:
        return [operator.index(index)]
    except TypeError:
        return [int(i) for i in index]


def select_indices(index: Any, n: int, what: str = "coordinates") -> list[int]:
    """The entries an index (an int, a slice or a list of ints) picks out of ``n``, negative ones
    counting from the end; a ValueError when one is out of range."""
    if isinstance(index, slice):
        for bound in (index.start, index.stop):
            if bound is not None and not -n <= bound <= n:
                raise ValueError(f"{index} is out of range: there are {n} {what}")
        return list(range(*index.indices(n)))
    picked = as_indices(index)
    for i in picked:
        if not -n <= i < n:
            raise ValueError(f"index {i} is out of range: there are {n} {what}")
    return [i % n for i in picked]


class Joint(Coordinate):
    """Entries of the robot's generalized coordinates q: an index, a slice or a list of indices.

    Built by hand, a slice needs a non-negative stop; ``robot.joint`` also counts from the end.
    """

    def __init__(self, index: int | slice | Sequence[int], unit: str = "") -> None:
        if isinstance(index, slice):
            if index.stop is None:
                raise ValueError("a Joint slice needs an explicit stop")
            if index.stop < 0 or (index.start or 0) < 0:
                raise ValueError(
                    "give a non-negative stop (and start): a Joint does not know how many "
                    "coordinates there are, but robot.joint counts from the end"
                )
            indices = list(range(index.stop))[index]
        else:
            indices = as_indices(index)
        super().__init__(len(indices), unit)
        self.indices = indices

    def value(self, ctx: Context) -> Any:
        """Selected entries of q."""
        return ca.vertcat(*[ctx.q[i] for i in self.indices])

    def __repr__(self) -> str:
        return f"Joint({self.indices}, unit={self.unit!r})"


class MotorAngle(Coordinate):
    """Entries of the robot's motor angles θ(q) [rad], as its actuation gives them: positive when a
    motor pulls a tendon. A limit on how far a motor may turn acts on this."""

    def __init__(self, actuation: Any, indices: Sequence[int]) -> None:
        super().__init__(len(indices), "rad")
        self.actuation = actuation
        self.indices = [int(i) for i in indices]

    def value(self, ctx: Context) -> Any:
        """Selected entries of θ(q)."""
        theta = self.actuation.motor_angles(ctx.q, ctx.view(self.actuation.params))
        return ca.vertcat(*[theta[i] for i in self.indices])

    def __repr__(self) -> str:
        return f"MotorAngle({self.indices})"


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
