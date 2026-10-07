"""The nodes of a planned motion, as the terms of a problem see them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Evaluate = Callable[[Any, list[Any], list[Any], list[Any]], tuple[list[Any], list[Any]]]


@dataclass
class Trajectory:
    """Node times ``t`` [s], and per node the configuration ``q``, velocity ``v``, acceleration
    ``a`` and motor torques ``u`` as CasADi columns, plus the ``blend`` weight of the new
    controller and, when it has virtual states, its state ``z`` (positions, then velocities).
    ``shapes`` gives the (nodes, size) of q, v, a and z. The first node is the fixed
    start of a motion, unless ``fixed_start`` is off (a static problem or a periodic one).

    The spacing ``dt`` is a number, or an expression of the horizon when it is free: integrate
    with it, and with ``times``, which holds the same kind of values. ``t`` stays the numbers at
    the starting horizon: the windows of the terms use it.
    """

    t: np.ndarray
    dt: Any
    q: list[Any]
    v: list[Any]
    a: list[Any]
    u: list[Any]
    blend: np.ndarray
    shapes: dict[str, tuple[int, int]]
    evaluate: Evaluate
    fixed_start: bool = True
    z: list[Any] = field(default_factory=list)

    @property
    def times(self) -> list[Any]:
        """The node times [s]: numbers, or expressions of the horizon when it is free."""
        return [k * self.dt if k else 0.0 for k in range(len(self.q))]

    @property
    def horizon(self) -> Any:
        """The time from the first node to the last [s]: a number, or the free horizon."""
        return (len(self.q) - 1) * self.dt

    def coordinate(self, coord: Any) -> tuple[list[Any], list[Any]]:
        """The value ``y`` and rate ``ẏ`` of a library coordinate at every node."""
        return self.evaluate(coord, self.q, self.v, self.times)

    def window(self, t_from: float = 0.0, t_to: float | None = None) -> list[int]:
        """Indices of the nodes with ``t_from <= t <= t_to`` (to the end by default)."""
        end = np.inf if t_to is None else t_to + 1e-9
        return [k for k, t in enumerate(self.t) if t_from - 1e-9 <= t <= end]
