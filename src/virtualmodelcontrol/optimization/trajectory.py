"""The nodes of a planned motion, as the terms of a problem see them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Evaluate = Callable[[Any, list[Any], list[Any], np.ndarray], tuple[list[Any], list[Any]]]


@dataclass
class Trajectory:
    """Node times ``t`` [s] (spacing ``dt``), and per node the configuration ``q``, velocity ``v``,
    acceleration ``a`` and motor torques ``u`` as CasADi columns, plus the ``blend`` weight of the
    new controller. ``shapes`` gives the (nodes, size) of q, v and a. The first node is the fixed
    start of a motion, unless ``fixed_start`` is off (a static problem)."""

    t: np.ndarray
    dt: float
    q: list[Any]
    v: list[Any]
    a: list[Any]
    u: list[Any]
    blend: np.ndarray
    shapes: dict[str, tuple[int, int]]
    evaluate: Evaluate
    fixed_start: bool = True

    def coordinate(self, coord: Any) -> tuple[list[Any], list[Any]]:
        """The value ``y`` and rate ``ẏ`` of a library coordinate at every node."""
        return self.evaluate(coord, self.q, self.v, self.t)

    def window(self, t_from: float = 0.0, t_to: float | None = None) -> list[int]:
        """Indices of the nodes with ``t_from <= t <= t_to`` (to the end by default)."""
        end = np.inf if t_to is None else t_to + 1e-9
        return [k for k, t in enumerate(self.t) if t_from - 1e-9 <= t <= end]
