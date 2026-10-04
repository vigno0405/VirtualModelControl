"""Static points of a log: the settled end of every stretch where a command was held."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def plateaus(
    t: ArrayLike,
    held: ArrayLike,
    tolerance: float,
    *,
    min_hold: float = 8.0,
    steady_fraction: float = 0.5,
) -> list[slice]:
    """The settled part of every hold of ``held``, as slices of the log.

    A hold is a stretch where ``held`` stays within ``tolerance`` of its first value for at least
    ``min_hold`` [s]; its settled part is the last ``steady_fraction`` of it. Average a signal over
    a slice for one static point; a large spread over it means the point never settled.
    """
    t = np.asarray(t, dtype=float)
    held = np.asarray(held, dtype=float).reshape(len(t), -1)
    out, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or np.max(np.abs(held[i] - held[start])) > tolerance:
            if t[i - 1] - t[start] >= min_hold:
                out.append(slice(start + int((i - start) * (1.0 - steady_fraction)), i))
            start = i
    return out
