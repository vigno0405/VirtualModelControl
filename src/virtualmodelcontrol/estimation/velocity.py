"""Velocity of a sampled signal: the low-passed finite difference of its consecutive samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

ALPHA = 0.3
"""Default smoothing: the share of the previous velocity that the new one keeps."""


class VelocityFilter:
    """Velocity of a sampled vector: the difference of consecutive samples over ``dt``, low-passed.

    ``v = (1 - alpha) raw + alpha v_prev``, ``alpha`` in [0, 1). Unlike the lab's, it does not
    invert markers (feed it the inversion's q), and ``dt`` [s] is the interval between the samples
    it receives, the sensor's own period. The first sample, after a ``reset`` too, gives zeros.
    """

    def __init__(self, dt: float, alpha: float = ALPHA) -> None:
        self._dt = dt
        self._alpha = alpha
        self.reset()

    def reset(self) -> None:
        """Forget the last sample: the next one gives zeros."""
        self._last: np.ndarray | None = None

    def update(self, q: ArrayLike) -> np.ndarray:
        """Take the next sample ``q`` and return the velocity."""
        q = np.asarray(q, dtype=float)
        if self._last is None:
            self._v = np.zeros_like(q)
        else:
            self._v = (1.0 - self._alpha) * (q - self._last) / self._dt + self._alpha * self._v
        self._last = q.copy()
        return self._v.copy()
