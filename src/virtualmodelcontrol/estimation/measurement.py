"""What a sensor saw: some of the coordinates q and rates v of the robot, with their noise."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def block_diagonal(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """The matrix with the square ``blocks`` along its diagonal."""
    out = np.zeros((sum(b.shape[0] for b in blocks),) * 2)
    i = 0
    for b in blocks:
        out[i : i + b.shape[0], i : i + b.shape[0]] = b
        i += b.shape[0]
    return out


def covariance(R: ArrayLike, m: int, what: str = "covariance") -> np.ndarray:
    """The (m, m) covariance of a variance, of m variances (one per coordinate) or of a matrix."""
    R = np.asarray(R, dtype=float)
    if R.ndim == 0:
        return R * np.eye(m)
    if R.shape == (m,):
        return np.diag(R)
    if R.shape == (m, m):
        return R
    raise ValueError(
        f"{what} must be a variance, {m} variances or a ({m}, {m}) matrix, not {R.shape}"
    )


class Measurement:
    """What one sensor saw at one step: the configuration ``q`` and/or the velocity ``v``.

    ``Rq`` and ``Rv`` are their covariances (a variance, one per coordinate or a matrix).
    ``observed`` lists the coordinates the sensor sees, in the order of ``q`` and ``v`` (all by
    default). A sensor that sees combinations of them, such as tendon lengths, has a ``matrix``
    instead (m × n): it reads ``matrix @ q`` and ``matrix @ v``. ``name`` is how the filter
    reports the sensor.
    """

    def __init__(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        Rq: ArrayLike | None = None,
        Rv: ArrayLike | None = None,
        observed: ArrayLike | None = None,
        name: str | None = None,
        matrix: ArrayLike | None = None,
    ) -> None:
        if q is None and v is None:
            raise ValueError("a measurement needs q, v or both")
        blocks = [(q, Rq, "Rq"), (v, Rv, "Rv")]
        seen = [np.asarray(x, dtype=float).ravel() for x, _, _ in blocks if x is not None]
        m = seen[0].size
        if any(x.size != m for x in seen):
            raise ValueError(f"q and v must have the same size, got {[x.size for x in seen]}")
        covariances = []
        for x, R, label in blocks:
            if x is not None:
                if R is None:
                    raise ValueError(f"{label} is needed with the values it belongs to")
                covariances.append(covariance(R, m, label))
        self.observed = None if observed is None else np.asarray(observed, dtype=int).ravel()
        if self.observed is not None and self.observed.size != m:
            raise ValueError(f"{self.observed.size} observed coordinates for {m} values")
        self.matrix = None if matrix is None else np.atleast_2d(np.asarray(matrix, dtype=float))
        if self.matrix is not None:
            if self.observed is not None:
                raise ValueError("a measurement has observed coordinates or a matrix, not both")
            if self.matrix.shape[0] != m:
                raise ValueError(f"the matrix has {self.matrix.shape[0]} rows for {m} values")
        self.y = np.concatenate(seen)
        self.R = block_diagonal(covariances)
        self.name = name
        self._has = (q is not None, v is not None)

    def H(self, n: int) -> np.ndarray:
        """The rows that the sensor sees of a state (q, v) of n coordinates each: those of the
        identity, or the ``matrix`` on q and on v."""
        m = self.y.size // sum(self._has)
        if self.matrix is not None:
            if self.matrix.shape[1] != n:
                raise ValueError(f"the matrix has {self.matrix.shape[1]} columns, the state {n}")
            none = np.zeros_like(self.matrix)
            rows = [
                np.hstack(pair)
                for pair, has in zip(
                    [(self.matrix, none), (none, self.matrix)], self._has, strict=True
                )
                if has
            ]
            return np.vstack(rows)
        if self.observed is None and m != n:
            raise ValueError(f"the measurement has {m} coordinates, the state {n}")
        idx = np.arange(n) if self.observed is None else self.observed
        rows = [idx + k * n for k, has in enumerate(self._has) if has]
        return np.eye(2 * n)[np.concatenate(rows)]
