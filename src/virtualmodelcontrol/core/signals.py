"""Signals: named SI arrays stamped with a time, each with a valid flag."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class Signals:
    """Named SI arrays at time ``t`` [s]; an array counts as valid only if flagged and finite."""

    def __init__(self, t: float = 0.0, **values: ArrayLike) -> None:
        self.t = float(t)
        self._data: dict[str, np.ndarray] = {}
        self._valid: dict[str, bool] = {}
        for name, value in values.items():
            self.set(name, value)

    def set(self, name: str, value: ArrayLike, valid: bool = True) -> None:
        """Store ``value`` (as a 1-D float array) under ``name``."""
        self._data[name] = np.atleast_1d(np.asarray(value, dtype=float))
        self._valid[name] = bool(valid)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._data[name]

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def get(self, name: str, default: np.ndarray | None = None) -> np.ndarray | None:
        """Array under ``name``, or ``default``."""
        return self._data.get(name, default)

    def is_valid(self, name: str) -> bool:
        """True if present, flagged valid and free of NaN and Inf."""
        return (
            name in self._data and self._valid[name] and bool(np.all(np.isfinite(self._data[name])))
        )

    @property
    def names(self) -> list[str]:
        """Names of the stored arrays."""
        return list(self._data)

    def __repr__(self) -> str:
        return f"Signals(t={self.t}, {', '.join(self._data)})"
