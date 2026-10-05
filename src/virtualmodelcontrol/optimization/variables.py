"""Decision variables stacked into one vector, each scaled so that the solver sees values near 1."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


class Variables:
    """Named decision variables with bounds and starting values.

    The solver sees ``value / scale``; ``add`` returns the physical value for the model.
    """

    def __init__(self) -> None:
        self._symbols: list[Any] = []
        self._lower: list[np.ndarray] = []
        self._upper: list[np.ndarray] = []
        self._init: list[np.ndarray] = []
        self.slices: dict[str, slice] = {}
        self.scales: dict[str, float] = {}
        self.size = 0

    def add(
        self, name: str, size: int, lower: Any, upper: Any, init: Any, scale: float = 1.0
    ) -> Any:
        """Add ``size`` variables; returns their value as an MX column."""
        if name in self.slices:
            raise ValueError(f"a variable named {name!r} already exists")
        symbol = ca.MX.sym(name, size)
        self._symbols.append(symbol)
        for store, value in ((self._lower, lower), (self._upper, upper), (self._init, init)):
            store.append(np.broadcast_to(np.asarray(value, dtype=float), size) / scale)
        self.slices[name] = slice(self.size, self.size + size)
        self.scales[name] = scale
        self.size += size
        return symbol * scale

    @property
    def x(self) -> Any:
        """The stacked (scaled) variables."""
        return ca.vertcat(*self._symbols)

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds, scaled."""
        return np.concatenate(self._lower) if self._lower else np.zeros(0)

    @property
    def upper(self) -> np.ndarray:
        """Upper bounds, scaled."""
        return np.concatenate(self._upper) if self._upper else np.zeros(0)

    @property
    def init(self) -> np.ndarray:
        """Starting values, scaled."""
        return np.concatenate(self._init) if self._init else np.zeros(0)

    def value(self, x: np.ndarray, name: str) -> np.ndarray:
        """Physical value of a variable in a (scaled) solution vector."""
        return np.asarray(x, dtype=float)[self.slices[name]] * self.scales[name]

    def scaled(self, name: str, value: Any) -> np.ndarray:
        """A physical value of a variable, scaled for the solver."""
        return np.asarray(value, dtype=float).ravel() / self.scales[name]
