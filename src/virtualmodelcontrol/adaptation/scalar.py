"""Laws that need no model: they scale or set a Param from the force alone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .limits import admissible, live_matching


class ForceRatio:
    """Scales live Params by the wanted force over the measured one.

    A step multiplies each Param by ``f_des / f_meas`` (the norms of the forces), but by at most
    ``max_change`` times the relative error: large steps far from the target, small ones near it.
    Nothing happens while the measured force is near zero. A new value stays within the Param's
    bounds, and a square matrix stays symmetric and positive semidefinite.
    """

    def __init__(
        self, controller: Any, params: str | Sequence[str], max_change: float = 0.05
    ) -> None:
        compiled = controller.compiled
        self.names, self.max_change = live_matching(compiled, params), max_change
        self._params = {n: compiled.params[n] for n in self.names}

    def step(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> float:
        """One step on ``controller`` (or a ``Tank`` around it); returns the factor applied."""
        measured = float(np.linalg.norm(f_meas))
        wanted = float(np.linalg.norm(f_des))
        if measured <= 1e-6:
            return 1.0
        change = self.max_change * abs(measured - wanted) / max(measured, wanted)
        ratio = float(np.clip(wanted / measured, 1.0 - change, 1.0 + change))
        live = controller.live_params()
        controller.set(
            {
                n: admissible(p, ratio * np.ravel(live[n], order="F"))
                for n, p in self._params.items()
            }
        )
        return ratio


class Stiffening:
    """Sets live Params to a stiffness that grows with the measured force, up to a limit.

    k(F) = low + (high − low) (1 − exp(−rate F)): ``low`` without force, ``high`` for a large one,
    half way at F = ln 2 / ``rate``. Every entry of a Param gets k (a square matrix, k times the
    identity).
    """

    def __init__(
        self,
        controller: Any,
        params: str | Sequence[str],
        low: float,
        high: float,
        rate: float,
    ) -> None:
        compiled = controller.compiled
        self.names = live_matching(compiled, params)
        self.low, self.high, self.rate = low, high, rate
        self._params = {n: compiled.params[n] for n in self.names}

    def stiffness(self, force: float) -> float:
        """k for a measured force [N]."""
        return self.low + (self.high - self.low) * (1.0 - np.exp(-self.rate * max(force, 0.0)))

    def step(self, controller: Any, force: float) -> float:
        """Set the Params on ``controller`` (or a ``Tank`` around it); returns k."""
        k = float(self.stiffness(force))
        values = {}
        for name, param in self._params.items():
            full = k * np.eye(param.shape[0]) if len(param.shape) == 2 else np.full(param.shape, k)
            values[name] = admissible(param, np.ravel(full, order="F"))
        controller.set(values)
        return k
