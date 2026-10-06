"""Laws that need no model: they scale or set a Param from the force alone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .limits import admissible, live_matching

NO_FORCE = 1e-6
"""[N] A measured force below this is no contact, and the ratio law leaves the Params alone."""


class ForceRatio:
    """Scales live Params by the wanted force over the measured one, a step at a time.

    A step multiplies each Param by ``f_des / f_meas`` (the norms of the forces), but by at most
    ``max_change`` times the relative error: for a ``max_change`` up to 1 it is that, always.
    """

    def __init__(
        self, controller: Any, params: str | Sequence[str], max_change: float = 0.05
    ) -> None:
        compiled = controller.compiled
        self.names, self.max_change = live_matching(compiled, params), max_change
        self.ratio = 1.0
        """The factor of the last step."""
        self._params = {n: compiled.params[n] for n in self.names}

    def step(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> float:
        """One step on ``controller`` (or a ``Tank`` around it).

        Returns the jump of the controller's energy [J] that was applied; ``ratio`` is the factor
        asked for. Nothing happens while the measured force is below ``NO_FORCE``.
        """
        measured = float(np.linalg.norm(f_meas))
        wanted = float(np.linalg.norm(f_des))
        self.ratio = 1.0
        if measured <= NO_FORCE:
            return 0.0
        change = self.max_change * abs(measured - wanted) / max(measured, wanted)
        self.ratio = float(np.clip(wanted / measured, 1.0 - change, 1.0 + change))
        live = controller.live_params()
        new = {
            n: admissible(p, self.ratio * np.ravel(live[n], order="F"))
            for n, p in self._params.items()
        }
        return float(controller.set(new))


class Stiffening:
    """Sets live Params to a stiffness that grows with the measured force, up to a limit.

    k(F) = low + (high - low) (1 - exp(-alpha F)): ``low`` without force, ``high`` for a large
    one. Every entry of a Param gets k (a square matrix, k times the identity).
    """

    def __init__(
        self,
        controller: Any,
        params: str | Sequence[str],
        low: float,
        high: float,
        alpha: float,
    ) -> None:
        compiled = controller.compiled
        self.names = live_matching(compiled, params)
        self.low, self.high, self.alpha = low, high, alpha
        self.k = low
        """The stiffness of the last step."""
        self._params = {n: compiled.params[n] for n in self.names}

    def stiffness(self, force: float) -> float:
        """k for a measured force [N] along the contact normal; none below zero."""
        return self.low + (self.high - self.low) * (1.0 - np.exp(-self.alpha * max(force, 0.0)))

    def step(self, controller: Any, force: float) -> float:
        """Set the Params on ``controller`` (or a ``Tank`` around it).

        Returns the jump of the controller's energy [J] that was applied; ``k`` is the stiffness
        asked for.
        """
        self.k = float(self.stiffness(force))
        values = {}
        for name, param in self._params.items():
            if len(param.shape) == 2:
                full = self.k * np.eye(param.shape[0])
            else:
                full = np.full(param.shape, self.k)
            values[name] = admissible(param, np.ravel(full, order="F"))
        return float(controller.set(values))
