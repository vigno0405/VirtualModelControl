"""Tank: an energy budget for changing a running controller."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class Tank:
    """A budget of energy [J] for changing a controller while it runs.

    ``set`` applies a change of live Params as far as the tank can pay for the jump it gives the
    controller's energy: the largest fraction of the step whose exact jump is at most ``level``.
    A step that releases energy is always applied whole and refills the tank, up to ``capacity``.
    ``fraction`` is what the last ``set`` applied. Run the tank in place of the controller, and
    ``step`` also refills it with what the controller's own dampers took since the last step;
    everything else is the controller's.
    """

    def __init__(self, controller: Any, level: float = 0.0, capacity: float = np.inf) -> None:
        self.controller, self.level, self.capacity = controller, float(level), float(capacity)
        self.fraction = 1.0
        self._t: float | None = None

    def reset(self, t: float, meas: Any = None, z0: ArrayLike | None = None) -> None:
        """Restart the controller. The tank keeps its level."""
        self.controller.reset(t, meas, z0=z0)
        self._t = None

    def step(self, t: float, meas: Any) -> Any:
        """One step of the controller; its dampers' work since the last step refills the tank."""
        command = self.controller.step(t, meas)
        if self._t is not None:
            taken = -self.controller.balance()["dissipation"] * (t - self._t)  # [J]
            self.level = min(self.level + taken, self.capacity)
        self._t = t
        return command

    def set(self, values: Mapping[str, ArrayLike] | None = None, **kwargs: ArrayLike) -> float:
        """Apply as much of the change as the tank pays for; returns the energy jump applied [J]."""
        goal = {n: np.ravel(v, order="F") for n, v in {**(values or {}), **kwargs}.items()}
        full = self.controller.jump(goal)  # also refuses a Param that is not live
        old = self.controller.live_params()
        start = {n: np.ravel(old[n], order="F") for n in goal}

        def step(a: float) -> dict[str, np.ndarray]:
            return {n: start[n] + a * (goal[n] - start[n]) for n in goal}

        def jump(a: float) -> float:
            return float(self.controller.jump(step(a)))

        alpha = 1.0
        if full > self.level:  # the first fraction of the step that costs more than the tank holds
            grid = np.linspace(0.0, 1.0, 65)
            k = next(i for i, a in enumerate(grid) if jump(a) > self.level)
            lo, hi = grid[k - 1], grid[k]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if jump(mid) <= self.level else (lo, mid)
            alpha = float(lo)
        applied = float(self.controller.set(step(alpha)))
        self.fraction = alpha
        self.level = min(self.level - applied, self.capacity)
        return applied

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)
