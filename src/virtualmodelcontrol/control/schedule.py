"""Schedules: live Params that follow given values over time, and swaps at given times.

A schedule holds points (t, value) of one live Param. With ``linear`` interpolation the value
moves straight from point to point (a ramp); with ``step`` it jumps at each point. Before the
first point the Param keeps its own value, after the last it holds the last one. Times count
from the controller's reset, the start of a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.signals import Signals
from .blending import SwapController

INTERPOLATIONS = ("linear", "step")


class Schedule:
    """Values of the live Param ``param`` over time: ``points`` [(t [s], value), ...] with
    ``linear`` or ``step`` ``interpolation``."""

    def __init__(
        self,
        param: str,
        points: Sequence[tuple[float, ArrayLike]],
        interpolation: str = "linear",
    ) -> None:
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}"
            )
        if not points:
            raise ValueError(f"the schedule of {param!r} needs at least one point")
        self.param, self.interpolation = param, interpolation
        self.times = np.array([float(t) for t, _ in points])
        if np.any(np.diff(self.times) < 0.0):
            raise ValueError(f"the times of the schedule of {param!r} must not decrease")
        self.values = np.array([np.asarray(v, dtype=float) for _, v in points])

    def value(self, t: float) -> np.ndarray | None:
        """The Param's value at ``t`` [s] from the start; None before the first point."""
        if t < self.times[0]:
            return None
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        if k == len(self.times) - 1 or self.interpolation == "step":
            return self.values[k]
        w = (t - self.times[k]) / (self.times[k + 1] - self.times[k])
        return (1.0 - w) * self.values[k] + w * self.values[k + 1]


class ScheduledController:
    """A controller whose live Params follow schedules, and which swaps at given times.

    Each schedule sets its Param on every controller that has it live: the one running and those
    in ``swaps``, a list of (time [s], controller, duration [s]); a swap blends to its controller
    as ``SwapController.swap`` does. ``reset`` starts again: the first controller runs and the
    scheduled Params are back at their own values. Everything else is the running controller's.
    """

    def __init__(
        self,
        controller: Any,
        schedules: Iterable[Schedule] = (),
        swaps: Iterable[tuple[float, Any, float]] = (),
    ) -> None:
        self.swaps = sorted(swaps, key=lambda swap: swap[0])
        if self.swaps and not isinstance(controller, SwapController):
            controller = SwapController(controller)
        self.controller = controller
        first = controller.controller if isinstance(controller, SwapController) else controller
        self._first = first
        pool = [first, *(c for _, c, _ in self.swaps if c is not first)]
        self.schedules = []
        for schedule in schedules:
            owners = [c for c in pool if schedule.param in c.compiled.live]
            if not owners:
                raise KeyError(
                    f"{schedule.param!r} is not a live Param of these controllers; compile with "
                    f"runtime=[{schedule.param!r}] to schedule it"
                )
            self.schedules.append((schedule, owners))
        self._t0: float | None = None
        self._sent: dict[int, np.ndarray] = {}
        self._next = 0  # the first swap not yet started

    def reset(self, t: float, meas: Signals | None = None, z0: ArrayLike | None = None) -> None:
        """Start again at time ``t`` [s]: the first controller, the scheduled Params at their own
        values, the schedules and swaps from their beginning."""
        if self.swaps:
            self.controller = SwapController(self._first)
        for schedule, owners in self.schedules:
            for owner in owners:
                owner.set({schedule.param: owner.compiled.params[schedule.param].value})
        self._t0, self._sent, self._next = t, {}, 0
        self.controller.reset(t, meas, z0)

    def step(self, t: float, meas: Signals) -> Signals:
        """Set the scheduled values that changed, start the swaps that are due, then step."""
        if self._t0 is None:
            self._t0 = t
        elapsed = t - self._t0
        changes: dict[int, tuple[Any, dict[str, np.ndarray]]] = {}
        for k, (schedule, owners) in enumerate(self.schedules):
            value = schedule.value(elapsed)
            if value is None or (k in self._sent and np.array_equal(self._sent[k], value)):
                continue
            self._sent[k] = value
            for owner in owners:
                changes.setdefault(id(owner), (owner, {}))[1][schedule.param] = value
        for owner, values in changes.values():
            owner.set(values)
        while self._next < len(self.swaps) and self.swaps[self._next][0] <= elapsed:
            _, target, duration = self.swaps[self._next]
            self.controller.swap(target, duration)
            self._next += 1
        return self.controller.step(t, meas)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)
