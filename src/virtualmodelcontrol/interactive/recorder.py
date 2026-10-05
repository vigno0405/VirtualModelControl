"""Recorder: what was changed by hand, as the schedule entries of a configuration."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class Recorder:
    """Notes the values and swaps applied between ``start`` and ``stop``, with times counted from
    ``start``, and turns them into the entries of an experiment's ``schedule``.

    A value is held until the next one, so the entries use ``step`` interpolation: replayed, they
    apply every value at the step it was applied at. Of values arriving closer than ``min_dt`` [s],
    only the last is kept. Each entry starts from the value the Param had when recording began.
    """

    def __init__(self, min_dt: float = 0.02) -> None:
        self.min_dt = min_dt
        self._baseline: dict[str, np.ndarray] = {}
        self._points: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._pending: dict[str, tuple[float, np.ndarray]] = {}
        self._swaps: list[tuple[str, float, float]] = []
        self._t0 = 0.0

    def start(self, t: float, values: Mapping[str, np.ndarray]) -> None:
        """Begin at time ``t`` [s], with the Params' values at this moment."""
        self._t0 = t
        self._baseline = {name: np.array(value, dtype=float) for name, value in values.items()}
        self._points, self._pending, self._swaps = {}, {}, []

    def applied(
        self, t: float, values: Mapping[str, np.ndarray], swaps: list[tuple[str, float]]
    ) -> None:
        """The values and swaps applied at time ``t`` [s]."""
        elapsed = t - self._t0
        for name, value in values.items():
            points = self._points.setdefault(name, [])
            if not points or elapsed - points[-1][0] >= self.min_dt:
                points.append((elapsed, np.array(value, dtype=float)))
                self._pending.pop(name, None)
            else:
                self._pending[name] = (elapsed, np.array(value, dtype=float))
        self._swaps += [(name, elapsed, duration) for name, duration in swaps]

    def stop(self) -> None:
        """End the recording: the last value of each Param is kept even if it was close to the one
        before."""
        for name, point in self._pending.items():
            self._points[name].append(point)
        self._pending = {}

    def schedule(self) -> list[dict[str, Any]]:
        """The recorded changes as ``schedule`` entries: one per Param, one per swap."""
        entries: list[dict[str, Any]] = []
        for name, points in self._points.items():
            first = [] if points[0][0] <= 0.0 else [(0.0, self._baseline[name])]
            entries.append(
                {
                    "param": name,
                    "points": [[t, np.asarray(v).tolist()] for t, v in first + points],
                    "interpolation": "step",
                }
            )
        entries += [{"swap": n, "at": at, "duration": d} for n, at, d in self._swaps]
        return entries

    def save(self, configuration: Any, path: str | os.PathLike[str]) -> Path:
        """Write ``configuration`` (an Experiment, a YAML file or a dict) to ``path`` with the
        recorded entries added to its ``experiment.schedule``; running that file repeats the
        session."""
        from ..config import files

        if hasattr(configuration, "to_dict"):
            spec = configuration.to_dict()
        elif isinstance(configuration, Mapping):
            spec = copy.deepcopy(dict(configuration))
        else:
            spec = files.read(configuration)
        settings = spec.setdefault("experiment", {})
        settings["schedule"] = [*(settings.get("schedule") or []), *self.schedule()]
        return files.write(spec, path)
