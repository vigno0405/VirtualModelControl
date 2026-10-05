"""Controls: the changes made by hand, waiting for the controller's next step."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


class Controls:
    """A mailbox between whoever changes a controller by hand (a window, keys, a joystick) and the
    control loop, which may run in another thread.

    ``set`` and ``nudge`` leave new values of live Params and ``swap`` asks for another controller;
    the loop ``take``s them just before its next step. ``values`` holds the latest value of every
    Param, for whoever shows or nudges it. ``paused``, ``recording`` and ``stopped`` are flags.
    """

    def __init__(self, values: Mapping[str, ArrayLike], controllers: Sequence[str] = ()) -> None:
        self._lock = threading.Lock()
        self._values = {name: np.array(value, dtype=float) for name, value in values.items()}
        self._pending: dict[str, np.ndarray] = {}
        self._swaps: list[tuple[str, float]] = []
        self.controllers = list(controllers)
        self.paused = self.recording = self.stopped = False

    @property
    def values(self) -> dict[str, np.ndarray]:
        """The latest value of every live Param (copies)."""
        with self._lock:
            return {name: value.copy() for name, value in self._values.items()}

    def set(self, name: str, value: ArrayLike) -> None:
        """Ask for a new value of the live Param ``name``."""
        with self._lock:
            self._put(name, np.array(value, dtype=float).reshape(self._shape(name)))

    def nudge(self, name: str, delta: ArrayLike) -> None:
        """Ask for the latest value of ``name`` plus ``delta``."""
        with self._lock:
            self._put(name, self._values[self._known(name)] + np.reshape(delta, self._shape(name)))

    def swap(self, name: str, duration: float = 1.0) -> None:
        """Ask for the controller called ``name``, blended in over ``duration`` [s]."""
        if name not in self.controllers:
            raise KeyError(f"no controller named {name!r}; known: {self.controllers}")
        with self._lock:
            self._swaps.append((name, float(duration)))

    def take(self) -> tuple[dict[str, np.ndarray], list[tuple[str, float]]]:
        """The values and swaps asked for since the last call, which they are removed by."""
        with self._lock:
            pending, swaps = self._pending, self._swaps
            self._pending, self._swaps = {}, []
        return pending, swaps

    def toggle_pause(self) -> bool:
        """Pause or resume a simulation; returns the new state."""
        self.paused = not self.paused
        return self.paused

    def toggle_recording(self) -> bool:
        """Start or stop recording what is applied; returns the new state."""
        self.recording = not self.recording
        return self.recording

    def stop(self) -> None:
        """Ask the loop to end, as Ctrl-C does."""
        self.stopped = True

    def _known(self, name: str) -> str:
        if name not in self._values:
            raise KeyError(f"{name!r} is not a live Param here; known: {list(self._values)}")
        return name

    def _shape(self, name: str) -> tuple[int, ...]:
        return self._values[self._known(name)].shape

    def _put(self, name: str, value: np.ndarray) -> None:
        self._values[name] = self._pending[name] = value
