"""Interactive: a controller that applies the changes made by hand just before each step."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..control import SwapController
from ..core.signals import Signals
from .controls import Controls
from .recorder import Recorder


class Interactive:
    """Wraps ``controller`` so that what ``controls`` holds is applied before each step. Run it
    like any controller, in a simulation or inside a robot's own node.

    A value goes to every controller that has the Param live: this one, called ``name``, and
    those in ``swaps`` (names to controllers, which ``controls.swap`` may ask for). While
    ``controls.recording``, ``recorder`` notes what is applied. ``controls.stopped`` ends the run
    as Ctrl-C does, and with ``pausable`` (a simulation) ``controls.paused`` holds it. ``last`` is
    the latest (t, measurement) for displays, and ``injected`` the energy [J] that the changes
    gave the running controller, as ``set`` computes it. Everything else is the controller's.
    """

    def __init__(
        self,
        controller: Any,
        *,
        name: str = "ctrl",
        swaps: Mapping[str, Any] | None = None,
        controls: Controls | None = None,
        recorder: Recorder | None = None,
        pausable: bool = False,
    ) -> None:
        swaps = dict(swaps or {})
        if not hasattr(controller, "swap"):
            controller = SwapController(controller)
        self.controller = controller
        self._owners = {name: _running(controller), **swaps}
        if controls is None:
            values: dict[str, np.ndarray] = {}
            for owner in self._owners.values():
                values.update(owner.live_params())
            controls = Controls(values, list(self._owners))
        self.controls, self.recorder, self.pausable = controls, recorder or Recorder(), pausable
        self.selected = name
        self.injected = 0.0
        self.last: tuple[float, Signals] | None = None
        self._applied = controls.values
        self._t0: float | None = None
        self._recording = False

    def unit(self, name: str) -> str:
        """The unit of the live Param ``name``."""
        owner = next(o for o in self._owners.values() if name in o.compiled.live)
        return str(owner.compiled.params[name].unit)

    def reset(self, t: float, meas: Signals | None = None, z0: Any = None) -> None:
        """Start again at time ``t`` [s]."""
        self._t0, self.injected, self.last = t, 0.0, None
        self.controller.reset(t, meas, z0)

    def step(self, t: float, meas: Signals) -> Signals:
        """Apply the changes asked for, then one control step of the controller."""
        controls = self.controls
        while self.pausable and controls.paused and not controls.stopped:
            time.sleep(0.01)
        if controls.stopped:
            raise KeyboardInterrupt
        if self._t0 is None:
            self._t0 = t
        values, swaps = controls.take()
        self._note(t - self._t0, values, swaps)
        for target, duration in swaps:
            self.controller.swap(self._owners[target], duration)
            self.selected = target
        running = _running(self.controller)
        for name, value in values.items():
            for owner in self._owners.values():
                if name in owner.compiled.live:
                    jump = owner.set({name: value})
                    self.injected += jump if owner is running else 0.0
        self._applied.update(values)
        out = self.controller.step(t, meas)
        self.last = (t, meas)
        return out

    def _note(self, t: float, values: Mapping[str, np.ndarray], swaps: list[Any]) -> None:
        """Tell the recorder what is about to be applied, and when recording starts or stops."""
        recording = self.controls.recording
        if recording and not self._recording:
            self.recorder.start(t, self._applied)
        elif self._recording and not recording:
            self.recorder.stop()
        self._recording = recording
        if recording and (values or swaps):
            self.recorder.applied(t, values, swaps)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)


def _running(controller: Any) -> Any:
    """The controller that runs inside any wrappers (a swap controller, a schedule)."""
    while hasattr(controller, "controller"):
        controller = controller.controller
    return controller
