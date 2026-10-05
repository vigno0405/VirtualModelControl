"""Session: a simulated robot run in real time with its window, until the window is closed."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from ..sim import RunLog, SimClock, run
from .interactive import Interactive
from .window import Window


class Session:
    """A simulation run in real time, in a worker thread, with a ``Window`` in this one.

    ``robot``, ``plant`` and ``controller`` are those of any simulated run, stepped every ``dt``
    [s] at ``speed`` simulated seconds per second. ``name``, ``swaps``, ``goals``, ``sliders`` and
    ``plane`` are those of ``Interactive`` and ``Window``. ``run`` opens the window and returns the
    log when it is closed; ``start`` and ``stop`` do the same without opening it. What was done
    by hand is in ``recorder``.
    """

    def __init__(
        self,
        robot: Any,
        plant: Any,
        controller: Any,
        *,
        dt: float,
        name: str = "ctrl",
        swaps: Mapping[str, Any] | None = None,
        goals: Sequence[str] = (),
        sliders: Mapping[str, tuple[float, float]] | None = None,
        plane: str = "xz",
        speed: float = 1.0,
        z0: Any = None,
        record: Sequence[str] = (),
    ) -> None:
        self.interactive = Interactive(controller, name=name, swaps=swaps, pausable=True)
        self.controls, self.recorder = self.interactive.controls, self.interactive.recorder
        self.window = Window(robot, self.interactive, goals=goals, sliders=sliders, plane=plane)
        self.plant, self.clock, self.z0, self.record = plant, SimClock(dt, speed), z0, record
        self.log: RunLog | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @classmethod
    def from_experiment(cls, experiment: Any, **kwargs: Any) -> Session:
        """A session of a ``vmc.config`` experiment: its robot, plant, controller (with its
        schedule), the controllers it swaps to, its rate and initial state."""
        swaps = {k: c for k, c in experiment.controllers.items() if k != experiment.name}
        return cls(
            experiment.robot,
            experiment.plant,
            experiment.controller,
            dt=1.0 / experiment.settings["rate"],
            name=experiment.name,
            swaps=swaps,
            z0=experiment.z0(),
            **kwargs,
        )

    def start(self) -> None:
        """Start the simulation in a worker thread."""
        self.plant.reset()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def stop(self) -> RunLog:
        """End the simulation and return its log."""
        self.controls.stop()
        if self._thread is not None:
            self._thread.join()
        self.recorder.stop()  # keeps the last values of a recording that was still going
        if self._error is not None:
            raise self._error
        assert self.log is not None
        return self.log

    def run(self) -> RunLog:
        """Open the window and simulate until it is closed; returns the log."""
        self.start()
        try:
            self.window.show()
        finally:
            log = self.stop()
        return log

    def _work(self) -> None:
        try:
            self.log = run(
                self.plant, self.interactive, self.clock, None, z0=self.z0, record=self.record
            )
        except BaseException as error:  # reported by stop, in the thread that asked
            self._error = error
