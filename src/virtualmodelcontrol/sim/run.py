"""The run loop: one loop for simulated and real plants, with a guard and a recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.signals import Signals

MEASUREMENTS = ("motor_position", "motor_velocity")


@dataclass
class SimClock:
    """Simulated time: each step advances the plant by ``dt`` [s]."""

    dt: float


@dataclass
class Guard:
    """Sends zero torque when a measurement is missing, flagged invalid, NaN or Inf."""

    required: tuple[str, ...] = MEASUREMENTS
    trips: int = 0

    def ok(self, meas: Signals) -> bool:
        """True if every required measurement is valid."""
        good = all(meas.is_valid(name) for name in self.required)
        self.trips += not good
        return good


@dataclass
class RunLog:
    """Recorded signals of a run, one row per step; ``arrays`` stacks them."""

    rows: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def append(self, **values: Any) -> None:
        """Add one row of named values."""
        for name, value in values.items():
            self.rows.setdefault(name, []).append(np.atleast_1d(np.asarray(value, dtype=float)))

    def arrays(self) -> dict[str, np.ndarray]:
        """Each signal as an (n_steps, ...) array."""
        return {name: np.array(rows) for name, rows in self.rows.items()}


def run(
    plant: Any, controller: Any, clock: SimClock, T: float, guard: Guard | None = None
) -> RunLog:
    """Run ``controller`` on a simulated ``plant`` for ``T`` [s]; returns the recorded run.

    Each step reads the plant, asks the controller for a command (zero torque if the guard
    trips), writes it and advances the plant by ``clock.dt``.
    """
    guard = Guard() if guard is None else guard
    log = RunLog()
    meas = plant.read()
    controller.reset(plant.t, meas)
    for _ in range(round(T / clock.dt)):
        meas = plant.read()
        if guard.ok(meas):
            cmd = controller.step(plant.t, meas)
        else:
            cmd = Signals(plant.t, motor_torque=np.zeros_like(plant.u))
        plant.write(cmd)
        log.append(t=plant.t, motor_torque=cmd["motor_torque"], **{n: meas[n] for n in meas.names})
        plant.advance(clock.dt)
    return log
