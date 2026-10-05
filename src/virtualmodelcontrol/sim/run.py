"""The run loop: one loop for simulated and real plants, with a guard and a recorder."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
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
class WallClock:
    """Real time: a step every ``dt`` [s] of the computer's clock. With ``stale`` [s], a reading
    older than that (the plant's ``t`` minus the reading's) sends zero torque. ``now`` and
    ``sleep`` are the clock's own (replaceable in tests)."""

    dt: float
    stale: float | None = None
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


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
    info: dict[str, float] = field(default_factory=dict)  # statistics of a real-time run

    def append(self, **values: Any) -> None:
        """Add one row of named values."""
        for name, value in values.items():
            # A copy: plants and controllers may update their arrays in place.
            self.rows.setdefault(name, []).append(np.array(value, dtype=float, ndmin=1))

    def arrays(self) -> dict[str, np.ndarray]:
        """Each signal as an (n_steps, ...) array."""
        return {name: np.array(rows) for name, rows in self.rows.items()}


def run(
    plant: Any,
    controller: Any,
    clock: SimClock | WallClock,
    T: float | None,
    guard: Guard | None = None,
    z0: Any = None,
) -> RunLog:
    """Run ``controller`` on ``plant`` for ``T`` [s]; returns the recorded run.

    Each step reads the plant, asks the controller for a command (zero torque if the guard
    trips), writes it, then advances a simulated plant by ``clock.dt`` or, with a ``WallClock``,
    waits for the next step. On real time, ``T=None`` runs until Ctrl-C, and Ctrl-C ends any run
    with the log so far. ``z0`` sets the controller's initial virtual state, or is a function of
    the first reading that returns it; the log also records that state, ``z``, when the
    controller has one.
    """
    guard = Guard() if guard is None else guard
    if isinstance(clock, WallClock):
        return _run_wall(plant, controller, clock, T, guard, z0)
    if T is None:
        raise ValueError("a simulated run needs its duration T")
    log = RunLog()
    meas = plant.read()
    motors = _motors(meas)
    _reset(controller, plant.t, meas, z0)
    for _ in range(round(T / clock.dt)):
        meas = plant.read()
        if guard.ok(meas):
            cmd = controller.step(plant.t, meas)
        else:
            cmd = Signals(plant.t, motor_torque=np.zeros(motors))
        plant.write(cmd)
        log.append(t=plant.t, motor_torque=cmd["motor_torque"], **{n: meas[n] for n in meas.names})
        if getattr(controller, "z", None) is not None and np.size(controller.z):
            log.append(z=controller.z)
        plant.advance(clock.dt)
    return log


def _motors(meas: Signals) -> int:
    """Number of motor torques: one per motor rate of a reading."""
    return int(np.size(meas["motor_velocity"]))


def _reset(controller: Any, t: float, meas: Signals, z0: Any) -> None:
    if callable(z0):
        z0 = z0(meas)
    if z0 is None:
        controller.reset(t, meas)
    else:
        controller.reset(t, meas, z0=z0)


def _run_wall(
    plant: Any, controller: Any, clock: WallClock, T: float | None, guard: Guard, z0: Any
) -> RunLog:
    """The run loop on real time: measured steps, stale readings refused, rate statistics."""
    log, limit = RunLog(), None if T is None else round(T / clock.dt)
    meas = plant.read()
    motors = _motors(meas)
    t0 = previous = tick = clock.now()
    _reset(controller, 0.0, meas, z0)
    steps = overruns = stale = 0
    try:
        while limit is None or steps < limit:
            now = clock.now()
            t = now - t0
            meas = plant.read()
            old = clock.stale is not None and plant.t - meas.t > clock.stale
            stale += old
            if guard.ok(meas) and not old:
                cmd = controller.step(t, meas)
            else:
                cmd = Signals(t, motor_torque=np.zeros(motors))
            plant.write(cmd)
            log.append(t=t, dt=now - previous, motor_torque=cmd["motor_torque"])
            log.append(**{n: meas[n] for n in meas.names})
            if getattr(controller, "z", None) is not None and np.size(controller.z):
                log.append(z=controller.z)
            steps += 1
            previous, tick = now, tick + clock.dt
            wait = tick - clock.now()
            if wait > 0:
                clock.sleep(wait)
            else:  # late: start again from now rather than catching up in a burst
                overruns, tick = overruns + 1, clock.now()
    except KeyboardInterrupt:  # Ctrl-C ends a real-time run with the log so far
        log.rows = {name: rows[:steps] for name, rows in log.rows.items()}
    dt = np.diff(log.arrays()["t"].ravel()) if steps > 1 else np.array([clock.dt])
    log.info = {
        "steps": steps,
        "rate": float(1.0 / dt.mean()),
        "dt_max": float(dt.max()),
        "overruns": overruns,
        "stale": stale,
        "guard_trips": guard.trips,
    }
    if overruns:
        warnings.warn(
            f"{overruns} of {steps} steps took longer than {1000 * clock.dt:.1f} ms "
            f"(slowest {1000 * dt.max():.1f} ms)",
            RuntimeWarning,
            stacklevel=3,
        )
    return log
