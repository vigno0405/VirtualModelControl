"""The run loop: one loop for simulated and real plants, with a guard and a recorder."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..core.signals import Signals
from .runlog import RunLog, library_version

MEASUREMENTS = ("motor_position", "motor_velocity")
RECORDS = ("params", "elements", "energy", "robot")
"""What ``run`` can also record at every step: the controller's live Params, each element's
coordinate, rate, force and share of the motor torques, the controller's energies, and, of a
simulated robot, the same of each of its own springs, dampers and contacts (``robot``)."""


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


def run(
    plant: Any,
    controller: Any,
    clock: SimClock | WallClock,
    T: float | None,
    guard: Guard | None = None,
    z0: Any = None,
    record: Iterable[str] | str = (),
) -> RunLog:
    """Run ``controller`` on ``plant`` for ``T`` [s]; returns the recorded run.

    Each step reads the plant, asks the controller for a command (zero torque if the guard
    trips), writes it, then advances a simulated plant by ``clock.dt`` or, with a ``WallClock``,
    waits for the next step. On real time, ``T=None`` runs until Ctrl-C, and Ctrl-C ends the run
    with the log so far. ``z0`` sets the controller's initial
    virtual state, or is a function of the first reading that returns it. The log holds, at every
    step, the measurements, the command (and the law's torque before any output stage), the
    virtual state ``z`` when the controller has one, and what ``record`` asks for (see
    ``RECORDS``); a value the step did not compute is NaN. Its ``meta`` holds the Params at the
    start and the plant's hardware profile.
    """
    guard = Guard() if guard is None else guard
    extras = _extras(record)
    if "robot" in ({record} if isinstance(record, str) else set(record)) and not hasattr(
        plant, "elements"
    ):
        raise ValueError(
            "record='robot' needs a simulated plant: a real robot does not report forces"
        )
    if isinstance(clock, WallClock):
        return _run_wall(plant, controller, clock, T, guard, z0, extras)
    if T is None:
        raise ValueError("a simulated run needs its duration T")
    log = RunLog()
    meas = plant.read()
    motors = _motors(meas)
    _reset(controller, plant.t, meas, z0)
    _describe(log, plant, controller)
    for _ in range(round(T / clock.dt)):
        meas = plant.read()
        good = guard.ok(meas)
        if good:
            cmd = controller.step(plant.t, meas)
        else:
            cmd = Signals(plant.t, motor_torque=np.zeros(motors))
        plant.write(cmd)
        log.step(**_values(plant.t, cmd, meas, controller, plant, extras if good else None))
        plant.advance(clock.dt)
    return log


def _values(
    t: float,
    cmd: Signals,
    meas: Signals,
    controller: Any,
    plant: Any,
    extras: Callable[[Any, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    """One step's record: the time, the command, the measurements, z and the extras."""
    values: dict[str, Any] = {"t": t}
    values.update({name: cmd[name] for name in cmd.names})
    values.update({name: meas[name] for name in meas.names})
    if getattr(controller, "z", None) is not None and np.size(controller.z):
        values["z"] = controller.z
    if extras is not None:
        values.update(extras(controller, plant))
    return values


def _extras(record: Iterable[str] | str) -> Callable[[Any, Any], dict[str, Any]]:
    """What ``record`` asks of a controller and a plant after a step, by log name."""
    record = {record} if isinstance(record, str) else set(record)
    if record - set(RECORDS):
        raise ValueError(f"record takes some of {RECORDS}, got {sorted(record)}")

    def extras(controller: Any, plant: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "params" in record:
            out.update({f"param/{name}": v for name, v in controller.live_params().items()})
        if "elements" in record:
            for element, quantities in controller.elements().items():
                out.update({f"element/{element}/{k}": v for k, v in quantities.items()})
        if "energy" in record:
            balance = controller.balance()
            out["energy/stored"], out["energy/kinetic"] = balance["stored"], balance["kinetic"]
            for name in ("port", "dissipation", "source"):
                out[f"power/{name}"] = balance[name]
        if "robot" in record:
            for element, quantities in plant.elements().items():
                out.update({f"robot/{element}/{k}": v for k, v in quantities.items()})
        return out

    return extras


def _describe(log: RunLog, plant: Any, controller: Any) -> None:
    """What the run is: the library, the start, the Params at the start, the hardware."""
    log.meta.update(
        library=library_version(), start=datetime.now().astimezone().isoformat(timespec="seconds")
    )
    compiled = getattr(controller, "compiled", None)
    if compiled is not None:
        live = controller.live_params()  # the controller's own values, which `set` may have changed
        log.meta["params"] = {
            name: {"value": live.get(name, param.value).tolist(), "unit": param.unit}
            for name, param in compiled.params.items()
        }
    profile = getattr(plant, "profile", None)
    if profile is not None and hasattr(profile, "to_dict"):
        log.meta["hardware"] = profile.to_dict()


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
    plant: Any,
    controller: Any,
    clock: WallClock,
    T: float | None,
    guard: Guard,
    z0: Any,
    extras: Callable[[Any, Any], dict[str, Any]],
) -> RunLog:
    """The run loop on real time: measured steps, stale readings refused, rate statistics."""
    log, limit = RunLog(), None if T is None else round(T / clock.dt)
    meas = plant.read()
    motors = _motors(meas)
    t0 = previous = tick = clock.now()
    _reset(controller, 0.0, meas, z0)
    _describe(log, plant, controller)
    steps = overruns = stale = 0
    try:
        while limit is None or steps < limit:
            now = clock.now()
            t = now - t0
            meas = plant.read()
            old = clock.stale is not None and plant.t - meas.t > clock.stale
            stale += old
            good = guard.ok(meas) and not old
            if good:
                cmd = controller.step(t, meas)
            else:
                cmd = Signals(t, motor_torque=np.zeros(motors))
            plant.write(cmd)
            values = _values(t, cmd, meas, controller, plant, extras if good else None)
            log.step(dt=now - previous, **values)
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
