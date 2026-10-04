"""A robot's Dynamixel motors as a plant: read angles and rates, write torques, stop safely."""

from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.signals import Signals
from .bus import (
    GOAL_CURRENT,
    GOAL_POSITION,
    OPERATING_MODE,
    OPERATING_MODES,
    PRESENT_POSITION,
    PRESENT_STATE,
    RETURN_DELAY_TIME,
    STATUS_RETURN_LEVEL,
    TORQUE_ENABLE,
    Bus,
    SdkBus,
    decode,
    encode,
)
from .homing import home
from .profile import HardwareProfile


class DynamixelPlant:
    """The motors of ``profile`` as a plant, through the Dynamixel SDK (no ROS).

    ``start`` configures the motors and takes their present positions as zero; ``read`` returns
    the motor angles [rad] and rates [rad/s, smoothed by the profile's ``velocity_filter`` at each
    read]; ``write`` sends motor torques [N·m] as goal currents. A watchdog sends zero torque when
    no command came for ``watchdog`` seconds (None: no watchdog). ``close`` sends zero torque and
    switches the torque motors off; it runs on leaving a ``with`` block (exceptions and Ctrl-C
    included) and at interpreter exit. Held motors keep their start position.
    """

    def __init__(
        self,
        profile: HardwareProfile,
        bus: Bus | None = None,
        *,
        watchdog: float | None = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if any(m.mode != "torque" for m in profile.commanded):
            raise NotImplementedError("DynamixelPlant drives torque-mode motors only")
        self.profile = profile
        self.bus = bus if bus is not None else SdkBus(profile.port, profile.baudrate)
        self.watchdog = watchdog
        self.trips = 0  # zero-torque commands sent by the watchdog
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._zero: np.ndarray = np.zeros(len(profile.ids))
        self._rates: np.ndarray = np.zeros(len(profile.ids))
        self._t0: float | None = None
        self._last_write: float | None = None
        self._tripped = False
        self._closed = False

    @property
    def t(self) -> float:
        """Time since ``start`` [s]."""
        return 0.0 if self._t0 is None else self._clock() - self._t0

    def home(self, **options: Any) -> bool:
        """Before ``start``: drive the motors with a ``home`` in the profile there, slowly (the
        options of ``homing.home``); returns whether they all got there."""
        ticks = {m.id: m.home for m in self.profile.commanded if m.home is not None}
        return home(self.bus, ticks, **options)

    def start(self) -> DynamixelPlant:
        """Configure every motor with a safe goal before its torque goes on; zero at the present
        positions; start the watchdog."""
        with self._lock:
            for m in self.profile.motors:
                self.bus.write(m.id, TORQUE_ENABLE[0], encode(0, TORQUE_ENABLE[1]))
                self.bus.write(m.id, RETURN_DELAY_TIME[0], encode(0, 1))
                self.bus.write(m.id, STATUS_RETURN_LEVEL[0], encode(1, 1))
                self.bus.write(m.id, OPERATING_MODE[0], encode(OPERATING_MODES[m.mode], 1))
                if m.mode == "torque":
                    self.bus.write(m.id, GOAL_CURRENT[0], encode(0, GOAL_CURRENT[1]))
                else:
                    here = self.bus.read(m.id, *PRESENT_POSITION)
                    self.bus.write(m.id, GOAL_POSITION[0], here)
                self.bus.write(m.id, TORQUE_ENABLE[0], encode(1, TORQUE_ENABLE[1]))
            self._zero = self._ticks()
        self._rates[:] = 0.0
        self._t0 = self._clock()
        atexit.register(self.close)
        if self.watchdog is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def _ticks(self) -> np.ndarray:
        data = self.bus.sync_read(self.profile.ids, *PRESENT_POSITION)
        return np.array([decode(data[i]) for i in self.profile.ids], dtype=float)

    def read(self) -> Signals:
        """Motor angles [rad] from the start positions and smoothed motor rates [rad/s]."""
        with self._lock:
            data = self.bus.sync_read(self.profile.ids, *PRESENT_STATE)
        raw = np.array([[decode(data[i][:4]), decode(data[i][4:])] for i in self.profile.ids])
        a = self.profile.velocity_filter
        self._rates = a * self.profile.velocities(raw[:, 0]) + (1.0 - a) * self._rates
        q = self.profile.angles(raw[:, 1], self._zero)
        return Signals(self.t, motor_position=q, motor_velocity=self._rates.copy())

    def write(self, cmd: Signals) -> None:
        """Send ``cmd["motor_torque"]`` [N·m] as goal currents."""
        self._send(self.profile.goal_currents(cmd["motor_torque"]))
        self._last_write, self._tripped = self._clock(), False

    def _send(self, currents: Any) -> None:
        data = {
            i: encode(c, GOAL_CURRENT[1]) for i, c in zip(self.profile.ids, currents, strict=True)
        }
        with self._lock:
            self.bus.sync_write(GOAL_CURRENT[0], data)

    def _watch(self) -> None:
        assert self.watchdog is not None
        while not self._stop.wait(self.watchdog / 4):
            late = self._last_write is not None and self._clock() - self._last_write > self.watchdog
            if late and not self._tripped:
                self._send(np.zeros(len(self.profile.ids), dtype=int))
                self.trips, self._tripped = self.trips + 1, True

    def close(self) -> None:
        """Zero torque, torque off on the torque motors, release the port; safe to call twice."""
        atexit.unregister(self.close)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._closed:
            return
        self._closed = True
        try:
            self._send(np.zeros(len(self.profile.ids), dtype=int))
            with self._lock:
                for m in self.profile.commanded:
                    self.bus.write(m.id, TORQUE_ENABLE[0], encode(0, TORQUE_ENABLE[1]))
        finally:
            self.bus.close()

    def __enter__(self) -> DynamixelPlant:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
