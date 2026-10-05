"""Homing: drive the motors slowly to recorded absolute positions, then hand them over in torque
mode with zero current. Deprecated: leaves in 0.4.0."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping

from .bus import (
    GOAL_CURRENT,
    GOAL_POSITION,
    OPERATING_MODE,
    PRESENT_POSITION,
    PROFILE_VELOCITY,
    TORQUE_ENABLE,
    Bus,
    decode,
    encode,
)
from .profile import TICKS_PER_TURN


def present_ticks(bus: Bus, ids: Iterable[int]) -> dict[int, int]:
    """Absolute encoder position of each motor [ticks], for example to record a home pose."""
    return {i: decode(bus.read(i, *PRESENT_POSITION)) for i in ids}


def _set(bus: Bus, ids: Iterable[int], mode: int, field: tuple[int, int], value: int) -> None:
    """Torque off, ``mode``, one register, torque on."""
    for i in ids:
        bus.write(i, TORQUE_ENABLE[0], encode(0, 1))
        bus.write(i, OPERATING_MODE[0], encode(mode, 1))
        bus.write(i, field[0], encode(value, field[1]))
        bus.write(i, TORQUE_ENABLE[0], encode(1, 1))


def home(
    bus: Bus,
    ticks: Mapping[int, int],
    *,
    speed: int = 40,
    threshold: int = 20,
    timeout: float = 15.0,
    max_error: int = TICKS_PER_TURN // 2,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Drive the motors to absolute ``ticks`` (by ID) and return whether all got there.

    The motors move in position mode at ``speed`` [raw units of 0.229 rpm; 40 is about 9 rpm]
    until each is within ``threshold`` ticks, or for at most ``timeout`` [s]; then they return to
    torque mode with zero current. A motor more than ``max_error`` ticks from its home has
    probably lost a turn: nothing moves, and the error names it.
    """
    far = {i: p - ticks[i] for i, p in present_ticks(bus, ticks).items()}
    far = {i: e for i, e in far.items() if abs(e) > max_error}
    if far:
        raise RuntimeError(f"motors more than {max_error} ticks from home (a lost turn?): {far}")
    _set(bus, ticks, 4, PROFILE_VELOCITY, speed)
    for i, t in ticks.items():
        bus.write(i, GOAL_POSITION[0], encode(t, GOAL_POSITION[1]))
    sleep(0.5)
    start, reached = clock(), False
    while not reached and clock() - start < timeout:
        now = present_ticks(bus, ticks)
        reached = all(abs(now[i] - t) <= threshold for i, t in ticks.items())
        if not reached:
            sleep(0.05)
    _set(bus, ticks, 0, GOAL_CURRENT, 0)
    return reached
