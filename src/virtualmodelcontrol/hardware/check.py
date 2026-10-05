"""Bus checks: which motors answer at which baud rate, and the USB latency timer on Linux.
Deprecated: leaves in 0.4.0."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .bus import MODEL_NUMBERS, Bus, SdkBus

BAUDRATES = (57_600, 1_000_000, 2_000_000, 3_000_000, 4_000_000)
"""Baud rates to try [bit/s]: the factory default, then those the lab's robots use."""


def scan(
    port: str = "/dev/ttyUSB0",
    baudrates: Iterable[int] = BAUDRATES,
    bus: Callable[[str, int], Bus] = SdkBus,
) -> dict[int, dict[int, str]]:
    """The motors that answer a broadcast ping at each baud rate: ``{baudrate: {id: model}}``."""
    found = {}
    for baudrate in baudrates:
        b = bus(port, baudrate)
        try:
            motors = b.ping_all()
        finally:
            b.close()
        if motors:
            found[baudrate] = {i: MODEL_NUMBERS.get(n, str(n)) for i, n in sorted(motors.items())}
    return found


def latency_timer(port: str = "/dev/ttyUSB0") -> int | None:
    """The USB serial adapter's latency timer [ms] of ``port``, on Linux; None elsewhere.

    1 ms keeps the loop fast; the default, 16 ms, can stretch every bus transaction to that
    long. To set it (as root):
    ``echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer``.
    """
    path = Path("/sys/bus/usb-serial/devices") / Path(port).name / "latency_timer"
    return int(path.read_text()) if path.exists() else None
