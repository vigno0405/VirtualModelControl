"""Hardware profiles: the motors of a robot, their order, signs and units, and the bus settings.

Raw units are those of Dynamixel X-series motors: encoder ticks, velocity units of 0.229 rpm and
goal currents, which a motor constant kt converts to torque. Everything else is SI, in the
library's convention (a positive angle pulls a tendon).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

TICKS_PER_TURN = 4096
"""Encoder ticks per motor turn."""

VELOCITY_UNIT = 0.229 * 2.0 * np.pi / 60.0
"""One raw velocity unit [rad/s] (0.229 rpm)."""

CURRENT_RANGE = 32767
"""Largest goal current the bus accepts, a 2-byte signed value [raw units]."""

KT = {"XL330-M288": 0.000354, "XC330-T288": 0.00115, "XM430-W350": 0.001783}
"""Motor constant of each Dynamixel model [N·m per unit of goal current]."""

MODES = ("torque", "velocity", "position", "hold")
"""Operating modes; ``hold`` keeps the motor at its start position, out of the controller's
reach (a wrist, a stiffness motor)."""

_RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_TURN


@dataclass(frozen=True)
class Motor:
    """One motor: bus ``id``, Dynamixel ``model``, ``sign`` against the library's convention,
    operating ``mode``, motor constant ``kt`` [N·m per unit of goal current; the model's when
    None] and an optional ``torque_limit`` [N·m]."""

    id: int
    model: str = "XC330-T288"
    sign: float = 1.0
    mode: str = "torque"
    kt: float | None = None
    torque_limit: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"motor {self.id}: mode {self.mode!r} is not one of {MODES}")

    @property
    def constant(self) -> float:
        """kt [N·m per unit of goal current]."""
        return KT[self.model] if self.kt is None else float(self.kt)


@dataclass(frozen=True)
class HardwareProfile:
    """The motors of a robot, in the library's motor order, and how to reach them.

    ``baudrate`` [bit/s], control ``rate`` [Hz], serial ``port``, the smoothing factor of measured
    velocities ``velocity_filter`` (1 keeps them raw), and ``bus_order``: the IDs in the order a
    ROS driver publishes them, when it differs from the motors' order.
    """

    motors: tuple[Motor, ...]
    baudrate: int = 1_000_000
    rate: float = 330.0
    port: str = "/dev/ttyUSB0"
    velocity_filter: float = 0.1
    bus_order: tuple[int, ...] | None = None

    @property
    def commanded(self) -> tuple[Motor, ...]:
        """The motors the controller drives, in its motor order."""
        return tuple(m for m in self.motors if m.mode != "hold")

    @property
    def held(self) -> tuple[Motor, ...]:
        """The motors held at their start position."""
        return tuple(m for m in self.motors if m.mode == "hold")

    @property
    def ids(self) -> list[int]:
        """IDs of the commanded motors, in the controller's motor order."""
        return [m.id for m in self.commanded]

    def _column(self, name: str) -> np.ndarray:
        return np.array([getattr(m, name) for m in self.commanded], dtype=float)

    def angles(self, ticks: Any, start: Any) -> np.ndarray:
        """Motor angles [rad] from encoder ticks, zero at the ``start`` ticks."""
        delta = np.asarray(ticks, dtype=float) - np.asarray(start, dtype=float)
        return self._column("sign") * delta * _RAD_PER_TICK

    def goal_ticks(self, angles: Any, start: Any) -> np.ndarray:
        """Goal positions [ticks] for motor angles [rad]: the inverse of ``angles``, rounded."""
        delta = self._column("sign") * np.asarray(angles, dtype=float) / _RAD_PER_TICK
        return np.rint(np.asarray(start, dtype=float) + delta).astype(np.int64)

    def velocities(self, raw: Any) -> np.ndarray:
        """Motor rates [rad/s] from raw velocities."""
        return self._column("sign") * np.asarray(raw, dtype=float) * VELOCITY_UNIT

    def goal_velocities(self, rates: Any) -> np.ndarray:
        """Raw goal velocities for motor rates [rad/s], rounded."""
        raw = self._column("sign") * np.asarray(rates, dtype=float) / VELOCITY_UNIT
        return _integers(raw, np.rint)

    def goal_currents(self, torques: Any) -> np.ndarray:
        """Goal currents for motor torques [N·m]: sign τ / kt.

        Each torque is clipped to its motor's ``torque_limit`` and the current to the bus range
        before the conversion to integers (which truncates towards zero), so a large command
        saturates instead of overflowing or flipping sign; a non-finite torque gives zero.
        """
        limit = np.array(
            [np.inf if m.torque_limit is None else m.torque_limit for m in self.commanded]
        )
        tau = np.clip(self._column("sign") * np.asarray(torques, dtype=float), -limit, limit)
        return _integers(tau / np.array([m.constant for m in self.commanded]))

    def torques(self, currents: Any) -> np.ndarray:
        """Motor torques [N·m] from raw currents."""
        kt = np.array([m.constant for m in self.commanded])
        return self._column("sign") * np.asarray(currents, dtype=float) * kt

    def from_bus(self, values: Any) -> np.ndarray:
        """A vector in ``bus_order`` (as a ROS driver publishes it) in the motors' order."""
        values = np.asarray(values, dtype=float)
        if self.bus_order is None:
            return values
        return values[[list(self.bus_order).index(i) for i in self.ids]]

    def to_bus(self, values: Any) -> np.ndarray:
        """A vector in the motors' order in ``bus_order``: the inverse of ``from_bus``."""
        values = np.asarray(values, dtype=float)
        if self.bus_order is None:
            return values
        return values[[self.ids.index(i) for i in self.bus_order]]

    def angles_from_degrees(self, degrees: Any) -> np.ndarray:
        """Motor angles [rad] from a driver's published angles or rates [deg, deg/s, bus order]."""
        return self._column("sign") * np.radians(self.from_bus(degrees))

    def degrees(self, angles: Any) -> np.ndarray:
        """Angles or rates [rad, rad/s] as a driver publishes them [deg, bus order, raw signs]."""
        return self.to_bus(np.degrees(self._column("sign") * np.asarray(angles, dtype=float)))

    def bus_torques(self, torques: Any) -> np.ndarray:
        """Motor torques [N·m] as a driver takes them on its torque topic (bus order, raw signs)."""
        return self.to_bus(self._column("sign") * np.asarray(torques, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        """Plain values, for YAML or JSON."""
        data = asdict(self)
        data["motors"] = [asdict(m) for m in self.motors]
        data["bus_order"] = None if self.bus_order is None else list(self.bus_order)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        """Inverse of ``to_dict``."""
        motors = tuple(Motor(**m) for m in data["motors"])
        order = data.get("bus_order")
        rest = {k: v for k, v in data.items() if k not in ("motors", "bus_order")}
        return cls(motors, bus_order=None if order is None else tuple(order), **rest)

    def save(self, path: str | Path) -> Path:
        """Write the profile as YAML; returns the path."""
        import yaml

        path = Path(path)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HardwareProfile:
        """Read a profile written by ``save``."""
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def replace(self, **changes: Any) -> HardwareProfile:
        """A copy with some fields changed, for example another ``port``."""
        return replace(self, **changes)


def _integers(raw: np.ndarray, rounding: Any = np.trunc) -> np.ndarray:
    """Zero for non-finite values, clip to the bus range, then round (default: towards zero)."""
    raw = np.where(np.isfinite(raw), raw, 0.0)
    return rounding(np.clip(raw, -CURRENT_RANGE, CURRENT_RANGE)).astype(np.int64)
