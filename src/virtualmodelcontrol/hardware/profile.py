"""Hardware profiles: the motors of a robot, their order and signs, and how a driver publishes them.

Everything inside the library is SI, with a positive angle pulling a tendon; a profile converts a
driver's degrees, motor order and signs to that and back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Motor:
    """One motor: bus ``id``, ``sign`` against the library's convention, and ``hold`` for a motor
    the controller does not drive (it stays at its start position)."""

    id: int
    sign: float = 1.0
    hold: bool = False


@dataclass(frozen=True)
class HardwareProfile:
    """The motors of a robot, in the library's motor order.

    ``rate`` is the control rate [Hz]; ``bus_order`` the IDs in the order a driver publishes them,
    when it differs from the motors' order.
    """

    motors: tuple[Motor, ...]
    rate: float = 330.0
    bus_order: tuple[int, ...] | None = None

    @property
    def commanded(self) -> tuple[Motor, ...]:
        """The motors the controller drives, in its motor order."""
        return tuple(m for m in self.motors if not m.hold)

    @property
    def held(self) -> tuple[Motor, ...]:
        """The motors held at their start position."""
        return tuple(m for m in self.motors if m.hold)

    @property
    def ids(self) -> list[int]:
        """IDs of the commanded motors, in the controller's motor order."""
        return [m.id for m in self.commanded]

    def _signs(self) -> np.ndarray:
        return np.array([m.sign for m in self.commanded], dtype=float)

    def from_bus(self, values: Any) -> np.ndarray:
        """A vector in ``bus_order`` (as a driver publishes it) in the motors' order."""
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
        return self._signs() * np.radians(self.from_bus(degrees))

    def degrees(self, angles: Any) -> np.ndarray:
        """Angles or rates [rad, rad/s] as a driver publishes them [deg, bus order, raw signs]."""
        return self.to_bus(np.degrees(self._signs() * np.asarray(angles, dtype=float)))

    def bus_torques(self, torques: Any) -> np.ndarray:
        """Motor torques [N·m] as a driver takes them (bus order, raw signs)."""
        return self.to_bus(self._signs() * np.asarray(torques, dtype=float))

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
        """A copy with some fields changed."""
        return replace(self, **changes)
