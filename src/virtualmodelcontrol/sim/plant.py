"""Plants: anything that can be read and commanded, simulated or real."""

from __future__ import annotations

from typing import Any, Protocol

from ..core.signals import Signals


class Plant(Protocol):
    """A robot as the controller sees it: ``read`` measurements, ``write`` commands.

    Hardware and simulators implement the same three methods, so one run loop serves both.
    """

    def read(self) -> Signals:
        """Latest measurements (``motor_position`` [rad], ``motor_velocity`` [rad/s], ...)."""
        ...

    def write(self, cmd: Signals) -> None:
        """Apply a command (``motor_torque`` [N·m])."""
        ...

    def close(self) -> None:
        """Release resources; a real plant stops its motors."""
        ...


class SimPlant(Plant, Protocol):
    """A simulated plant: it also resets and advances its own time ``t`` [s]."""

    t: float

    def reset(self, x0: Any = None, seed: int | None = None) -> None:
        """Back to an initial state."""
        ...

    def advance(self, dt: float) -> None:
        """Integrate over ``dt`` [s] holding the last command."""
        ...
