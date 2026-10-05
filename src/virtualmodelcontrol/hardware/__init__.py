"""Hardware profiles: each robot's motors, their IDs, signs and constants.

The Dynamixel plant, homing and the bus tools are deprecated and leave in 0.4.0; using their
names from here warns.
"""

import importlib
import warnings
from typing import Any

from .profile import KT, MODES, HardwareProfile, Motor

_DEPRECATED = {
    "Bus": ".bus",
    "FakeBus": ".bus",
    "SdkBus": ".bus",
    "latency_timer": ".check",
    "scan": ".check",
    "DynamixelPlant": ".dynamixel",
    "home": ".homing",
    "present_ticks": ".homing",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED:
        warnings.warn(
            f"virtualmodelcontrol.hardware.{name} is deprecated and leaves in 0.4.0: the library "
            "is communication-agnostic, so the motors are driven outside it (hardware profiles "
            "stay)",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(_DEPRECATED[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KT", "MODES", "HardwareProfile", "Motor"]
