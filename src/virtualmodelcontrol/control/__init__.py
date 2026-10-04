"""Control: controllers built from compiled virtual mechanisms."""

from .controller import VMCController
from .output import FrictionCompensation, Pretension, TorqueLimit, TorqueOffset

__all__ = [
    "FrictionCompensation",
    "Pretension",
    "TorqueLimit",
    "TorqueOffset",
    "VMCController",
]
