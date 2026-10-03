"""Control: controllers built from compiled virtual mechanisms."""

from .controller import VMCController
from .output import (
    EfficiencyCorrection,
    FrictionCompensation,
    Pretension,
    TorqueLimit,
    TorqueOffset,
)

__all__ = [
    "EfficiencyCorrection",
    "FrictionCompensation",
    "Pretension",
    "TorqueLimit",
    "TorqueOffset",
    "VMCController",
]
