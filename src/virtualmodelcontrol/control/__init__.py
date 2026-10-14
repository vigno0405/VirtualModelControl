"""Control: controllers built from compiled virtual mechanisms."""

from .blending import SwapController, blend_weight
from .controller import VMCController
from .output import (
    FrictionCompensation,
    Pretension,
    StaticFrictionCompensation,
    TorqueLimit,
    TorqueOffset,
)
from .projection import project_psd
from .schedule import Schedule, ScheduledController
from .state import StateController
from .tank import Tank

__all__ = [
    "FrictionCompensation",
    "Pretension",
    "Schedule",
    "ScheduledController",
    "StateController",
    "StaticFrictionCompensation",
    "SwapController",
    "Tank",
    "TorqueLimit",
    "TorqueOffset",
    "VMCController",
    "blend_weight",
    "project_psd",
]
