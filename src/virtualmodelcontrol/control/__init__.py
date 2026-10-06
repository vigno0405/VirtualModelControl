"""Control: controllers built from compiled virtual mechanisms."""

from . import underactuated
from .blending import SwapController, blend_weight
from .controller import VMCController
from .output import FrictionCompensation, Pretension, TorqueLimit, TorqueOffset
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
    "SwapController",
    "Tank",
    "TorqueLimit",
    "TorqueOffset",
    "VMCController",
    "blend_weight",
    "project_psd",
    "underactuated",
]
