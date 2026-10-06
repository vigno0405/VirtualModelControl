"""Adaptation: laws that change a running controller's Params to track a goal."""

from .force import ForceTracking
from .position import PositionRegulation
from .scalar import ForceRatio, Stiffening
from .stiffness import StiffnessTracking

__all__ = ["ForceRatio", "ForceTracking", "PositionRegulation", "Stiffening", "StiffnessTracking"]
