"""Adaptation: laws that change a running controller's Params to track a goal."""

from .force import ForceTracking
from .scalar import ForceRatio, Stiffening

__all__ = ["ForceRatio", "ForceTracking", "Stiffening"]
