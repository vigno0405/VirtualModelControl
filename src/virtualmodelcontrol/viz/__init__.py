"""Figures: the lab style, drawing robots, springs, goals and forces, saving as PDF and SVG."""

from .draw import (
    draw_force,
    draw_goal,
    draw_point,
    draw_robot,
    draw_spring,
    label_axes,
    project,
    skeleton,
)
from .style import PALETTE, rc, save, style, use_style

__all__ = [
    "PALETTE",
    "draw_force",
    "draw_goal",
    "draw_point",
    "draw_robot",
    "draw_spring",
    "label_axes",
    "project",
    "rc",
    "save",
    "skeleton",
    "style",
    "use_style",
]
