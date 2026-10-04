"""Figures and animations: the lab style, drawing robots and virtual elements, saving."""

from .animation import animate
from .draw import (
    draw_damper,
    draw_force,
    draw_frame,
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
    "animate",
    "draw_damper",
    "draw_force",
    "draw_frame",
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
