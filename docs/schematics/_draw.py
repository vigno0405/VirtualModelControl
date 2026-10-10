"""Boxes, arrows and labels for the schematics, in the figures' style (inches as data units)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from virtualmodelcontrol.viz import PALETTE

BLUE, RED, TEAL, NAVY = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]
LAVENDER, GREY = PALETTE[5], PALETTE[9]


def canvas(width: float, height: float) -> tuple[Any, Any]:
    """A blank figure whose data units are inches, origin at the bottom left."""
    fig = plt.figure(figsize=(width, height), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = NAVY,
    *,
    dashed: bool = False,
    size: float | None = None,
) -> tuple[float, float, float, float]:
    """A rounded box centered at (x, y) with centered text; returns (left, right, bottom, top)."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=color,
            alpha=0.12,
            edgecolor="none",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(x, y, text, ha="center", va="center", size=size, linespacing=1.3)
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2


def arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
    *,
    color: str = "black",
    bend: float = 0.0,
    side: str = "right",
    dashed: bool = False,
    size: float | None = None,
) -> None:
    """An arrow from start to end, optionally bent and labelled beside its middle."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.0,
            color=color,
            connectionstyle=f"arc3,rad={bend}",
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if text:
        (x0, y0), (x1, y1) = start, end
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if side == "right":
            ax.text(xm + 0.12, ym, text, ha="left", va="center", size=size)
        elif side == "left":
            ax.text(xm - 0.12, ym, text, ha="right", va="center", size=size)
        else:
            ax.text(xm, ym + 0.12, text, ha="center", va="bottom", size=size)


def code(text: str) -> str:
    """Text set in the typewriter face of the figures (for names from the library)."""
    return r"$\mathtt{" + text.replace("_", r"\_").replace(" ", r"\ ") + "}$"
