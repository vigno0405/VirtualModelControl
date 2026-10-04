"""Energy flow: what each mechanism stores, dissipates and is given, and the port between them."""

from __future__ import annotations

from typing import Any

from ._draw import BLUE, GREY, NAVY, RED, TEAL, arrow, box, canvas


def figure() -> Any:
    """The controller above the robot. Arrows point the way energy flows when the term is
    positive; dissipation is never positive, so it always leaves."""
    fig, ax = canvas(6.4, 8.9)
    x, w, h = 1.85, 3.3, 1.45  # the column of the two mechanisms [in]
    ctrl = box(ax, x, 6.6, w, h, "controller, $E = V + T$\nsprings $V$,\nvirtual masses $T$", NAVY)
    robot = box(ax, x, 3.45, w, h, "robot, $E = T + V$\nmasses $T$,\nsprings and gravity $V$", TEAL)

    # sources give energy to a mechanism (or take it, when negative)
    ax.text(x, 8.8, "force sources,\ngravity compensation", ha="center", va="top")
    arrow(ax, (x, 8.05), (x, ctrl[3]), "source", color=RED)
    arrow(ax, (x, 1.95), (x, robot[2]), "source", color=GREY, dashed=True)
    ax.text(x, 1.85, "external forces", ha="center", va="top", color=GREY)

    # the motors carry power from the controller to the robot
    arrow(ax, (x, ctrl[2]), (x, robot[3]), color=BLUE)
    mid = (ctrl[2] + robot[3]) / 2
    ax.text(x + 0.15, mid + 0.22, r"port $= \tau^{\top} v$", va="center")
    ax.text(x + 0.15, mid - 0.22, "= input", va="center")
    ax.text(x - 0.15, mid, "motor\ntorques", ha="right", va="center", color=GREY)

    # dampers take energy out of each mechanism
    for (_, right, low, high), sink in ((ctrl, "dampers"), (robot, "own\ndamping")):
        y = (low + high) / 2
        arrow(ax, (right, y), (right + 1.45, y), color=GREY)
        ax.text(right + 0.72, y + 0.1, "dissipation", ha="center", va="bottom")
        ax.text(right + 0.72, y - 0.1, r"$\leq 0$", ha="center", va="top")
        ax.text(right + 1.55, y, sink, va="center")

    # the balances the compiled functions satisfy
    ax.text(0.15, 0.95, r"controller:  $\dot E = -$port $+$ dissipation $+$ source", va="center")
    ax.text(0.15, 0.35, r"robot:  $\dot E = $input $+$ dissipation $+$ source", va="center")
    return fig
