"""The control loop: read, guard, step, output stages, write, advance, log."""

from __future__ import annotations

from typing import Any

from ._draw import BLUE, GREY, NAVY, RED, TEAL, arrow, box, canvas, code


def figure() -> Any:
    """One step of ``vmc.sim.run``, the same loop that will drive the hardware."""
    fig, ax = canvas(6.4, 7.4)
    x = 3.0  # the column the loop runs down
    plant = box(ax, x, 6.75, 4.0, 0.95, "plant\n" + code("ModelPlant") + ", or the robot", TEAL)
    guard = box(ax, x, 5.0, 4.0, 0.95, "guard\nzero torque on a bad reading", RED)
    law = box(ax, x, 3.25, 4.0, 0.95, code("VMCController.step") + "\nthe compiled law", NAVY)
    out = box(ax, x, 1.5, 4.0, 0.95, "output stages\nopt-in, for the hardware", BLUE)
    arrow(ax, (x, plant[2]), (x, guard[3]), code("read()") + r"  $\theta,\ \dot\theta$")
    arrow(ax, (x, guard[2]), (x, law[3]))
    arrow(ax, (x, law[2]), (x, out[3]), r"torques $u$")
    # back to the plant on the right, then the plant moves on by one step
    ax.plot([out[1], 5.6, 5.6], [1.5, 1.5, 6.75], color="black", lw=2.0)
    arrow(ax, (5.6, 6.75), (plant[1], 6.75))
    ax.text(5.72, 4.1, code("write(u)"), rotation=90, ha="left", va="center")
    # the plant then moves on by one control period
    arrow(ax, (plant[0], 6.5), (plant[0], 7.0), bend=-1.6)
    ax.text(0.05, 7.25, code("advance(dt)"), ha="left", va="bottom")
    ax.text(x, 0.35, "every step is recorded in a " + code("RunLog"), ha="center", color=GREY)
    return fig
