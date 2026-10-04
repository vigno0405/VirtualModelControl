"""The four steps of using the library: robot, virtual elements, compile, run on a plant."""

from __future__ import annotations

from typing import Any

from ._draw import BLUE, GREY, LAVENDER, NAVY, TEAL, arrow, box, canvas, code

STEPS = [  # title, what it holds or does, the library's name for it, colour
    ("1. describe the robot", "kinematics, motors, masses", "robots.helyx.arm()", TEAL),
    (
        "2. place the virtual elements",
        "springs, dampers, sources",
        "ctrl.add(name, component)",
        NAVY,
    ),
    ("3. compile", "one function, angles to torques", "compile(system)", LAVENDER),
    (
        "4. run on a plant",
        "a simulator or the real robot",
        "sim.run(plant, controller, clock, T)",
        BLUE,
    ),
]


def figure() -> Any:
    """Four boxes down the page, and the simulator built from the robot alone (left)."""
    fig, ax = canvas(6.4, 7.6)
    x, w, h = 3.75, 5.1, 1.2  # the column of boxes [in]
    ys = [6.85, 4.95, 3.05, 1.15]
    boxes = [
        box(ax, x, y, w, h, f"{title}\n{text}\n{code(name)}", color)
        for (title, text, name, color), y in zip(STEPS, ys, strict=True)
    ]
    labels = ["its coordinates", code("VirtualMechanismSystem"), code("VMCController")]
    for upper, lower, label in zip(boxes[:-1], boxes[1:], labels, strict=True):
        arrow(ax, (x - 1.3, upper[2]), (x - 1.3, lower[3]), label)
    # A simulator needs the robot only: a path around the controller.
    left = boxes[0][0] - 0.45
    ax.plot([boxes[0][0], left, left], [ys[0], ys[0], ys[3]], color=GREY, lw=2.0)
    arrow(ax, (left, ys[3]), (boxes[3][0], ys[3]), color=GREY)
    ax.text(
        left - 0.1,
        (ys[0] + ys[3]) / 2,
        code("ModelPlant(robot)"),
        color=GREY,
        rotation=90,
        ha="right",
        va="center",
    )
    return fig
