"""The library's layers, the edges beside them, and the packages planned in the roadmap."""

from __future__ import annotations

from typing import Any

from ._draw import BLUE, GREY, NAVY, RED, TEAL, arrow, box, canvas, code

# Bottom to top, as the import contracts in pyproject.toml order them.
LAYERS = [
    ("core", "parameters, spaces, signals"),
    ("mechanisms", "coordinates, components"),
    ("models", "kinematics, actuation"),
    ("system", "robot + controller"),
    ("compiler, dynamics", "control law, equations of motion"),
    ("control", "controller, output stages"),
    ("sim", "plants, run loop, log"),
]


def figure() -> Any:
    """Each layer imports only the ones below it; the edges may use all of them."""
    fig, ax = canvas(6.4, 8.35)
    x, w, h, gap = 2.55, 3.9, 0.78, 0.25
    tops = []
    for k, (name, what) in enumerate(LAYERS):
        y = 0.6 + k * (h + gap)
        box(ax, x, y, w, h, code(name) + "\n" + what, TEAL if name == "sim" else NAVY, size=16)
        tops.append(y)
    planned = tops[-1] + h + gap
    box(
        ax,
        x,
        planned,
        w,
        h,
        "estimation, optimization,\nlearning (planned)",
        GREY,
        dashed=True,
        size=16,
    )
    # a controller needs the layers up to control, nothing above
    low, high = tops[0] - h / 2, tops[5] + h / 2
    ax.plot([0.42, 0.3, 0.3, 0.42], [low, low, high, high], color=RED, lw=2)
    ax.text(
        0.2,
        (low + high) / 2,
        "a controller needs only these",
        rotation=90,
        ha="right",
        va="center",
        color=RED,
        size=16,
    )
    # the edges: they may import every layer, and no layer imports them
    ex = 5.75
    for y, label, kind in ((tops[6], "robots", BLUE), (tops[4], "viz", BLUE)):
        box(ax, ex, y, 1.1, h, code(label), kind, size=16)
        arrow(ax, (ex - 0.56, y), (x + w / 2, y), color=GREY)
    box(ax, ex, tops[2], 1.1, h + 0.2, "hardware,\nROS\n(planned)", GREY, dashed=True, size=14)
    ax.text(ex, tops[6] + h / 2 + 0.25, "edges", ha="center", color=GREY, size=16)
    return fig
