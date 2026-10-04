"""The ADAPT hand from the front at q = 0: its five digits and the motor that drives each joint."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

from ._draw import GREY, NAVY, RED


def figure() -> Any:
    """Front view (x right, z up) with each motor's index beside the joints it drives; the
    motor that drives joints of several fingers (the spread) is joined to each of them."""
    hand = adapt.hand()
    coupling = np.asarray(hand.model.coupling.value)  # (20 joints, 13 motors)
    q = np.zeros(coupling.shape[1])
    lines = viz.skeleton(hand, q)  # per digit: its four joints, then the tip
    digit = np.repeat(np.arange(len(lines)), 4)  # the digit of each joint (row of coupling)
    shared = [m for m in range(q.size) if len(set(digit[coupling[:, m] != 0])) > 1]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    viz.draw_robot(ax, hand, q, color=NAVY)
    labels = []
    for m in range(q.size):
        rows = np.flatnonzero(coupling[:, m])
        points = np.array([lines[digit[r]][r % 4] for r in rows])
        if m in shared:  # one label below the fingers, joined to every joint it turns
            x, z = points[:, 0].mean(), points[:, 2].min() - 0.03
            for p in points:
                ax.plot([x, p[0]], [z, p[2]], color=GREY, lw=1.5, ls="--", zorder=0)
        else:  # beside its joints, on the left of the digit as it runs to the tip
            line = lines[digit[rows[0]]]
            step = line[rows[-1] % 4 + 1] - line[rows[0] % 4]
            left = np.array([-step[2], step[0]]) / np.hypot(step[0], step[2])
            x, z = points[:, [0, 2]].mean(axis=0) + 0.012 * left
        ax.text(x, z, str(m), color=RED, ha="center", va="center")
        labels.append((x, z))
    x, z = np.vstack([np.vstack(lines)[:, [0, 2]], labels]).T
    ax.set_xlim(x.min() - 0.012, x.max() + 0.012)
    ax.set_ylim(z.min() - 0.012, z.max() + 0.012)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")
    ax.set_aspect("equal", adjustable="box")
    return fig
