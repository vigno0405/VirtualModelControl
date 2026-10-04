"""The two-arm Helyx: both arms upright on one frame, their bases, motor order and gravity."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import bimanual

from ._draw import GREY, NAVY


def figure() -> Any:
    """Front view (x-z) of the pair at rest: segments, bases, motor indices, frame and gravity."""
    robot = bimanual.arms()
    p = robot.params
    q = np.zeros(robot.model.space.nq)
    kin = vmc.Kinematics(robot)
    g = np.asarray(p["gravity"].value, dtype=float)
    sizes = [act.motor_sizes(space)[0] for space, act, _ in robot.actuation.parts]
    first = dict(zip(bimanual.ARMS, np.cumsum([0, *sizes[:-1]]), strict=True))

    fig, ax = plt.subplots(figsize=(4.8, 6.4))
    viz.draw_robot(ax, robot, q, zorder=1)
    bases = [np.asarray(p[f"{arm}.mount.position"].value, dtype=float) for arm in bimanual.ARMS]
    xs = [b[0] for b in bases]
    ax.plot([min(xs) - 0.05, max(xs) + 0.05], [-0.022] * 2, color=GREY, lw=8, solid_capstyle="butt")
    for arm, base, n in zip(bimanual.ARMS, bases, sizes, strict=True):
        top = kin.position(q, (arm, 1.0))
        ax.text(base[0], top[2] + 0.03, arm, ha="center", va="bottom")
        last = first[arm] + n - 1
        ax.text(base[0], -0.05, f"motors\n{first[arm]} to {last}", ha="center", va="top")
    # segment ends and lengths, on the right arm (the left one is the same by default)
    L0 = [float(p[f"right.seg{i}.L0"].value) for i in (1, 2, 3)]
    breaks = np.concatenate([[0.0], np.cumsum(L0)]) / sum(L0)
    x = bases[0][0]
    for i, s in enumerate(breaks[1:]):
        z = kin.position(q, ("right", s))[2]
        ax.plot([x - 0.015, x + 0.015], [z, z], color=NAVY, lw=2)
        mid = kin.position(q, ("right", (breaks[i] + s) / 2))[2]
        ax.text(x + 0.03, mid, f"{1000 * L0[i]:.0f} mm", va="center")
    viz.draw_frame(ax, np.eye(3), [0.0, 0.0, 0.0], length=0.07)
    z0 = 0.55 * sum(L0)
    viz.draw_force(ax, [0.0, 0.0, z0], 0.1 * g / np.linalg.norm(g), scale=1.0, color=GREY)
    ax.text(0.02, z0 - 0.05, "$g$", color=GREY, va="center")
    ticks = sorted([0.0, *xs])
    ax.set_xticks(ticks, [f"${tick:g}$" for tick in ticks])
    ax.set_xlim(-0.26, 0.3)
    ax.set_ylim(-0.17, sum(L0) + 0.1)
    viz.label_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    return fig
