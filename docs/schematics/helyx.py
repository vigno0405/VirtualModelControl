"""The Helyx soft arm: segments, arc parameter, frames, and the tendons of each segment."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx

from ._draw import GREY, NAVY, RED, TEAL


def sections(params: Any, prefix: str = "", first: int = 0, n: int = 3) -> Any:
    """Each segment's section seen along its z axis (towards you): the tendons at their angles,
    with the index of the motor that pulls each one; ``prefix`` selects one arm of an assembly."""
    fig, axes = plt.subplots(1, n, figsize=(6.4, 2.9), layout="none")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86, wspace=0.05)
    for i, ax in enumerate(np.atleast_1d(axes), start=1):
        d = 1000 * float(params[f"{prefix}seg{i}.d"].value)  # [mm]
        ax.add_patch(plt.Circle((0, 0), d, fill=False, color=GREY, lw=2))
        for j, delta in enumerate(np.asarray(params[f"{prefix}seg{i}.delta"].value)):
            x, y = d * np.cos(delta), d * np.sin(delta)
            ax.plot(x, y, "o", color=RED, ms=10)
            ax.text(1.55 * x, 1.55 * y, str(first + 3 * (i - 1) + j), ha="center", va="center")
        for (u, v), name, color in (((1, 0), "x", RED), ((0, 1), "y", TEAL)):
            ax.annotate(
                "",
                xy=(0.5 * d * u, 0.5 * d * v),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": color},
            )
            label = (0.55 * d, -0.32 * d) if name == "x" else (-0.32 * d, 0.55 * d)
            ax.text(*label, f"${name}$", color=color, ha="center", va="center")
        ax.set_title(f"segment {i}")
        ax.set_xlim(-1.9 * d, 1.9 * d)
        ax.set_ylim(-1.9 * d, 1.9 * d)
        ax.set_aspect("equal")
        ax.axis("off")
    return fig


def figure(geometry: str = "145-290-290") -> Any:
    """Side view of one geometry at rest: segments, arc parameter, base frame and gravity."""
    arm = helyx.arm(geometry)
    p = arm.params
    L0 = np.array([float(p[f"seg{i}.L0"].value) for i in (1, 2, 3)])
    breaks = np.concatenate([[0.0], np.cumsum(L0)]) / L0.sum()
    g = np.asarray(p["gravity"].value, dtype=float)
    kin = vmc.Kinematics(arm)
    q = np.zeros(9)

    fig, ax = plt.subplots(figsize=(4.6, 6.0))
    viz.draw_robot(ax, arm, q, zorder=1)
    for s in breaks:
        z = kin.position(q, s)[2]
        ax.plot([-0.02, 0.02], [z, z], color=NAVY, lw=2)
        ax.text(0.035, z, f"$s = {s:.2g}$", va="center", color=NAVY)
    for i in range(3):
        z = kin.position(q, (breaks[i] + breaks[i + 1]) / 2)[2]
        ax.text(0.035, z, f"segment {i + 1}\n{1000 * L0[i]:.0f} mm", va="center")
    viz.draw_frame(ax, np.eye(3), [-0.12, 0, 0.0], length=0.06)
    if np.hypot(g[0], g[2]) > 1e-9:  # gravity in the drawing's plane: an arrow
        z0 = 0.62 * L0.sum()
        viz.draw_force(ax, [-0.12, 0, z0], 0.1 * g / np.linalg.norm(g), scale=1.0, color=GREY)
        ax.text(-0.105, z0 + 0.05 * np.sign(g[2]), "$g$", color=GREY, va="center")
    else:
        ax.text(-0.19, 0.62 * L0.sum(), "$g$ along $-y$", color=GREY)
    ax.set_xlim(-0.2, 0.25)
    ax.set_ylim(-0.1, 1.06 * L0.sum())
    if g[2] > 0:  # the arm hangs: z points down
        ax.invert_yaxis()
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")
    ax.set_aspect("equal", adjustable="box")
    return fig
