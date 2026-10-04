"""The ADAPT finger as mounted: phalanges, joints and their ranges, sites, motors and gravity."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

from ._draw import GREY, RED, TEAL

JOINTS = ("MCP", "PIP", "DIP")


def figure() -> Any:
    """Side view (y-z, z down as mounted) of the straight finger: phalanx lengths, joint ranges,
    the tip and the centres of gravity, which motor turns which joint, the frame and gravity."""
    finger = adapt.finger()
    p = finger.params
    C = np.asarray(p["coupling"].value)
    joints = [np.asarray(p[f"j{i}.point"].value, dtype=float) for i in (1, 2, 3)]
    tip = np.asarray(p["tip.position"].value, dtype=float)
    ends = [*joints[1:], tip]
    g = np.asarray(p["gravity"].value, dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    viz.draw_robot(ax, finger, np.zeros(2), plane="yz", zorder=1)
    ax.add_patch(plt.Rectangle((-0.011, -0.006), 0.007, 0.012, color=GREY, alpha=0.5))  # base
    r = 0.011  # [m], radius of the range wedges
    for k, (name, joint, end) in enumerate(zip(JOINTS, joints, ends, strict=True)):
        lo, hi = adapt.JOINT_LIMITS[name]
        phi = np.linspace(lo, hi, 30)  # a positive angle turns y towards z
        y = joint[1] + np.r_[0.0, r * np.cos(phi), 0.0]
        z = joint[2] + np.r_[0.0, r * np.sin(phi), 0.0]
        ax.fill(y, z, color=TEAL, alpha=0.25, lw=0)
        label = rf"${np.degrees(lo):.0f}$ to ${np.degrees(hi):.0f}^\circ$"
        ax.text(joint[1] + 0.2 * r, joint[2] + 1.2 * r, label, color=TEAL, va="top")
        ax.text(joint[1], joint[2] - 0.005, name, ha="center", va="bottom")
        length = 1000 * np.linalg.norm(end - joint)  # [mm], staggered so that labels never meet
        height = -0.014 - 0.008 * (k % 2)
        ax.text((joint[1] + end[1]) / 2, height, f"{length:g} mm", ha="center", va="bottom")
    cogs = np.array([p[f"{link}_cog.position"].value for link in ("mcp", "pip", "dip")])
    ax.plot(cogs[:, 1], cogs[:, 2], "+", color=RED, ms=14, mew=2.5, zorder=4)
    ax.plot(tip[1], tip[2], "o", color=RED, ms=9, zorder=4)
    ax.text(tip[1] + 0.003, tip[2], "tip", va="center")
    # which motor turns which joint (θ₀, θ₁: the motor angles), and the key to the markers
    mcp = rf"$\mathrm{{MCP}} = {C[0, 0]:.3f}\,\theta_0$"
    pip = rf"$\mathrm{{PIP}} = \mathrm{{DIP}} = {C[1, 1]:.3f}\,\theta_1$"
    ax.text(0.0, 0.034, mcp, va="center")
    ax.text(0.0, 0.045, pip, va="center")
    ax.plot(0.002, 0.056, "+", color=RED, ms=14, mew=2.5)
    ax.text(0.007, 0.056, "centres of gravity", va="center")
    # the base frame, drawn beside the base: y along the finger, z down, x into the page
    origin = np.array([0.0, -0.022, -0.016])
    viz.draw_frame(ax, np.eye(3), origin, plane="yz", length=0.012)
    ax.plot(origin[1], origin[2], "o", ms=12, mfc="white", mec=RED, mew=2, zorder=6)
    ax.plot(origin[1], origin[2], "x", ms=7, color=RED, mew=2, zorder=7)
    ax.text(origin[1] - 0.003, origin[2], "$x$", color=RED, ha="right", va="center")
    start = np.array([0.0, -0.022, 0.012])
    viz.draw_force(ax, start, 0.016 * g / np.linalg.norm(g), plane="yz", scale=1.0, color=GREY)
    ax.text(start[1] + 0.003, start[2] + 0.009, "$g$", color=GREY, va="center")
    ax.set_xlim(-0.034, 0.1)
    ax.set_ylim(-0.032, 0.062)
    ax.invert_yaxis()  # as mounted: z points down
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")  # the lengths are written on the drawing
    return fig
