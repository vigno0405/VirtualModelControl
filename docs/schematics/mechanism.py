"""What a mechanism is: the finger as a robot mechanism, then a controller on its coordinates."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

from ._draw import BLUE, GREY, NAVY, RED, TEAL, code

Q = np.array([0.9, 0.5])  # motor angles [rad] of the drawn pose, inside every joint range
ARROW = 0.012  # [m], drawn length of the weights and of their compensation
LEADER = {"arrowstyle": "-", "color": GREY, "lw": 1.2, "shrinkB": 4}


def _ground(ax: Any, y: float, z: float, w: float, h: float) -> None:
    """A fixed, hatched block with its corner at (y, z)."""
    ax.add_patch(Rectangle((y, z), w, h, facecolor="none", edgecolor=GREY, hatch="///", lw=1.5))


def _angle(ax: Any, joint: Any, before: Any, after: Any, text: str) -> None:
    """The angle from the previous link's direction (dashed) to the next link's."""
    a0, a1 = np.arctan2(before[1], before[0]), np.arctan2(after[1], after[0])
    r = 0.016
    ext = joint + 1.3 * r * np.array([np.cos(a0), np.sin(a0)])
    ax.plot(*np.c_[joint, ext], color=GREY, lw=1.5, ls="--", zorder=0)
    arc = np.linspace(a0, a1, 30)
    ax.plot(joint[0] + r * np.cos(arc), joint[1] + r * np.sin(arc), color=RED, lw=2)
    where = joint + 1.75 * r * np.array([np.cos(a0), np.sin(a0)]) + [0.0, -0.002]
    ax.text(*where, text, color=RED, ha="center", va="bottom")


def _coil(ax: Any, joint: Any) -> None:
    """A spiral spring around a joint: a spring on that joint's angle."""
    phi = np.linspace(0.0, 3.5 * np.pi, 120)
    r = 0.0019 + 0.0024 * phi / phi[-1]
    ax.plot(joint[0] + r * np.cos(phi), joint[1] + r * np.sin(phi), color=TEAL, lw=1.6, zorder=2)


def figure() -> Any:
    """The robot (its coordinates, masses and gravity) above; below, the controller on the same
    coordinates: a spring to a goal, a damper, joint-limit springs and gravity compensation."""
    finger = adapt.finger()
    kin = vmc.Kinematics(finger)
    g = np.asarray(finger.params["gravity"].value, dtype=float)
    down = ARROW * g / np.linalg.norm(g)
    yz = {s: kin.position(Q, s)[1:] for s in ("pip", "dip", "tip")}
    joints = [np.zeros(2), yz["pip"], yz["dip"]]
    ends = [yz["pip"], yz["dip"], yz["tip"]]
    cogs = [kin.position(Q, f"{k}_cog") for k in ("mcp", "pip", "dip")]
    tip = kin.position(Q, "tip")

    width, heights = (-0.032, 0.103), [(0.072, -0.012), (0.08, -0.024)]  # [m], z down
    fig, axes = plt.subplots(
        2, 1, figsize=(6.4, 9.0), layout="none",
        gridspec_kw={"height_ratios": [abs(b - a) for a, b in heights]},
    )  # fmt: skip
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, hspace=0.12)
    for ax, title in zip(axes, ("the robot", "the controller"), strict=True):
        ax.set_title(title)
        _ground(ax, -0.016, -0.007, 0.01, 0.014)  # the palm
        ax.plot([-0.006, 0.0], [0.0, 0.0], color=GREY, lw=3, zorder=0)

    # The robot: its coordinates (joint angles, the tip) and its components (masses, gravity).
    top = axes[0]
    viz.draw_robot(top, finger, Q, plane="yz", zorder=1)
    for i in range(3):
        before = np.array([1.0, 0.0]) if i == 0 else joints[i] - joints[i - 1]
        _angle(top, joints[i], before, ends[i] - joints[i], rf"$\theta_{i + 1}$")
    viz.draw_point(top, tip, plane="yz", color=NAVY, markersize=9)
    top.text(yz["tip"][0] + 0.004, yz["tip"][1] + 0.003, "tip", va="top")
    for cog in cogs:
        viz.draw_point(top, cog, plane="yz", color="black", markersize=9)
        viz.draw_force(top, cog, down, plane="yz", scale=1.0, color=BLUE)
    top.annotate(
        code("PointMass"), cogs[0][1:], (0.0, 0.034), ha="center", va="center", arrowprops=LEADER
    )
    top.annotate(
        code("Gravity"),
        cogs[1][1:] + down[1:],
        (0.03, 0.062),
        ha="center",
        va="center",
        arrowprops=LEADER,
    )
    viz.draw_frame(top, np.eye(3), [0, -0.03, 0.045], plane="yz", length=0.012)

    # The controller: virtual elements on the robot's coordinates.
    bottom = axes[1]
    viz.draw_robot(bottom, finger, Q, plane="yz", color="0.8", zorder=1)
    goal = tip + np.array([0.0, 0.024, -0.016])
    viz.draw_spring(bottom, tip, goal, plane="yz", coils=4)
    viz.draw_goal(bottom, goal, plane="yz")
    bottom.text(goal[1], goal[2] - 0.006, code("LinearSpring"), ha="center", va="bottom")
    bottom.text(goal[1] + 0.006, goal[2], "goal", color=RED, va="center")
    floor = yz["tip"][1] + 0.02
    viz.draw_damper(bottom, [0, yz["tip"][0], floor], tip, plane="yz", width=0.003)
    _ground(bottom, yz["tip"][0] - 0.008, floor, 0.016, 0.004)
    bottom.text(yz["tip"][0] + 0.006, floor - 0.009, code("LinearDamper"), va="center")
    for joint in joints:
        _coil(bottom, joint)
    bottom.annotate(
        code("LimitSpring"),
        joints[1] + [-0.003, 0.006],
        (0.012, 0.042),
        ha="center",
        va="center",
        arrowprops=LEADER,
    )
    for cog in cogs:
        viz.draw_force(bottom, cog, -down, plane="yz", scale=1.0, color=RED)
    bottom.annotate(
        code("GravityCompensation"),
        cogs[0][1:] - down[1:],
        (0.03, -0.019),
        ha="center",
        va="center",
        arrowprops=LEADER,
    )

    for ax, height in zip(axes, heights, strict=True):
        ax.set_xlim(*width)
        ax.set_ylim(*height)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    return fig
