"""The turtle's controller: two cranks tied to a virtual flywheel that a speed regulator drives."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import turtle

from ._draw import GREY, LAVENDER, NAVY, RED, TEAL, box, canvas


def _xz(x: float, z: float) -> list[float]:
    """A point of the drawing (inches) as the 3D point the viz primitives expect."""
    return [x, 0.0, z]


def _branch(ax: Any, x: float, top: float, bottom: float) -> None:
    """A spring K and a damper C side by side between two bars, centered on x."""
    for z in (top, bottom):
        ax.plot([x - 0.3, x + 0.3], [z, z], color="black", lw=2.0)
    viz.draw_spring(ax, _xz(x - 0.3, top), _xz(x - 0.3, bottom), width=0.12)
    viz.draw_damper(ax, _xz(x + 0.3, top), _xz(x + 0.3, bottom), width=0.12)
    ax.text(x - 0.5, (top + bottom) / 2, "$K$", ha="right", va="center")
    ax.text(x + 0.5, (top + bottom) / 2, "$C$", ha="left", va="center")


def _crank(ax: Any, x: float, z: float, angle: float, label: str) -> None:
    """A crank turning about (x, z), its arm at ``angle``, the bias torque round it."""
    r = 0.42
    ax.add_patch(Circle((x, z), r, fill=False, edgecolor=GREY, lw=1.5, ls="--"))
    tip = (x + r * np.cos(angle), z + r * np.sin(angle))
    ax.plot([x, tip[0]], [z, tip[1]], color=NAVY, lw=6, solid_capstyle="round")
    ax.plot(x, z, "o", color="white", markeredgecolor=NAVY, markersize=9, markeredgewidth=2)
    a, b = np.radians(200), np.radians(-20)  # the bias torque: an arc arrow below the crank
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.62 * np.cos(a), z + 0.62 * np.sin(a)),
            (x + 0.62 * np.cos(b), z + 0.62 * np.sin(b)),
            connectionstyle="arc3,rad=0.55",
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.8,
            color=RED,
        )
    )
    ax.text(x + 0.72, z - 0.42, r"$\tau_b$", color=RED, ha="left", va="center")
    ax.text(x, z - 0.95, label, ha="center", va="top")


def figure() -> Any:
    """The flywheel φ (inertia J_v) driven towards the speed ω̄ through b_v; each crank tied to
    it by a spring K and a damper C, the right one to φ − δ; a bias torque τ_b on both cranks;
    under each crank, its name and motor sign."""
    fig, ax = canvas(6.0, 6.8)
    x0, xl, xr = 3.0, 1.15, 4.85  # flywheel, left and right cranks
    zf, zc = 4.5, 1.65  # heights of the flywheel and of the cranks
    # the speed regulator: a damper b_v from the commanded speed to the flywheel
    box(ax, x0, zf + 1.8, 2.2, 0.62, r"speed $\bar\omega$", TEAL)
    viz.draw_damper(ax, _xz(x0, zf + 1.49), _xz(x0, zf + 0.6), width=0.14)
    ax.text(x0 + 0.3, zf + 1.05, "$b_v$", ha="left", va="center")
    # the flywheel: a state of the controller, with inertia
    ax.add_patch(Circle((x0, zf), 0.6, facecolor=LAVENDER, alpha=0.35, edgecolor=NAVY, lw=2))
    ax.plot([x0, x0 + 0.6 * np.cos(1.0)], [zf, zf + 0.6 * np.sin(1.0)], color=NAVY, lw=3)
    ax.text(x0 - 0.2, zf - 0.15, r"$\varphi$", ha="center", va="center")
    ax.text(x0 + 0.72, zf + 0.55, "$J_v$", ha="left", va="center")
    # the two branches: the left crank follows φ, the right one φ − δ
    xd = 4.2  # the phase offset, on the way to the right branch
    ax.plot([x0 - 0.6, xl, xl], [zf, zf, zf - 0.6], color="black", lw=2.0)
    ax.plot([x0 + 0.6, xd - 0.35], [zf, zf], color="black", lw=2.0)
    ax.plot([xd + 0.35, xr, xr], [zf, zf, zf - 0.6], color="black", lw=2.0)
    box(ax, xd, zf, 0.7, 0.55, r"$-\delta$", RED)
    for x in (xl, xr):
        _branch(ax, x, zf - 0.6, zf - 2.0)
        ax.plot([x, x], [zf - 2.0, zc + 0.42], color="black", lw=2.0)
    angles = (1.0, 1.0 - np.pi / 3)  # the right crank drawn δ = 60° behind the left one
    for x, crank, sign, angle in zip(
        (xl, xr), turtle.CRANKS, turtle.MOTOR_SIGNS, angles, strict=True
    ):
        _crank(ax, x, zc, angle, f"{crank}, sign ${sign:+.0f}$")
    return fig
