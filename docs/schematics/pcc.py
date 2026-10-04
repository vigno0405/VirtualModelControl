"""One piecewise-constant-curvature segment, bent, with its frames and its bending angle."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.models import PCC

from ._draw import GREY, NAVY, RED


def figure(length: float = 0.2, radius: float = 0.03, dx: float = 0.02) -> Any:
    """A segment of rest length ``length`` [m] and section radius ``radius`` [m], bent by ``dx``."""
    segment = vmc.Mechanism("segment", model=PCC([length], radius))
    q = np.array([dx, 0.0, 0.0])
    kin = vmc.Kinematics(segment)
    end, R_end = kin.position(q, 1.0), kin.rotation(q, 1.0)
    theta = dx / radius  # bending angle D / d with Dy = 0

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 0], [0, length], color="0.8", lw=6, solid_capstyle="round")
    viz.draw_robot(ax, segment, q)
    viz.draw_frame(ax, np.eye(3), [-0.09, 0, 0], length=0.04, label="base")
    viz.draw_frame(ax, R_end, end, length=0.04)
    # the tangent at the end leans by theta from the base z axis
    ax.plot([end[0], end[0]], [end[2], end[2] + 0.07], color=GREY, ls="--", lw=1.5)
    ax.text(end[0] - 0.005, end[2] + 0.085, r"$\theta = D/d$", color=GREY, ha="center")
    ax.text(-0.012, length * 0.55, r"rest, $L_0$", color=GREY, ha="right")
    mid = kin.position(q, 0.5)
    ax.text(mid[0] + 0.015, mid[2], r"arc, $L_0 + D_l$", color=NAVY, va="center")
    ax.text(-0.15, length + 0.075, r"bends towards $(D_x, D_y)$", color=RED)
    ax.set_xlim(-0.16, 0.2)
    ax.set_ylim(-0.04, length + 0.12)
    ax.set_aspect("equal", adjustable="box")
    viz.label_axes(ax)
    return fig
