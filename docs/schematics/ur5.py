"""The UR5 arm from the side, its joints numbered at their axes, and its DH table."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from scipy.spatial.transform import Rotation

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import ur5

from ._draw import GREY, NAVY

POSE = (np.pi, -1.2, 1.4, -np.pi / 2 - 0.2, -np.pi / 2, -np.pi / 2)
"""Joint angles of the drawing [rad]: reaching along +x, the flange facing down."""


def _axes(arm: Any, q: Any) -> np.ndarray:
    """Direction of each joint's axis at q, in the base frame (rows)."""
    out, R = [], np.eye(3)
    for i, angle in enumerate(np.asarray(q, dtype=float)):
        w = np.asarray(arm.params[f"j{i + 1}.axis"].value, dtype=float)
        w = w / np.linalg.norm(w)
        out.append(R @ w)
        R = R @ Rotation.from_rotvec(w * angle).as_matrix()
    return np.array(out)


def _links(points: np.ndarray, i: int) -> list[np.ndarray]:
    """Unit vectors (x, z) along the links that leave joint i, as seen (the next one first)."""
    p, out = points[i, [0, 2]], []
    for js in (range(i + 1, len(points)), range(i - 1, -1, -1)):
        for j in js:  # the nearest joint that does not hide behind this one
            d = points[j, [0, 2]] - p
            if np.linalg.norm(d) > 1e-3:
                out.append(d / np.linalg.norm(d))
                break
    return out


def figure(q: Any = POSE) -> Any:
    """Side view (x right, z up): joints 1 to 6, a dashed stub on the axes that lie in the page,
    and the base and tool frames."""
    arm = ur5.arm()
    q = np.asarray(q, dtype=float)
    points = viz.skeleton(arm, q)[0]  # the six joints, then the flange
    kin = vmc.Kinematics(arm)
    R, tool = kin.rotation(q, "tool"), kin.position(q, "tool")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.axhline(0.0, color=GREY, lw=1.5, zorder=0)
    viz.draw_robot(ax, arm, q, color=NAVY)
    seen = [points[:, [0, 2]]]  # everything drawn, to fit the view
    for i, a in enumerate(_axes(arm, q)):
        p, links = points[i, [0, 2]], _links(points, i)
        u = -np.sum(links, axis=0)  # away from the links
        if np.linalg.norm(u) < 0.3:  # a straight run: beside it
            u = np.array([-links[0][1], links[0][0]])
        if i > 0 and np.allclose(points[i - 1, [0, 2]], p, atol=1e-3):
            u = -u  # hidden behind the previous joint: the other side
        u = u / np.linalg.norm(u)
        a = a[[0, 2]]
        if np.linalg.norm(a) > 0.5:  # the axis lies in the page: a stub, off the next link
            stub = -a if np.dot(a, links[0]) > 0.9 else a if np.dot(a, links[0]) < -0.9 else u
            ax.plot(*np.array([p, p + 0.1 * stub]).T, color=GREY, lw=1.8, ls="--", zorder=4)
            seen.append([p + 0.1 * stub])
            if np.dot(stub, u) > 0.9:  # the stub took the label's place: beside it, outwards
                side = np.array([-u[1], u[0]])
                u = side if np.dot(side, p - points[:, [0, 2]].mean(axis=0)) >= 0 else -side
        ax.text(*(p + 0.065 * u), str(i + 1), ha="center", va="center")
        seen.append([p + 0.065 * u])
    for rotation, origin, length in ((np.eye(3), np.zeros(3), 0.12), (R, tool, 0.1)):
        viz.draw_frame(ax, rotation, origin, length=length)
        seen.append(origin[[0, 2]] + 1.3 * length * rotation[[0, 2]].T)
    for x, z, text in ((0.09, -0.05, "base"), (tool[0] + 0.07, tool[2] - 0.05, "tool")):
        ax.text(x, z, text, ha="center", va="top")
        seen.append([[x, z - 0.05]])
    x, z = np.vstack(seen).T
    ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
    ax.set_ylim(z.min() - 0.05, z.max() + 0.05)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")
    ax.set_aspect("equal", adjustable="box")
    return fig


def _angle(alpha: float) -> str:
    """An angle [rad] as a multiple of π/2 when it is one."""
    k = alpha / (np.pi / 2)
    if not np.isclose(k, round(k)):
        return f"{alpha:.4g}"
    return {0: "0", 1: "π/2", -1: "−π/2", 2: "π", -2: "−π"}.get(round(k), f"{alpha:.4g}")


def dh_table() -> Markdown:
    """The template's Denavit-Hartenberg table: d, a [m] and α [rad] of each joint."""
    rows = ["| Joint | Name | d [m] | a [m] | α [rad] |", "|---|---|---|---|---|"]
    for i, name in enumerate(ur5.JOINTS):
        d, a, alpha = ur5.DH_D[i], ur5.DH_A[i], ur5.DH_ALPHA[i]
        rows.append(f"| {i + 1} | `{name}` | {d:.5g} | {a:.5g} | {_angle(alpha)} |")
    return Markdown("\n".join(rows))
