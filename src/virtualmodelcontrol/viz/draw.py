"""Drawing robots, springs, goals and forces on matplotlib axes, in a chosen plane."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from numpy.typing import ArrayLike

from ..core.params import constants
from ..models import PCC, Assembly, LinearCoupling, SerialChain, evaluate_frame
from .style import PALETTE

AXES = {"x": 0, "y": 1, "z": 2}


def _model(robot: Any) -> Any:
    return robot.model if hasattr(robot, "components") else robot


def project(points: ArrayLike, plane: str = "xz") -> tuple[np.ndarray, np.ndarray]:
    """The two coordinates of 3D points seen in ``plane`` ("xz": x to the right, z up)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return p[:, AXES[plane[0]]], p[:, AXES[plane[1]]]


def skeleton(robot: Any, q: ArrayLike, n: int = 60) -> list[np.ndarray]:
    """Polylines (k, 3) [m] tracing the robot's body at configuration q."""
    return [line for line, _ in _parts(robot, q, n)]


def _parts(robot: Any, q: ArrayLike, n: int = 60) -> list[tuple[np.ndarray, bool]]:
    """Each polyline of the body, and whether its inner vertices are joints (rigid chains)."""
    model, q = _model(robot), np.asarray(q, dtype=float)
    p = constants(model.params)
    if isinstance(model, Assembly):
        parts = []
        for name, part in model.parts.items():
            R_m, t_m = model.mount(ca.DM(q), name, p)
            R, t = np.array(ca.evalf(R_m)), np.array(ca.evalf(t_m)).ravel()
            for line, rigid in _parts(part, q[model._slices[name]], n):
                parts.append((line @ R.T + t, rigid))
        return parts
    if isinstance(model, LinearCoupling):
        return _parts(model.model, np.array(model.coupling.value) @ q, n)
    if isinstance(model, PCC):
        s = ca.SX.sym("s")
        f = ca.Function("backbone", [s], [model.frame(ca.DM(q), s, p)[1]])
        return [(np.array(f.map(n)(np.linspace(0.0, 1.0, n))).T, False)]
    if isinstance(model, SerialChain):
        points = [np.array(x).ravel() for x in model.joint_points(ca.DM(q), p)]
        end = next((site for site in ("tip", "tool") if site in model.sites), None)
        if end is not None:
            points.append(evaluate_frame(model, q, end)[1])
        return [(np.array(points), True)]
    if not model.sites:  # joint-space models have no geometry to draw
        return []
    return [(np.array([evaluate_frame(model, q, site)[1] for site in model.sites]), True)]


def draw_robot(
    ax: Any,
    robot: Any,
    q: ArrayLike,
    *,
    plane: str = "xz",
    color: str | None = None,
    linewidth: float | None = None,
    joints: bool = True,
    label: str | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Draw the robot's body; joints of rigid chains as dots."""
    return _draw_parts(ax, _parts(robot, q), plane, color, linewidth, joints, label, **kwargs)


def _draw_parts(
    ax: Any,
    parts: list[tuple[np.ndarray, bool]],
    plane: str,
    color: str | None,
    linewidth: float | None,
    joints: bool,
    label: str | None,
    **kwargs: Any,
) -> list[Any]:
    color = PALETTE[3] if color is None else color
    linewidth = 6.0 if linewidth is None else linewidth
    artists = []
    for i, (line, rigid) in enumerate(parts):
        x, y = project(line, plane)
        artists += ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            label=label if i == 0 else None,
            **kwargs,
        )
        if joints and rigid and len(line) > 1:
            artists += ax.plot(
                x[:-1],
                y[:-1],
                "o",
                color="white",
                markeredgecolor=color,
                markersize=linewidth * 1.4,
                markeredgewidth=2,
                zorder=3,
            )
    ax.set_aspect("equal", adjustable="datalim")
    return artists


def draw_point(ax: Any, point: ArrayLike, *, plane: str = "xz", **kwargs: Any) -> list[Any]:
    """A dot at a 3D point."""
    x, y = project(point, plane)
    kwargs = cbook.normalize_kwargs(kwargs, Line2D)
    kwargs.setdefault("color", PALETTE[1])
    return ax.plot(x, y, "o", zorder=4, **kwargs)


def draw_goal(ax: Any, point: ArrayLike, *, plane: str = "xz", **kwargs: Any) -> list[Any]:
    """A cross marking a goal."""
    x, y = project(point, plane)
    kwargs = cbook.normalize_kwargs(kwargs, Line2D)
    kwargs.setdefault("color", PALETTE[1])
    kwargs.setdefault("markersize", 14)
    kwargs.setdefault("markeredgewidth", 3)
    return ax.plot(x, y, "x", zorder=4, **kwargs)


def draw_spring(
    ax: Any,
    start: ArrayLike,
    end: ArrayLike,
    *,
    plane: str = "xz",
    coils: int = 6,
    width: float | None = None,
    **kwargs: Any,
) -> list[Any]:
    """A zig-zag spring between two 3D points, as seen in ``plane``."""
    (x0,), (y0,) = project(start, plane)
    (x1,), (y1,) = project(end, plane)
    a, b = np.array([x0, y0]), np.array([x1, y1])
    length = np.linalg.norm(b - a)
    if length == 0:
        return []
    d = (b - a) / length
    nrm = np.array([-d[1], d[0]])
    amplitude = 0.06 * float(length) if width is None else width
    t = np.linspace(0.2, 0.8, 2 * coils + 1)
    offsets = amplitude * np.array([0, *[(-1) ** k for k in range(1, 2 * coils)], 0])
    pts = [a, *(a + (b - a) * ti + nrm * oi for ti, oi in zip(t, offsets, strict=True)), b]
    xs, ys = np.array(pts).T
    kwargs = cbook.normalize_kwargs(kwargs, Line2D)
    kwargs.setdefault("color", PALETTE[2])
    kwargs.setdefault("linewidth", 2.0)
    return ax.plot(xs, ys, **kwargs)


def draw_damper(
    ax: Any,
    start: ArrayLike,
    end: ArrayLike,
    *,
    plane: str = "xz",
    width: float | None = None,
    **kwargs: Any,
) -> list[Any]:
    """A dashpot (a piston in a cylinder) between two 3D points, as seen in ``plane``."""
    (x0,), (y0,) = project(start, plane)
    (x1,), (y1,) = project(end, plane)
    a, b = np.array([x0, y0]), np.array([x1, y1])
    length = float(np.linalg.norm(b - a))
    if length == 0:
        return []
    d, nrm = (b - a) / length, np.array([-(b - a)[1], (b - a)[0]]) / length
    w = 0.08 * length if width is None else width

    def at(t: float, o: float = 0.0) -> np.ndarray:
        return a + d * length * t + nrm * o

    paths = [
        [at(0.0), at(0.3)],  # rod to the cylinder
        [at(0.65, w), at(0.3, w), at(0.3, -w), at(0.65, -w)],  # cylinder
        [at(0.5, 0.7 * w), at(0.5, -0.7 * w)],  # piston
        [at(0.5), at(1.0)],  # piston rod
    ]
    kwargs = cbook.normalize_kwargs(kwargs, Line2D)
    kwargs.setdefault("color", PALETTE[0])
    kwargs.setdefault("linewidth", 2.0)
    kwargs.setdefault("solid_capstyle", "butt")
    artists = []
    for path in paths:
        xs, ys = np.array(path).T
        artists += ax.plot(xs, ys, **kwargs)
    return artists


def draw_frame(
    ax: Any,
    rotation: ArrayLike,
    origin: ArrayLike,
    *,
    plane: str = "xz",
    length: float = 0.05,
    label: str | None = None,
    **kwargs: Any,
) -> list[Any]:
    """The x, y, z axes of a frame (red, teal, blue, or ``color``) [m], skipping axes seen
    end-on; other keyword arguments style the arrows."""
    R = np.asarray(rotation, dtype=float)
    (x0,), (y0,) = project(origin, plane)
    kwargs = cbook.normalize_kwargs(kwargs, FancyArrowPatch)
    kwargs.setdefault("linewidth", 2.0)
    colors = (
        [kwargs.pop("color")] * 3 if "color" in kwargs else [PALETTE[1], PALETTE[2], PALETTE[0]]
    )
    artists = []
    for k, (name, color) in enumerate(zip("xyz", colors, strict=True)):
        (dx,), (dy,) = project(R[:, k] * length, plane)
        if np.hypot(dx, dy) < 0.15 * length:
            continue
        arrow = {"arrowstyle": "-|>", "color": color, **kwargs}
        artists.append(
            ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=arrow, zorder=5)
        )
        artists.append(
            ax.text(
                x0 + 1.3 * dx, y0 + 1.3 * dy, f"${name}$", color=color, ha="center", va="center"
            )
        )
    if label is not None:
        artists.append(ax.text(x0, y0, f"{label}  ", ha="right", va="top"))
    return artists


def draw_force(
    ax: Any,
    origin: ArrayLike,
    force: ArrayLike,
    *,
    plane: str = "xz",
    scale: float = 0.05,
    **kwargs: Any,
) -> Any:
    """An arrow for a force [N] at a point, ``scale`` metres per newton."""
    (x,), (y,) = project(origin, plane)
    (fx,), (fy,) = project(np.asarray(force, dtype=float) * scale, plane)
    kwargs.setdefault("color", PALETTE[1])
    return ax.annotate(
        "",
        xy=(x + fx, y + fy),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "lw": 2.5, **kwargs},
    )


def label_axes(ax: Any, plane: str = "xz", unit: str = "m") -> None:
    """Axis labels for a plane view."""
    ax.set_xlabel(f"${plane[0]}$ [{unit}]")
    ax.set_ylabel(f"${plane[1]}$ [{unit}]")
