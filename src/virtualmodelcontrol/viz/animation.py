"""Animations of a run: the robot drawn frame by frame, saved as MP4, GIF or animated WebP."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from numpy.typing import ArrayLike

from ..models import Kinematics
from .draw import _draw_parts, _parts, draw_goal, draw_spring, label_axes, project
from .style import PALETTE, rc

_SAVE = {"savefig.bbox": None, "figure.constrained_layout.use": True}
FORMATS = (".mp4", ".gif", ".webp")


def animate(
    robot: Any,
    log: Any,
    path: str | Path | None = None,
    *,
    plane: str = "xz",
    fps: int = 25,
    speed: float = 1.0,
    springs: Iterable[tuple[Any, ArrayLike]] = (),
    trace: Any = None,
    draw: Callable[[Any, dict[str, Any]], Any] | None = None,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    invert: bool = False,
    figsize: tuple[float, float] | None = None,
    font_size: float = 18.0,
    dpi: float = 100.0,
) -> Any:
    """Animate a run from ``log`` (``t`` and ``q``): the robot, springs from sites to fixed points,
    the path of the ``trace`` site, and ``draw(ax, row)`` for anything else. The suffix of ``path``
    (.mp4, .gif, .webp) picks the format; without a path, returns the matplotlib animation.
    """
    if path is not None and Path(path).suffix.lower() not in FORMATS:
        raise ValueError(
            f"unknown animation format {Path(path).suffix!r}: use {', '.join(FORMATS)}"
        )
    rows = log.arrays() if hasattr(log, "arrays") else {k: np.asarray(v) for k, v in log.items()}
    t = np.asarray(rows["t"], dtype=float).ravel()
    q = np.atleast_2d(np.asarray(rows["q"], dtype=float))
    frames = _frames(t, fps, speed)
    springs = [(site, np.asarray(point, dtype=float)) for site, point in springs]
    kin = Kinematics(robot) if springs or trace is not None else None

    bodies = [_parts(robot, q[i]) for i in frames]
    anchors = [[kin.position(q[i], site) for site, _ in springs] for i in frames] if kin else []
    path_points = (
        np.array([kin.position(q[i], trace) for i in frames])
        if kin is not None and trace is not None
        else None
    )
    if limits is None:
        limits = _limits(bodies, springs, anchors, path_points, plane)
    (x0, x1), (y0, y1) = limits
    if figsize is None:
        height = 5.0
        figsize = (float(np.clip(height * (x1 - x0) / (y1 - y0) + 1.2, 3.6, 8.0)), height)
    # Frames are drawn many times: mathtext instead of a LaTeX run per frame, saved as drawn.
    style = {**rc(usetex=False, font_size=font_size), **_SAVE}

    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def update(k: int) -> list[Any]:
        i = frames[k]
        with mpl.rc_context(style):  # also when the animation is saved or shown later
            ax.cla()
            if path_points is not None:
                px, py = project(path_points[: k + 1], plane)
                ax.plot(px, py, color=PALETTE[0], linewidth=2.0, alpha=0.8)
            _draw_parts(ax, bodies[k], plane, None, None, True, None)
            for (_, goal), anchor in zip(springs, anchors[k] if springs else [], strict=True):
                draw_spring(ax, anchor, goal, plane=plane)
                draw_goal(ax, goal, plane=plane)
            if draw is not None:
                draw(ax, {name: values[i] for name, values in rows.items()} | {"t": t[i]})
            ax.set_xlim(x0, x1)
            ax.set_ylim((y1, y0) if invert else (y0, y1))
            ax.set_aspect("equal", adjustable="box")
            label_axes(ax, plane)
            ax.set_title(f"$t = {t[i]:.2f}$ s")
        return []  # redrawn whole (no blitting)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps)
    if path is None:
        return animation
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with mpl.rc_context(style):
            animation.save(path, writer=_writer(path, fps), dpi=dpi)
    finally:
        plt.close(fig)
    return path


def _frames(t: np.ndarray, fps: int, speed: float) -> np.ndarray:
    """Indices of the logged rows shown at ``fps`` frames per second of video."""
    times = np.arange(t[0], t[-1] + 1e-12, speed / fps)
    return np.clip(np.searchsorted(t, times - 1e-12), 0, len(t) - 1)


def _limits(
    bodies: list[Any],
    springs: list[Any],
    anchors: list[list[np.ndarray]],
    path_points: np.ndarray | None,
    plane: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits that hold every frame, with a margin."""
    points = [line for body in bodies for line, _ in body]
    points += [goal[None, :] for _, goal in springs]
    points += [anchor[None, :] for frame in anchors for anchor in frame]
    if path_points is not None:
        points.append(path_points)
    if not points:
        raise ValueError("nothing to draw: give a robot with geometry, or set limits= and draw=")
    x, y = project(np.vstack(points), plane)
    span = max(np.ptp(x), np.ptp(y), 1e-3)
    margin = 0.1 * span
    return (x.min() - margin, x.max() + margin), (y.min() - margin, y.max() + margin)


def _writer(path: Path, fps: int) -> Any:
    """The movie writer for a file suffix."""
    if path.suffix.lower() == ".mp4":
        import imageio_ffmpeg

        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMpegWriter(fps=fps, codec="h264", extra_args=["-movflags", "+faststart"])
    return PillowWriter(fps=fps)  # .gif or .webp
