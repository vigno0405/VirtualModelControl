"""Animations: every format is written with the right number of frames, small enough for the
docs, for robots with and without geometry, without leaving figures open."""

import matplotlib

matplotlib.use("Agg")

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx, turtle

GOAL = np.array([0.25, 0.0, 0.55])


@pytest.fixture(scope="module")
def run():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - GOAL, 600.0))
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(1 / 330), T=5.0)
    return arm, log


@pytest.mark.parametrize("suffix", [".mp4", ".gif", ".webp"])
def test_every_format_has_one_frame_per_tick_of_the_video_clock(run, tmp_path, suffix):
    arm, log = run
    path = vmc.viz.animate(
        arm, log, tmp_path / f"arm{suffix}", fps=10, springs=[(1.0, GOAL)], trace=1.0
    )
    t = log.arrays()["t"].ravel()
    expected = len(np.arange(t[0], t[-1] + 1e-12, 1 / 10))
    if suffix == ".mp4":
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    else:
        frames = Image.open(path).n_frames
    assert frames == expected
    assert plt.get_fignums() == []


def test_a_five_second_mp4_fits_in_the_docs_budget(run, tmp_path):
    arm, log = run
    path = vmc.viz.animate(arm, log, tmp_path / "arm.mp4", springs=[(1.0, GOAL)], invert=True)
    assert 0 < path.stat().st_size < 2_000_000


def test_speed_shortens_the_video(run, tmp_path):
    arm, log = run
    animation = vmc.viz.animate(arm, log, fps=5)  # nothing saved: the animation itself
    assert isinstance(animation, FuncAnimation)
    animation.save(tmp_path / "own.gif", writer=PillowWriter(fps=5))
    plt.close("all")
    slow = vmc.viz.animate(arm, log, tmp_path / "slow.gif", fps=5)
    fast = vmc.viz.animate(arm, log, tmp_path / "fast.gif", fps=5, speed=2.0)
    assert Image.open(fast).n_frames == (Image.open(slow).n_frames + 1) // 2


def test_a_robot_without_geometry_is_drawn_by_the_callback(tmp_path):
    t = np.linspace(0.0, 1.0, 101)
    log = {"t": t[:, None], "q": np.column_stack([t, -t]), "z": np.column_stack([t, 0 * t])}
    seen = []

    def draw(ax, row):
        seen.append(row["t"])
        for angle, x in zip(row["q"], (-1.0, 1.0), strict=True):
            ax.plot([x, x + np.cos(angle)], [0.0, np.sin(angle)], color="k")

    path = vmc.viz.animate(
        turtle.robot(), log, tmp_path / "cranks.gif", fps=10, draw=draw, limits=((-2, 2), (-1, 1))
    )
    assert Image.open(path).n_frames == 11 and len(seen) >= 11
    with pytest.raises(ValueError, match="nothing to draw"):
        vmc.viz.animate(turtle.robot(), log, tmp_path / "no.gif")


def test_unknown_format_is_refused(run, tmp_path):
    arm, log = run
    with pytest.raises(ValueError, match="unknown animation format"):
        vmc.viz.animate(arm, log, tmp_path / "arm.avi", fps=5)
    assert plt.get_fignums() == []
