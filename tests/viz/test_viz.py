"""Figures: every robot type draws, and saving writes both a PDF and an SVG."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt, bimanual, helyx, turtle, ur5


def test_skeletons_have_the_right_shape():
    arm = viz.skeleton(helyx.arm(), np.zeros(9))
    assert len(arm) == 1 and arm[0].shape == (60, 3)
    np.testing.assert_allclose(arm[0][-1], [0, 0, 0.725], atol=1e-12)
    assert viz.skeleton(adapt.finger(), [0.0, 0.0])[0].shape == (4, 3)  # 3 joints + tip
    assert len(viz.skeleton(adapt.hand(), np.zeros(13))) == 5
    tips = [line[-1] for line in viz.skeleton(bimanual.arms(), np.zeros(18))]
    np.testing.assert_allclose(tips, [[0.125, 0, 0.58], [-0.125, 0, 0.58]], atol=1e-12)
    assert viz.skeleton(turtle.robot(), [0.0, 0.0]) == []  # joint space: nothing to draw
    q = np.array([0.3, -1.0, 1.2, -0.5, 0.4, 0.2])
    tool = vmc.Kinematics(ur5.arm()).position(q, "tool")
    np.testing.assert_allclose(viz.skeleton(ur5.arm(), q)[0][-1], tool, atol=1e-12)  # to the flange


def test_joint_dots_only_on_rigid_chains():
    with viz.style(usetex=False):
        fig, ax = plt.subplots()
        arms = viz.draw_robot(ax, bimanual.arms(), np.zeros(18))
        finger = viz.draw_robot(ax, adapt.finger(), [0.3, 0.5])
        plt.close(fig)
    assert len(arms) == 2  # two continuum lines, no joint markers
    assert len(finger) == 2 and len(finger[1].get_xdata()) == 3  # MCP, PIP, DIP


def test_drawing_and_saving(tmp_path):
    with viz.style(usetex=False):
        fig, ax = plt.subplots()
        viz.draw_robot(ax, helyx.arm(), np.zeros(9), label="arm")
        viz.draw_spring(ax, [0, 0, 0.7], [0.2, 0, 0.6])
        viz.draw_goal(ax, [0.2, 0, 0.6])
        viz.draw_point(ax, [0, 0, 0.725])
        viz.draw_force(ax, [0, 0, 0.725], [1.0, 0, 0])
        assert len(viz.draw_damper(ax, [0, 0, 0.7], [0.2, 0, 0.6])) == 4
        assert viz.draw_damper(ax, [0, 0, 0], [0, 0.1, 0]) == []  # seen end-on
        frame = viz.draw_frame(ax, np.eye(3), [0, 0, 0], label="base")
        assert len(frame) == 5  # x and z arrows with their letters; y is seen end-on
        viz.label_axes(ax)
        pdf, svg = viz.save(fig, tmp_path / "figure")
        plt.close(fig)
    assert pdf.stat().st_size > 0 and svg.stat().st_size > 0


def test_style_keywords_and_their_aliases_reach_the_lines():
    with viz.style(usetex=False):
        fig, ax = plt.subplots()
        damper = viz.draw_damper(ax, [0, 0, 0], [0.2, 0, 0], lw=3.0, solid_capstyle="round")
        spring = viz.draw_spring(ax, [0, 0, 0], [0.2, 0, 0], lw=1.0)
        frame = viz.draw_frame(ax, np.eye(3), [0, 0, 0], linewidth=4.0, color="k")
        plt.close(fig)
    assert damper[0].get_linewidth() == 3.0 and damper[0].get_solid_capstyle() == "round"
    assert spring[0].get_linewidth() == 1.0
    assert frame[0].arrow_patch.get_linewidth() == 4.0 and frame[1].get_color() == "k"
