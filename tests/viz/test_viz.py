"""Figures: every robot type draws, and saving writes both a PDF and an SVG."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt, bimanual, helyx


def test_skeletons_have_the_right_shape():
    arm = viz.skeleton(helyx.arm(), np.zeros(9))
    assert len(arm) == 1 and arm[0].shape == (60, 3)
    np.testing.assert_allclose(arm[0][-1], [0, 0, 0.725], atol=1e-12)
    assert viz.skeleton(adapt.finger(), [0.0, 0.0])[0].shape == (4, 3)  # 3 joints + tip
    assert len(viz.skeleton(adapt.hand(), np.zeros(13))) == 5
    tips = [line[-1] for line in viz.skeleton(bimanual.arms(), np.zeros(18))]
    np.testing.assert_allclose(tips, [[0.125, 0, 0.58], [-0.125, 0, 0.58]], atol=1e-12)


def test_drawing_and_saving(tmp_path):
    with viz.style(usetex=False):
        fig, ax = plt.subplots()
        viz.draw_robot(ax, helyx.arm(), np.zeros(9), label="arm")
        viz.draw_spring(ax, [0, 0, 0.7], [0.2, 0, 0.6])
        viz.draw_goal(ax, [0.2, 0, 0.6])
        viz.draw_point(ax, [0, 0, 0.725])
        viz.draw_force(ax, [0, 0, 0.725], [1.0, 0, 0])
        viz.label_axes(ax)
        pdf, svg = viz.save(fig, tmp_path / "figure")
        plt.close(fig)
    assert pdf.stat().st_size > 0 and svg.stat().st_size > 0
