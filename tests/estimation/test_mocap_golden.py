"""The inversion and the velocity filter against the lab's: markers to Δ, and the velocity of Δ.

The lab inverts three markers (at s = 0.5, 0.75, 1.0 of the arm, in its base frame) in closed
form, section by section, and low-passes the finite difference of consecutive Δ. The fixture holds
the markers of random bends (clean and with 0.4 mm of noise) with the lab's Δ for them, and a run
of 40 frames at 120 Hz, two of them lost, with the lab's Δ and Δ̇ at each frame for two smoothings.
"""

from pathlib import Path

import numpy as np
import pytest

from virtualmodelcontrol.estimation import Inversion, VelocityFilter
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "mocap.npz")
ARC = [0.5, 0.75, 1.0]
N_COLD = int(DATA["n_cold"])  # the bends a cold start can be trusted with come first
FIRST_AFTER = int(np.flatnonzero(~DATA["run_ok"])[-1]) + 1  # the first frame after the dropout
AGREE = 1e-8  # [m] in Δ, a millionth of a bend: the lab's closed form is good to 4e-10


@pytest.fixture
def inversion():
    return Inversion(helyx.arm("290-145-145"), ARC)


def warm_guesses(n):
    """Starts a few millimetres off the bends, as a filter's estimate would be."""
    return DATA["delta"][:n] + 2e-3 * np.random.default_rng(0).standard_normal((n, 9))


def test_from_the_neutral_configuration_the_inversion_finds_the_labs_delta(inversion):
    for markers, lab, bend in zip(
        DATA["markers"][:N_COLD], DATA["lab_delta"][:N_COLD], DATA["delta"][:N_COLD], strict=True
    ):
        inversion.reset()
        q = inversion(markers)
        assert inversion.converged
        np.testing.assert_allclose(q, lab, atol=AGREE)
        np.testing.assert_allclose(q, bend, atol=AGREE)
    assert np.abs(DATA["lab_delta"][:N_COLD]).max() > 0.01  # the bends are not small


def test_from_a_warm_start_it_finds_the_labs_delta_over_the_whole_range_of_bends(inversion):
    guesses = warm_guesses(len(DATA["delta"]))
    for markers, lab, q0 in zip(DATA["markers"], DATA["lab_delta"], guesses, strict=True):
        q = inversion(markers, q0=q0)
        assert inversion.converged
        np.testing.assert_allclose(q, lab, atol=AGREE)
    assert np.abs(DATA["lab_delta"][N_COLD:]).max() > 0.03  # up to 1.5 rad in a section


def test_noisy_markers_give_the_labs_delta_of_the_noisy_markers(inversion):
    lab_noisy = DATA["lab_noisy_delta"]
    assert np.abs(lab_noisy - DATA["lab_delta"]).max() > 1e-3  # the noise does move Δ
    for markers, lab in zip(DATA["noisy"][:N_COLD], lab_noisy[:N_COLD], strict=True):
        inversion.reset()
        np.testing.assert_allclose(inversion(markers), lab, atol=AGREE)
        assert inversion.converged
    guesses = warm_guesses(len(lab_noisy))
    for markers, lab, q0 in zip(DATA["noisy"], lab_noisy, guesses, strict=True):
        np.testing.assert_allclose(inversion(markers, q0=q0), lab, atol=AGREE)
        assert inversion.converged


def test_the_inversion_follows_the_run_frame_by_frame_and_starts_again_after_the_dropout(
    inversion,
):
    solved = 0
    for k, ok in enumerate(DATA["run_ok"]):
        if not ok:
            inversion.reset()  # the markers are gone: forget the warm start
            continue
        np.testing.assert_allclose(
            inversion(DATA["run_markers"][k]), DATA["run_delta"][k], atol=AGREE
        )
        assert inversion.converged
        solved += 1
    assert solved == DATA["run_ok"].sum() == 38


def test_the_filter_gives_the_labs_velocity_of_the_labs_delta_with_a_reset_at_the_dropout():
    for alpha, lab in zip(DATA["alphas"], DATA["run_delta_dot"], strict=True):
        vf = VelocityFilter(float(DATA["frame_dt"]), float(alpha))
        for k, ok in enumerate(DATA["run_ok"]):
            if not ok:
                vf.reset()
                continue
            np.testing.assert_allclose(
                vf.update(DATA["run_delta"][k]), lab[k], rtol=1e-12, atol=1e-14
            )
        assert np.nanmax(np.abs(lab)) > 0.05  # the velocities it agrees on are not small
        np.testing.assert_array_equal(lab[[0, FIRST_AFTER]], np.zeros((2, 9)))  # first samples


def test_the_inversion_and_the_filter_together_are_the_labs_velocity_estimator(inversion):
    for alpha, lab in zip(DATA["alphas"], DATA["run_delta_dot"], strict=True):
        vf = VelocityFilter(float(DATA["frame_dt"]), float(alpha))
        inversion.reset()
        for k, ok in enumerate(DATA["run_ok"]):
            if not ok:
                inversion.reset()
                vf.reset()
                continue
            q = inversion(DATA["run_markers"][k])
            np.testing.assert_allclose(q, DATA["run_delta"][k], atol=AGREE)
            np.testing.assert_allclose(vf.update(q), lab[k], atol=1e-7)  # 1/dt = 120 amplifies
