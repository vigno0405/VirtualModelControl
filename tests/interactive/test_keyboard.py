"""Keys that move a goal and toggle recording, pausing and swapping."""

import numpy as np
import pytest

from virtualmodelcontrol.interactive import Controls, Keyboard


def keyboard(plane="xz", step=0.01):
    controls = Controls({"goal": [0.0, 0.0, 0.4]}, ["ctrl", "gentle", "stiff"])
    return Keyboard(controls, "goal", plane=plane, step=step), controls


def goal(controls):
    return controls.values["goal"]


@pytest.mark.parametrize(
    "plane, keys, expected",
    [
        ("xz", ["right", "right", "up"], [0.02, 0.0, 0.41]),
        ("xz", ["left", "down", "pageup"], [-0.01, 0.01, 0.39]),
        ("yz", ["right", "up", "pagedown"], [-0.01, 0.01, 0.41]),
        ("xy", ["right", "up", "pageup"], [0.01, 0.01, 0.41]),
    ],
)
def test_arrows_move_the_goal_in_the_plane_shown_and_page_keys_along_the_third_axis(
    plane, keys, expected
):
    pad, controls = keyboard(plane)
    # Left/right and up/down are the plane's first and second axes; PageUp/PageDown the other.
    for key in keys:
        assert pad.press(key)
    np.testing.assert_allclose(goal(controls), expected, atol=1e-12)


def test_the_step_is_halved_and_doubled():
    pad, controls = keyboard(step=0.01)
    pad.press("]")
    pad.press("right")
    pad.press("[")
    pad.press("[")
    pad.press("right")
    assert pad.step == pytest.approx(0.005)
    np.testing.assert_allclose(goal(controls)[0], 0.02 + 0.005)


def test_space_pauses_r_records_and_digits_swap():
    pad, controls = keyboard()
    pad.press(" ")
    pad.press("r")
    assert controls.paused and controls.recording
    pad.press(" ")
    assert not controls.paused
    assert pad.press("2") and pad.press("3")
    assert controls.take()[1] == [("gentle", 1.0), ("stiff", 1.0)]


def test_keys_that_stand_for_nothing_are_not_used():
    pad, controls = keyboard()
    assert not pad.press("x") and not pad.press("4") and not pad.press("0")  # only 3 controllers
    assert controls.take() == ({}, [])
