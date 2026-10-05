"""A game controller's stick as the speed of a goal, tested with a stand-in device."""

import sys

import numpy as np
import pytest

from virtualmodelcontrol.interactive import Controls, Joystick, open_joystick


class Device:
    """Sticks and buttons set by the test."""

    def __init__(self):
        self.axes, self.buttons = {}, set()

    def get_axis(self, i):
        return self.axes.get(i, 0.0)

    def get_button(self, i):
        return i in self.buttons


def joystick(**kwargs):
    controls = Controls({"goal": [0.0, 0.0, 0.4]}, ["ctrl", "gentle"])
    device = Device()
    return Joystick(controls, "goal", device, **kwargs), controls, device


def test_the_stick_is_the_goals_speed_with_a_dead_zone():
    stick, controls, device = joystick(speed=0.2, deadzone=0.1)
    device.axes = {0: 0.05, 1: -0.05}  # inside the dead zone: nothing moves
    stick.poll(0.1)
    assert controls.take() == ({}, [])
    device.axes = {0: 1.0, 1: -1.0}  # full right, full up (up is negative on a stick)
    stick.poll(0.1)
    np.testing.assert_allclose(controls.take()[0]["goal"], [0.02, 0.0, 0.42])
    device.axes = {0: 0.55}  # half of what is left of the travel after the dead zone
    stick.poll(0.5)
    np.testing.assert_allclose(controls.take()[0]["goal"], [0.02 + 0.2 * 0.5 * 0.5, 0.0, 0.42])


def test_the_right_stick_moves_the_third_axis_and_the_plane_is_a_choice():
    stick, controls, device = joystick(plane="yz", speed=1.0)
    device.axes = {0: 1.0, 3: -1.0}
    stick.poll(0.01)
    np.testing.assert_allclose(controls.take()[0]["goal"], [0.01, 0.01, 0.4], atol=1e-12)


def test_buttons_act_once_when_pressed_not_while_held():
    stick, controls, device = joystick()
    device.buttons = {0}
    for _ in range(3):  # held down for three polls
        stick.poll(0.01)
    assert controls.recording
    device.buttons = set()
    stick.poll(0.01)
    device.buttons = {0}
    stick.poll(0.01)
    assert not controls.recording  # pressed again: toggled again


def test_the_bumpers_swap_to_the_next_and_previous_controller():
    stick, controls, device = joystick()
    for pressed in ({5}, set(), {5}, set(), {4}):
        device.buttons = pressed
        stick.poll(0.01)
    assert [name for name, _ in controls.take()[1]] == ["gentle", "ctrl", "gentle"]


def test_without_pygame_opening_a_joystick_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", None)  # as if it were not installed
    with pytest.raises(ImportError, match=r"virtualmodelcontrol\[joystick\]"):
        open_joystick()
