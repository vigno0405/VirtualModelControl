"""The window, driven offscreen with the events a person would make."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent

import virtualmodelcontrol as vmc
from virtualmodelcontrol.interactive import Controls, Interactive, Joystick, Window
from virtualmodelcontrol.interactive.window import _configuration

DT = 1 / 330
GOAL = "ctrl.reach.goal"
STIFFNESS = "ctrl.reach.stiffness"


def window_of(experiment, steps=5, **kwargs):
    """A window on the experiment's robot, after the controller has taken a few steps."""
    swaps = {k: c for k, c in experiment.controllers.items() if k != experiment.name}
    interactive = Interactive(experiment.controller, swaps=swaps, pausable=True)
    plant = experiment.plant
    plant.reset()
    interactive.reset(plant.t, plant.read())
    for _ in range(steps):
        plant.write(interactive.step(plant.t, plant.read()))
        plant.advance(DT)
    kwargs.setdefault("goals", [GOAL])
    kwargs.setdefault("sliders", {STIFFNESS: (0.0, 600.0)})
    window = Window(experiment.robot, interactive, **kwargs)
    window.refresh()
    window.fig.canvas.draw()  # the layout the events are placed in
    return window, interactive.controls


@pytest.fixture(autouse=True)
def close_windows():
    yield
    plt.close("all")


def pixels(window, u, v):
    return window.ax.transData.transform((u, v))


def mouse(window, name, x, y):
    event = MouseEvent(name, window.fig.canvas, x, y, button=1)
    window.fig.canvas.callbacks.process(name, event)


def click(window, widget):
    x, y = widget.ax.transAxes.transform((0.5, 0.5))
    mouse(window, "button_press_event", x, y)
    mouse(window, "button_release_event", x, y)


def test_it_draws_the_robot_and_the_goal_where_they_are(experiment):
    window, controls = window_of(experiment)
    goal = controls.values[GOAL]
    crosses = [line for line in window.ax.lines if line.get_marker() == "x"]
    assert len(crosses) == 1
    np.testing.assert_allclose(
        [crosses[0].get_xdata()[0], crosses[0].get_ydata()[0]], [goal[0], goal[2]]
    )
    assert window.ax.get_xlabel() and "t = " in window._status.get_text()
    assert "ctrl.reach.goal" in window._status.get_text()


def test_the_view_stays_where_it_was_first_set(experiment):
    window, _ = window_of(experiment)
    first = (window.ax.get_xlim(), window.ax.get_ylim())
    window.interactive.controller.controller.reset(0.0)  # anything: the picture is redrawn
    window.refresh()
    assert (window.ax.get_xlim(), window.ax.get_ylim()) == first
    (x0, x1), (y0, y1) = first
    assert x1 - x0 == pytest.approx(y1 - y0)  # square, so the drawing is not distorted


def test_sliders_follow_the_controls_and_drive_them(experiment):
    window, controls = window_of(experiment)
    controls.take()
    controls.set(STIFFNESS, 450.0)  # as a key or a joystick would
    controls.take()
    window.refresh()
    assert window.sliders[STIFFNESS].val == 450.0
    assert controls.take() == ({}, [])  # following is not a new change
    window.sliders[STIFFNESS].set_val(200.0)  # as a hand on the slider would
    assert controls.take()[0][STIFFNESS] == 200.0


def test_dragging_a_goal_moves_it_in_the_plane_and_leaves_the_third_component(experiment):
    window, controls = window_of(experiment)
    controls.set(GOAL, [0.02, 0.03, 0.43])
    window.refresh()
    window.fig.canvas.draw()
    x, y = pixels(window, 0.02, 0.43)
    mouse(window, "button_press_event", x + 3, y - 3)  # near the cross
    x2, y2 = pixels(window, 0.10, 0.45)
    mouse(window, "motion_notify_event", x2, y2)
    mouse(window, "button_release_event", x2, y2)
    np.testing.assert_allclose(controls.values[GOAL], [0.10, 0.03, 0.45], atol=1e-9)
    x3, y3 = pixels(window, 0.15, 0.50)
    mouse(window, "motion_notify_event", x3, y3)  # after the release nothing follows the mouse
    np.testing.assert_allclose(controls.values[GOAL], [0.10, 0.03, 0.45], atol=1e-9)


def test_a_press_away_from_every_goal_grabs_nothing(experiment):
    window, controls = window_of(experiment)
    before = controls.values[GOAL]
    x, y = pixels(window, *(before[[0, 2]] + [0.1, -0.1]))
    mouse(window, "button_press_event", x, y)
    x2, y2 = pixels(window, 0.0, 0.2)
    mouse(window, "motion_notify_event", x2, y2)
    np.testing.assert_array_equal(controls.values[GOAL], before)


def test_keys_reach_the_keyboard_and_matplotlibs_own_keys_are_off(experiment):
    window, controls = window_of(experiment)
    canvas = window.fig.canvas
    handlers = [ref().__name__ for ref in canvas.callbacks.callbacks["key_press_event"].values()]
    assert "key_press_handler" not in handlers and "_key" in handlers  # s would save, f maximize
    controls.take()
    for key in ("right", "pageup", "r"):
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key))
    pending = controls.take()[0][GOAL]
    np.testing.assert_allclose(pending, controls.values[GOAL])
    assert controls.recording
    np.testing.assert_allclose(pending - [0.0, 0.0, 0.435], [0.01, 0.01, 0.0], atol=1e-12)


def test_the_buttons_and_the_radio_buttons_act_on_the_controls(experiment):
    window, controls = window_of(experiment)
    controls.take()
    window.radio.set_active(1)  # the second controller
    assert controls.take()[1] == [("gentle", 1.0)]
    click(window, window.record)
    click(window, window.pause)
    assert controls.recording and controls.paused
    window.refresh()
    assert window.record.label.get_text() == "Stop" and window.pause.label.get_text() == "Resume"
    assert "recording" in window._status.get_text() and "paused" in window._status.get_text()
    click(window, window.record)
    assert not controls.recording


def test_a_window_without_a_pausable_simulation_has_no_pause_button(experiment):
    swaps = {"gentle": experiment.controllers["gentle"]}
    interactive = Interactive(experiment.controller, swaps=swaps)  # a robot's own node
    window = Window(experiment.robot, interactive, goals=[GOAL])
    assert window.pause is None and window.sliders == {}
    window.refresh()  # before the first step there is nothing to draw but the goal
    assert [line for line in window.ax.lines if line.get_marker() == "x"]


def test_closing_the_window_stops_the_run(experiment):
    window, controls = window_of(experiment)
    assert not controls.stopped
    canvas = window.fig.canvas  # a toolkit sends this when its window is closed
    canvas.callbacks.process("close_event", CloseEvent("close_event", canvas))
    assert controls.stopped


def test_a_joystick_is_polled_at_every_refresh(experiment):
    window, controls = window_of(experiment)

    class Device:
        def get_axis(self, i):
            return 1.0 if i == 0 else 0.0

        def get_button(self, i):
            return False

    window.joystick = Joystick(controls, GOAL, Device(), speed=0.2, deadzone=0.0)
    controls.take()
    window.refresh(0.5)
    np.testing.assert_allclose(controls.take()[0][GOAL] - [0.0, 0.0, 0.435], [0.1, 0.0, 0.0])


def test_the_configuration_comes_from_the_motors_when_the_plant_does_not_report_it(experiment):
    plant = experiment.plant
    plant.reset(([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0, 0.01], np.zeros(9)))
    meas = plant.read()
    bare = vmc.Signals(
        0.0, motor_position=meas["motor_position"], motor_velocity=meas["motor_velocity"]
    )
    np.testing.assert_allclose(_configuration(experiment.robot, bare), meas["q"], atol=1e-12)
    np.testing.assert_array_equal(_configuration(experiment.robot, meas), meas["q"])


def test_a_mailbox_of_another_origin_works_as_well():
    controls = Controls({"goal": [0.0, 0.0, 0.4]}, ["ctrl"])
    assert controls.values["goal"].shape == (3,)
