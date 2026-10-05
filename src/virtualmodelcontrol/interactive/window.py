"""Window: the robot drawn live, with goals to drag and controls for a running controller."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from ..core.params import constants
from ..core.signals import Signals
from ..models import Direct
from ..viz import draw_goal, draw_robot, label_axes, project
from .interactive import Interactive
from .joystick import Joystick
from .keyboard import AXES, Keyboard

GRAB = 15.0  # [px] how close to a goal a press grabs it
# matplotlib's own defaults, whatever style the script uses: the lab style's LaTeX text would be
# typeset again at every refresh.
LOOK: Any = {key: value for key, value in mpl.rcParamsDefault.items() if key != "backend"}
MARGIN = 0.6  # of the robot's extent, around it


class Window:
    """A matplotlib window for ``interactive``: ``robot`` drawn in ``plane`` at the latest
    measurement, the Params in ``goals`` (three components) as crosses to drag, a slider for each
    Param in ``sliders`` (name to (low, high)), buttons to swap controllers, to record and, in a
    simulation, to pause. The keys are those of ``Keyboard``; a ``joystick`` is polled at every
    refresh. Closing the window stops the run. ``show`` opens it; ``refresh`` redraws once, from
    the thread that owns the window.
    """

    def __init__(
        self,
        robot: Any,
        interactive: Interactive,
        *,
        goals: Sequence[str] = (),
        sliders: Mapping[str, tuple[float, float]] | None = None,
        plane: str = "xz",
        step: float = 0.01,
        swap_duration: float = 1.0,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        joystick: Joystick | None = None,
        period: float = 0.04,
    ) -> None:
        self.robot, self.interactive, self.controls = robot, interactive, interactive.controls
        self.goals, self.plane, self.joystick, self.period = list(goals), plane, joystick, period
        self.swap_duration = swap_duration
        self._limits: tuple[tuple[float, float], tuple[float, float]] | None = limits
        self._drag: str | None = None
        self.keyboard = (
            Keyboard(
                self.controls, self.goals[0], plane=plane, step=step, swap_duration=swap_duration
            )
            if self.goals
            else None
        )
        with mpl.rc_context(LOOK):
            self.fig = plt.figure(figsize=(9.0, 5.2))
            self._build(interactive, sliders or {})
        canvas = self.fig.canvas
        default_keys = getattr(canvas.manager, "key_press_handler_id", None)
        if default_keys is not None:  # matplotlib's own key bindings would clash with ours
            canvas.mpl_disconnect(default_keys)
        canvas.mpl_connect("key_press_event", self._key)
        canvas.mpl_connect("button_press_event", self._press)
        canvas.mpl_connect("motion_notify_event", self._move)
        canvas.mpl_connect("button_release_event", self._release)
        canvas.mpl_connect("close_event", lambda event: self.controls.stop())
        self._tick_time = time.monotonic()

    def _build(self, interactive: Interactive, sliders: Mapping[str, tuple[float, float]]) -> None:
        """The axes of the robot, the sliders, the controller buttons and the status line."""
        self.ax = self.fig.add_axes((0.08, 0.14, 0.54, 0.80))
        self._status = self.fig.text(0.08, 0.03, "", fontsize=9)
        values = self.controls.values
        self.sliders: dict[str, Slider] = {}
        for k, (name, (low, high)) in enumerate(sliders.items()):
            ax = self.fig.add_axes((0.72, 0.9 - 0.1 * k, 0.23, 0.03))
            ax.set_title(f"{name} [{interactive.unit(name)}]", loc="left", fontsize=9)
            slider = Slider(ax, "", low, high, valinit=float(values[name].ravel()[0]))

            def changed(value: float, name: str = name) -> None:
                self.controls.set(name, value)

            slider.on_changed(changed)
            self.sliders[name] = slider
        names = self.controls.controllers
        ax = self.fig.add_axes((0.70, 0.25, 0.25, 0.05 * len(names) + 0.02))
        self.radio = RadioButtons(ax, names, active=names.index(interactive.selected))
        self.radio.on_clicked(self._swap)
        ax = self.fig.add_axes((0.70, 0.1, 0.11, 0.07))
        self.record = Button(ax, "Record")
        self.record.on_clicked(lambda event: self.controls.toggle_recording())
        self.pause = None
        if interactive.pausable:
            ax = self.fig.add_axes((0.84, 0.1, 0.11, 0.07))
            self.pause = Button(ax, "Pause")
            self.pause.on_clicked(lambda event: self.controls.toggle_pause())

    def refresh(self, dt: float = 0.0) -> None:
        """Poll the joystick for ``dt`` [s], then redraw the robot, the goals and the controls."""
        if self.joystick is not None and dt > 0.0:
            self.joystick.poll(dt)
        values, last, ax = self.controls.values, self.interactive.last, self.ax
        with mpl.rc_context(LOOK):
            ax.cla()
            drawn = []
            if last is not None:
                q = _configuration(self.robot, last[1])
                drawn += draw_robot(ax, self.robot, q, plane=self.plane)
            drawn += [draw_goal(ax, values[name], plane=self.plane)[0] for name in self.goals]
            if self._limits is None and drawn:  # the view is fixed by the first picture
                x, y = (np.concatenate([np.ravel(a.get_data()[i]) for a in drawn]) for i in (0, 1))
                self._limits = _box((x, y))
            if self._limits is not None:
                (x0, x1), (y0, y1) = self._limits
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=9, labelfontfamily="sans-serif", width=0.8, length=3.5)
            label_axes(ax, self.plane)
        for name, slider in self.sliders.items():  # follow the keys and the joystick
            slider.eventson = False
            slider.set_val(float(values[name].ravel()[0]))
            slider.eventson = True
        self._status.set_text(self._state(last, values))
        self.record.label.set_text("Stop" if self.controls.recording else "Record")
        if self.pause is not None:
            self.pause.label.set_text("Resume" if self.controls.paused else "Pause")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the window and keep it up to date until it is closed."""
        self._timer = self.fig.canvas.new_timer(interval=int(1000 * self.period))
        self._timer.add_callback(self._tick)
        self._timer.start()
        plt.show()

    def _tick(self) -> None:
        now = time.monotonic()
        self.refresh(now - self._tick_time)
        self._tick_time = now

    def _state(self, last: Any, values: Mapping[str, np.ndarray]) -> str:
        parts = [] if last is None else [f"t = {last[0]:.2f} s"]
        parts += [f"{name} = {np.array2string(values[name], precision=3)}" for name in self.goals]
        parts += ["recording"] * self.controls.recording + ["paused"] * self.controls.paused
        return "   ".join(parts)

    def _swap(self, name: str | None) -> None:
        if name is not None:
            self.controls.swap(name, self.swap_duration)

    def _key(self, event: Any) -> None:
        if self.keyboard is not None:
            self.keyboard.press(event.key)

    def _press(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        values = self.controls.values
        for name in self.goals:
            u, v = (float(a[0]) for a in project(values[name], self.plane))
            x, y = self.ax.transData.transform((u, v))
            if np.hypot(event.x - x, event.y - y) <= GRAB:
                self._drag = name
                return

    def _move(self, event: Any) -> None:
        if self._drag is None or event.inaxes is not self.ax:
            return
        value = self.controls.values[self._drag]
        first, second = (AXES.index(c) for c in self.plane)
        value[first], value[second] = event.xdata, event.ydata
        self.controls.set(self._drag, value)

    def _release(self, event: Any) -> None:
        self._drag = None


def _configuration(robot: Any, meas: Signals) -> np.ndarray:
    """The robot's configuration at a measurement: ``q`` when the plant reports it, else through
    the transmission's inverse from the motor angles."""
    if "q" in meas:
        return meas["q"]
    actuation = robot.actuation if robot.actuation is not None else Direct()
    theta = ca.DM(meas["motor_position"])
    return np.array(
        ca.evalf(actuation.config_from_motors(theta, constants(actuation.params)))
    ).ravel()


def _box(span: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square axis limits around points seen in a plane, with a margin."""
    centre = [(a.min() + a.max()) / 2 for a in span]
    half = max(max(np.ptp(a) for a in span), 1e-3) / 2 * (1.0 + MARGIN)
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)
