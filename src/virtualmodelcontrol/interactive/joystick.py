"""Joystick: a game controller's stick as the speed of a goal."""

from __future__ import annotations

from typing import Any

import numpy as np

from .controls import Controls

AXES = "xyz"


def open_joystick(index: int = 0) -> Any:
    """The game controller number ``index`` through pygame, which ``pip install
    "virtualmodelcontrol[joystick]"`` adds. It has ``get_axis(i)`` and ``get_button(i)``."""
    try:
        import pygame
    except ImportError as error:
        raise ImportError(
            'a joystick needs pygame: pip install "virtualmodelcontrol[joystick]"'
        ) from error
    pygame.init()
    pygame.joystick.init()
    device = pygame.joystick.Joystick(index)
    device.init()

    class Device:
        def get_axis(self, i: int) -> float:
            pygame.event.pump()
            return float(device.get_axis(i))

        def get_button(self, i: int) -> bool:
            pygame.event.pump()
            return bool(device.get_button(i))

    return Device()


class Joystick:
    """Rate control of a goal Param of three components: the stick is the goal's velocity, up to
    ``speed`` [m/s], with a dead zone ``deadzone`` (a fraction of the stick's travel). The left
    stick (axes 0 and 1) moves it in ``plane``, the right stick's vertical axis (3) the third
    axis. A pressed button 0 toggles recording, buttons 4 and 5 swap to the previous and the next
    controller. ``poll(dt)`` reads ``device`` and moves the goal by ``dt`` [s] of motion."""

    def __init__(
        self,
        controls: Controls,
        goal: str,
        device: Any,
        *,
        plane: str = "xz",
        speed: float = 0.1,
        deadzone: float = 0.1,
        swap_duration: float = 1.0,
    ) -> None:
        self.controls, self.goal, self.device = controls, goal, device
        self.speed, self.deadzone, self.swap_duration = speed, deadzone, swap_duration
        first, second = (AXES.index(c) for c in plane)
        third = next(i for i in range(3) if i not in (first, second))
        self._axes = ((0, first, 1.0), (1, second, -1.0), (3, third, -1.0))  # stick up is -1
        self._down: set[int] = set()
        self._index = 0

    def poll(self, dt: float) -> None:
        """Move the goal for ``dt`` [s] and act on newly pressed buttons."""
        delta = np.zeros(3)
        for stick, axis, sign in self._axes:
            delta[axis] = sign * self._dead(self.device.get_axis(stick)) * self.speed * dt
        if delta.any():
            self.controls.nudge(self.goal, delta)
        for button, action in ((0, self._record), (4, self._previous), (5, self._next)):
            pressed = bool(self.device.get_button(button))
            if pressed and button not in self._down:
                action()
            if pressed:
                self._down.add(button)
            else:
                self._down.discard(button)

    def _dead(self, x: float) -> float:
        """The stick's deflection with the dead zone removed and the rest rescaled to 1."""
        if abs(x) <= self.deadzone:
            return 0.0
        return float(np.sign(x) * (abs(x) - self.deadzone) / (1.0 - self.deadzone))

    def _record(self) -> None:
        self.controls.toggle_recording()

    def _previous(self) -> None:
        self._swap(-1)

    def _next(self) -> None:
        self._swap(1)

    def _swap(self, step: int) -> None:
        names = self.controls.controllers
        if names:
            self._index = (self._index + step) % len(names)
            self.controls.swap(names[self._index], self.swap_duration)
