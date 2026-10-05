"""Keyboard: keys that move a goal and toggle recording, pausing and swapping."""

from __future__ import annotations

import numpy as np

from .controls import Controls

AXES = "xyz"


class Keyboard:
    """Keys for a goal Param of three components: the arrows move it in ``plane`` by ``step`` [m],
    PageUp and PageDown along the third axis, ``[`` and ``]`` halve and double the step, space
    pauses, ``r`` records and ``1`` to ``9`` swap to the n-th controller. ``press`` takes a key
    as matplotlib names it and returns whether it was used."""

    def __init__(
        self,
        controls: Controls,
        goal: str,
        *,
        plane: str = "xz",
        step: float = 0.01,
        swap_duration: float = 1.0,
    ) -> None:
        self.controls, self.goal, self.step, self.swap_duration = (
            controls,
            goal,
            step,
            swap_duration,
        )
        first, second = (AXES.index(c) for c in plane)
        third = next(i for i in range(3) if i not in (first, second))
        self._moves = {
            "left": (first, -1),
            "right": (first, 1),
            "down": (second, -1),
            "up": (second, 1),
            "pagedown": (third, -1),
            "pageup": (third, 1),
        }

    def press(self, key: str) -> bool:
        """Do what ``key`` stands for; False if it stands for nothing."""
        controls = self.controls
        if key in self._moves:
            axis, sign = self._moves[key]
            delta = np.zeros(3)
            delta[axis] = sign * self.step
            controls.nudge(self.goal, delta)
        elif key == "[":
            self.step /= 2
        elif key == "]":
            self.step *= 2
        elif key == " ":
            controls.toggle_pause()
        elif key == "r":
            controls.toggle_recording()
        elif key.isdigit() and 0 < int(key) <= len(controls.controllers):
            controls.swap(controls.controllers[int(key) - 1], self.swap_duration)
        else:
            return False
        return True
