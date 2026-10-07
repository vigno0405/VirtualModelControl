"""Extremum seeking: the live Params that lower a measured cost, found by dithering them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .limits import admissible, live_matching


class DitherSeeking:
    """Gradient descent of a measured cost on live Params, from the cost alone.

    Each Param is held at its estimate θ̂ plus a dither a sin(ω t), every one at its own frequency.
    The correlation of the cost with sin(ω t) over the last ``window`` seconds is the cost's slope
    with respect to that Param, 2/a times the mean of (J − J̄) sin(ω t), and θ̂ moves down it by
    ``gain`` times the slope, at most ``max_rate`` per second, within the Param's bounds. Take
    ω = 2π n / ``window`` with a different whole n for each Param, which makes the tones orthogonal
    over the window, so that one Param's slope does not leak into another's; and slow against the
    cost's own rhythm.

    The slope arrives about half a window late, so keep ``gain`` times the cost's curvature times
    the window well under 1, or the estimate overshoots and rings.

    ``params`` are scalar live Params (names or globs, in the order of the arguments below);
    ``amplitude``, ``frequency`` [rad/s], ``gain`` and ``max_rate`` are one value or one per Param.
    Run it on a ``Tank`` and the changes are paid from its energy, which is the passivity cap.
    """

    def __init__(
        self,
        controller: Any,
        params: str | Sequence[str],
        amplitude: ArrayLike,
        frequency: ArrayLike,
        gain: ArrayLike,
        window: float,
        max_rate: ArrayLike = np.inf,
    ) -> None:
        compiled = controller.compiled
        names = [
            live_matching(compiled, p)[0] for p in ([params] if isinstance(params, str) else params)
        ]
        n = len(names)
        self.names = names
        self._params = {name: compiled.params[name] for name in names}
        for name, param in self._params.items():
            if param.size != 1:
                raise ValueError(f"{name} has {param.size} entries: the dither takes scalar Params")
        self.amplitude, self.frequency, self.gain, self.max_rate = (
            np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()
            for x in (amplitude, frequency, gain, max_rate)
        )
        self.window = float(window)
        live = controller.live_params()
        self.estimate = np.array([float(np.ravel(live[name])[0]) for name in names])
        """θ̂, the center of each Param's dither."""
        self.slope = np.zeros(n)
        """The last estimate of the cost's slope with respect to each Param."""
        self._t: list[float] = []
        self._cost: list[float] = []

    def step(self, controller: Any, cost: float, t: float) -> float:
        """Take the cost measured at time ``t``, move the estimates and set the dithered values on
        ``controller`` (or a ``Tank`` around it). Returns the energy jump applied [J]."""
        self._t.append(float(t))
        self._cost.append(float(cost))
        while self._t[-1] - self._t[0] > self.window:
            self._t.pop(0)
            self._cost.pop(0)
        if len(self._t) > 2 and self._t[-1] - self._t[0] > 0.9 * self.window:
            times, costs = np.array(self._t), np.array(self._cost)
            weights = np.gradient(times)
            centred = costs - np.sum(costs * weights) / np.sum(weights)
            probes = np.sin(np.outer(self.frequency, times))
            self.slope = 2.0 / self.amplitude * (probes @ (centred * weights)) / np.sum(weights)
            dt = times[-1] - times[-2]
            rate = np.clip(self.gain * self.slope, -self.max_rate, self.max_rate)
            for i, name in enumerate(self.names):
                moved = self.estimate[i] - rate[i] * dt
                self.estimate[i] = admissible(self._params[name], np.array([moved]))[0]
        values = self.estimate + self.amplitude * np.sin(self.frequency * t)
        new = {
            name: admissible(self._params[name], np.array([value]))
            for name, value in zip(self.names, values, strict=True)
        }
        return float(controller.set(new))
