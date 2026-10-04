"""Smooth element swaps: one set of virtual elements replaces another with a quintic blend.

The old torque fades out while the new one fades in over ``duration`` [s],

    u = (1 - w) u_old + w u_new,   w = 10 s^3 - 15 s^4 + 6 s^5,   s = t / duration,

so the torque and its first two time derivatives stay continuous. A planner uses the same
``blend_weight``, so it plans what the controller executes.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from ..core.signals import Signals


def blend_weight(t: float, duration: float) -> float:
    """w(t) in [0, 1]: 0 at t <= 0, 1 at t >= ``duration`` (and at once for a zero duration)."""
    if duration <= 0.0:
        return 1.0
    s = min(max(t / duration, 0.0), 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


class SwapController:
    """A controller whose virtual elements can be swapped for another controller's, smoothly.

    ``swap(new, duration)`` starts a blend at the next step: both controllers run, and the
    torques mix with ``blend_weight`` until the new controller takes over. A swap asked for
    during a blend starts when it ends (the latest request wins). Everything else (``params``,
    ``set``, ``z``, ``energy``) is the current controller's.
    """

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self._incoming: list[Any] | None = None  # [controller, duration, start]
        self._queued: tuple[Any, float] | None = None

    @property
    def blending(self) -> bool:
        """True while a swap is under way."""
        return self._incoming is not None

    def swap(self, controller: Any, duration: float) -> None:
        """Replace the current controller by ``controller`` over ``duration`` [s]."""
        if self._incoming is None:
            self._incoming = [controller, duration, None]
        else:
            self._queued = (controller, duration)

    def reset(self, t: float, meas: Signals | None = None, z0: ArrayLike | None = None) -> None:
        """Reset the current controller (a swap under way carries on)."""
        self.controller.reset(t, meas, z0)

    def step(self, t: float, meas: Signals) -> Signals:
        """One control step; during a swap, the blend of both controllers' torques."""
        out = self.controller.step(t, meas)
        if self._incoming is None:
            return out
        new, duration, start = self._incoming
        if start is None:
            start = self._incoming[2] = t
            new.reset(t, meas)
        w = blend_weight(t - start, duration)
        incoming = new.step(t, meas)
        mixed = {
            name: (1.0 - w) * out[name] + w * incoming[name]
            for name in out.names
            if name in incoming.names
        }
        if w >= 1.0:
            self.controller, self._incoming = new, None
            if self._queued is not None:
                self._incoming, self._queued = [*self._queued, None], None
        return Signals(t, **mixed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)
