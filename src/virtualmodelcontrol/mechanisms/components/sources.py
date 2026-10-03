"""Source components: forces whose power is metered in the energy accounting."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.registry import register
from ...core.units import M_S2, damping_unit, force_unit
from ..coordinates.base import Context, Coordinate
from ..coordinates.ops import Stack
from .base import Component
from .inertance import PointMass


@register("component", "force_source")
class ForceSource(Component):
    """A force f on a coordinate, held in a live Param; its power f·ẏ is metered."""

    kind = "source"

    def __init__(self, coord: Coordinate, force: Any) -> None:
        super().__init__(coord)
        free = (-np.inf, np.inf)
        self.force_value = self._param(
            "force", force, unit=force_unit(coord.unit), scope="stage", bounds=free
        )

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """The Param, as a column."""
        return ca.reshape(ctx.param(self.force_value), self.coord.dim, 1)


@register("component", "speed_regulator")
class SpeedRegulator(Component):
    """Drives a coordinate towards a commanded speed: f = b (ω_cmd(t) − ẏ), a metered source.

    ``speed`` ω [coordinate unit/s] is reached through a linear ramp of ``ramp_time`` [s] from
    the controller's start; ``gain`` b in force per unit speed.
    """

    kind = "source"

    def __init__(self, coord: Coordinate, gain: Any, speed: Any, ramp_time: Any = 0.0) -> None:
        super().__init__(coord)
        self.gain = self._param("gain", gain, unit=damping_unit(coord.unit), scope="stage")
        free = (-np.inf, np.inf)
        self.speed = self._param("speed", speed, unit=f"{coord.unit}/s", scope="stage", bounds=free)
        self.ramp_time = self._param("ramp_time", ramp_time, unit="s", scope="episode")

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """b (ω ramp(t) − ẏ)."""
        T = ctx.param(self.ramp_time)
        ramp = ca.if_else(T > 0, ca.fmin(ctx.t / ca.fmax(T, 1e-12), 1.0), 1.0)
        return ctx.param(self.gain) * (ctx.param(self.speed) * ramp - yd)


@register("component", "gravity_compensation")
class GravityCompensation(Component):
    """Forces cancelling gravity on a robot's point masses, f_i = −m_i g (a metered source).

    ``gravity`` g is a vector in the robot's base frame [m/s²]; by default the robot's own
    ``gravity`` Param, shared so both always agree.
    """

    kind = "source"

    def __init__(self, robot: Any, gravity: Any = None) -> None:
        masses = [c for c in robot.components.values() if isinstance(c, PointMass)]
        if not masses:
            raise ValueError(f"robot {robot.name!r} has no PointMass components")
        super().__init__(Stack(*[m.coord for m in masses]))
        self.masses = masses
        if gravity is None:
            if "gravity" not in robot.params:
                raise ValueError(f"robot {robot.name!r} has no 'gravity' Param; pass gravity=...")
            gravity = robot.params["gravity"]
        self.gravity = self._param(
            "gravity", gravity, unit=M_S2, scope="design", bounds=(-np.inf, np.inf)
        )

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−m_i g on each mass, stacked."""
        g = ca.reshape(ctx.param(self.gravity), 3, 1)
        return ca.vertcat(*[-ctx.param(m.mass) * g for m in self.masses])
