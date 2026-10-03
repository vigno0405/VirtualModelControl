"""Dissipation components: dampers with f(y, ẏ) and f·ẏ ≤ 0."""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...core.registry import register
from ...core.units import damping_unit
from ..coordinates.base import Context, Coordinate
from .base import Component, scaled


@register("component", "linear_damper")
class LinearDamper(Component):
    """Linear damper, f = −D ẏ.

    ``damping`` D is a scalar, one value per axis, or a matrix [N·s/m, or N·m·s/rad on angles].
    """

    kind = "dissipation"

    def __init__(self, coord: Coordinate, damping: Any) -> None:
        super().__init__(coord)
        self.damping = self._param("damping", damping, unit=damping_unit(coord.unit), scope="stage")

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−D ẏ."""
        return -scaled(ctx.param(self.damping), yd)


@register("component", "tanh_damper")
class TanhDamper(Component):
    """Saturating damper, per axis: f = −F tanh(d ẏ / F), within ±F on each axis.

    ``damping`` d is the slope at ẏ = 0, a scalar or one value per axis; ``max_force`` F [N].
    """

    kind = "dissipation"

    def __init__(self, coord: Coordinate, damping: Any, max_force: Any) -> None:
        super().__init__(coord)
        self.damping = self._param("damping", damping, unit=damping_unit(coord.unit), scope="stage")
        self.max_force = self._param("max_force", max_force, unit="N", scope="stage")

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−F tanh(d ẏ / F)."""
        d, F = ctx.param(self.damping), ctx.param(self.max_force)
        return -F * ca.tanh(d * yd / F)
