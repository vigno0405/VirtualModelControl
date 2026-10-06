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


@register("component", "diode_damper")
class DiodeDamper(Component):
    """A damper that acts in one direction of motion only: f = −D gate(σ ẏ) ẏ per axis.

    ``sign`` σ is +1 to damp positive rates and −1 to damp negative ones; the gate is a step, or
    a smooth one over ``smoothing`` [coordinate unit/s]. It never adds energy.
    """

    kind = "dissipation"

    def __init__(
        self, coord: Coordinate, damping: Any, sign: Any = 1.0, smoothing: Any = 0.0
    ) -> None:
        super().__init__(coord)
        self.damping = self._param("damping", damping, unit=damping_unit(coord.unit), scope="stage")
        self.sign = self._param("sign", sign, unit="", scope="episode", bounds=(-1.0, 1.0))
        self.smoothing = self._param(
            "smoothing", smoothing, unit=f"{coord.unit}/s", scope="episode"
        )

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−D gate(σ ẏ) ẏ."""
        w, x = ctx.param(self.smoothing), ctx.param(self.sign) * yd
        smooth = 0.5 * (1 + ca.tanh(x / (2 * ca.fmax(w, 1e-300))))
        gate = ca.if_else(w > 0, smooth, ca.if_else(x > 0, 1.0, 0.0))
        return -scaled(ctx.param(self.damping), gate * yd)


@register("component", "contact_damper")
class ContactDamper(Component):
    """Damper active only in contact (d < 0) on a signed distance d: f = −D σ(−d/w) ḋ.

    ``damping`` D [N·s/m]; the gate σ switches over ``smoothing`` w [m] (a step for w = 0).
    """

    kind = "dissipation"

    def __init__(self, coord: Coordinate, damping: Any, smoothing: Any = 0.0) -> None:
        super().__init__(coord)
        self.damping = self._param("damping", damping, unit=damping_unit(coord.unit), scope="stage")
        self.smoothing = self._param("smoothing", smoothing, unit=coord.unit, scope="episode")

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−D gate(d) ḋ."""
        w = ctx.param(self.smoothing)
        smooth = 0.5 * (1 - ca.tanh(y / (2 * ca.fmax(w, 1e-300))))  # σ(−d/w), no overflow
        gate = ca.if_else(w > 0, smooth, ca.if_else(y < 0, 1.0, 0.0))
        return -ctx.param(self.damping) * gate * yd
