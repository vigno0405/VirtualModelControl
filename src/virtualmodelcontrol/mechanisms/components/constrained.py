"""Constrained elements: springs and dampers that act along one direction n only (a cart).

Each is its plain element on the projection of the coordinate on n, s = n̂ᵀ y, so its force is
n̂ times the plain element's force on s: the plane normal to n is left free. ``normal`` n is an
``episode`` Param, normalized when used.
"""

from __future__ import annotations

from typing import Any

from ...core.params import Param, as_param
from ...core.registry import register
from ..coordinates.base import Coordinate
from ..coordinates.ops import Projection
from .dissipation import LinearDamper, TanhDamper
from .storage import GaussianSpring, LinearSpring, TanhSpring


class _AlongNormal(Projection):
    """s = n̂ᵀ y, with n the element's ``normal`` Param (named as the element takes it)."""

    def __init__(self, coord: Coordinate, normal: Any) -> None:
        super().__init__(coord, as_param(normal, "normal", scope="episode"))

    def params(self) -> dict[str, Param]:
        """The normal."""
        return {"normal": self.direction}


@register("component", "constrained_linear_spring")
class ConstrainedLinearSpring(LinearSpring):
    """Linear spring along ``normal`` only: f = −k n̂ n̂ᵀ y, ``stiffness`` k."""

    def __init__(self, coord: Coordinate, stiffness: Any, normal: Any) -> None:
        super().__init__(_AlongNormal(coord, normal), stiffness)


@register("component", "constrained_tanh_spring")
class ConstrainedTanhSpring(TanhSpring):
    """Saturating spring along ``normal`` only: f = −n̂ F tanh(k n̂ᵀ y / F), within ±F."""

    def __init__(self, coord: Coordinate, stiffness: Any, max_force: Any, normal: Any) -> None:
        super().__init__(_AlongNormal(coord, normal), stiffness, max_force)


@register("component", "constrained_gaussian_spring")
class ConstrainedGaussianSpring(GaussianSpring):
    """Repulsive Gaussian along ``normal`` only: f = n̂ A exp(−s²/2σ²) s with s = n̂ᵀ y."""

    def __init__(self, coord: Coordinate, strength: Any, sigma: Any, normal: Any) -> None:
        super().__init__(_AlongNormal(coord, normal), strength, sigma)


@register("component", "constrained_linear_damper")
class ConstrainedLinearDamper(LinearDamper):
    """Linear damper along ``normal`` only: f = −d n̂ n̂ᵀ ẏ, ``damping`` d."""

    def __init__(self, coord: Coordinate, damping: Any, normal: Any) -> None:
        super().__init__(_AlongNormal(coord, normal), damping)


@register("component", "constrained_tanh_damper")
class ConstrainedTanhDamper(TanhDamper):
    """Saturating damper along ``normal`` only: f = −n̂ F tanh(d n̂ᵀ ẏ / F), within ±F."""

    def __init__(self, coord: Coordinate, damping: Any, max_force: Any, normal: Any) -> None:
        super().__init__(_AlongNormal(coord, normal), damping, max_force)
