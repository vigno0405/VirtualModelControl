"""Signed distances from a point to surfaces: positive outside, negative when penetrating."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.params import Param, Scope, as_param
from ...core.symbolic import smooth_norm
from .base import Context, Coordinate


class PlaneDistance(Coordinate):
    """Signed distance [m] from a point to a plane: d = n̂ᵀ (p − p₀), positive on the normal's side.

    ``normal`` and ``origin`` (a point on the plane [m]) are ``episode`` Params.
    """

    def __init__(
        self,
        point: Coordinate,
        normal: Any,
        origin: Any = (0.0, 0.0, 0.0),
        scope: Scope = "episode",
    ) -> None:
        super().__init__(1, "m")
        self.point = point
        free = (-np.inf, np.inf)
        self.normal = as_param(normal, "normal", scope=scope, bounds=free)
        self.origin = as_param(origin, "origin", unit="m", scope=scope, bounds=free)

    def params(self) -> dict[str, Param]:
        """Normal and origin."""
        return {"normal": self.normal, "origin": self.origin}

    def children(self) -> tuple[Coordinate, ...]:
        """The point."""
        return (self.point,)

    def value(self, ctx: Context) -> Any:
        """n̂ᵀ (p − p₀)."""
        n = ca.reshape(ctx.param(self.normal), 3, 1)
        p0 = ca.reshape(ctx.param(self.origin), 3, 1)
        return ca.dot(n, ctx.value(self.point) - p0) / ca.norm_2(n)


class SphereDistance(Coordinate):
    """Signed distance [m] from a point to a sphere's surface: d = ‖p − c‖ − r, negative inside.

    ``center`` [m] and ``radius`` [m] are ``episode`` Params.
    """

    def __init__(
        self, point: Coordinate, center: Any, radius: Any, scope: Scope = "episode"
    ) -> None:
        super().__init__(1, "m")
        self.point = point
        self.center = as_param(center, "center", unit="m", scope=scope, bounds=(-np.inf, np.inf))
        self.radius = as_param(radius, "radius", unit="m", scope=scope, bounds=(0.0, np.inf))

    def params(self) -> dict[str, Param]:
        """Centre and radius."""
        return {"center": self.center, "radius": self.radius}

    def children(self) -> tuple[Coordinate, ...]:
        """The point."""
        return (self.point,)

    def value(self, ctx: Context) -> Any:
        """‖p − c‖ − r."""
        c = ca.reshape(ctx.param(self.center), 3, 1)
        return smooth_norm(ctx.value(self.point) - c) - ctx.param(self.radius)
