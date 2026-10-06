"""Signed distances from a point to surfaces: positive outside, negative when penetrating."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.params import Param, Scope, as_param
from ...core.symbolic import smooth_norm
from .base import Context, Coordinate


class SurfaceDistance(Coordinate):
    """A signed distance [m] from a point to a surface; ``unit_normal`` is where it grows."""

    def __init__(self, point: Coordinate) -> None:
        super().__init__(1, "m")
        self.point = point

    def children(self) -> tuple[Coordinate, ...]:
        """The point."""
        return (self.point,)

    def surface(self, p: Any, ctx: Context) -> Any:
        """The distance of the point ``p`` (3, 1), as a function of the point."""
        raise NotImplementedError

    def value(self, ctx: Context) -> Any:
        """The distance of the point."""
        return self.surface(ctx.value(self.point), ctx)

    def unit_normal(self, ctx: Context) -> Any:
        """The unit vector (3, 1) along which the distance grows at the point."""
        s = ca.SX.sym("p", 3)
        gradient = ca.substitute(ca.gradient(self.surface(s, ctx), s), s, ctx.value(self.point))
        return gradient / ca.sqrt(ca.dot(gradient, gradient) + 1e-18)


class PlaneDistance(SurfaceDistance):
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
        super().__init__(point)
        free = (-np.inf, np.inf)
        self.normal = as_param(normal, "normal", scope=scope, bounds=free)
        self.origin = as_param(origin, "origin", unit="m", scope=scope, bounds=free)

    def params(self) -> dict[str, Param]:
        """Normal and origin."""
        return {"normal": self.normal, "origin": self.origin}

    def surface(self, p: Any, ctx: Context) -> Any:
        """n̂ᵀ (p − p₀)."""
        n = ca.reshape(ctx.param(self.normal), 3, 1)
        p0 = ca.reshape(ctx.param(self.origin), 3, 1)
        return ca.dot(n, p - p0) / ca.norm_2(n)


class SphereDistance(SurfaceDistance):
    """Signed distance [m] from a point to a sphere's surface: d = ‖p − c‖ − r, negative inside.

    ``center`` [m] and ``radius`` [m] are ``episode`` Params.
    """

    def __init__(
        self, point: Coordinate, center: Any, radius: Any, scope: Scope = "episode"
    ) -> None:
        super().__init__(point)
        self.center = as_param(center, "center", unit="m", scope=scope, bounds=(-np.inf, np.inf))
        self.radius = as_param(radius, "radius", unit="m", scope=scope, bounds=(0.0, np.inf))

    def params(self) -> dict[str, Param]:
        """Centre and radius."""
        return {"center": self.center, "radius": self.radius}

    def surface(self, p: Any, ctx: Context) -> Any:
        """‖p − c‖ − r."""
        c = ca.reshape(ctx.param(self.center), 3, 1)
        return smooth_norm(p - c) - ctx.param(self.radius)


class CapsuleDistance(SurfaceDistance):
    """Signed distance [m] from a point to a capsule's surface: the distance to the segment from
    ``a`` to ``b`` [m], less ``radius`` [m]; negative inside. A capsule with a = b is a sphere.
    """

    def __init__(
        self, point: Coordinate, a: Any, b: Any, radius: Any, scope: Scope = "episode"
    ) -> None:
        super().__init__(point)
        free = (-np.inf, np.inf)
        self.a = as_param(a, "a", unit="m", scope=scope, bounds=free)
        self.b = as_param(b, "b", unit="m", scope=scope, bounds=free)
        self.radius = as_param(radius, "radius", unit="m", scope=scope, bounds=(0.0, np.inf))

    def params(self) -> dict[str, Param]:
        """The segment's ends and the radius."""
        return {"a": self.a, "b": self.b, "radius": self.radius}

    def surface(self, p: Any, ctx: Context) -> Any:
        """‖p − (a + h (b − a))‖ − r, with h the closest point of the segment."""
        a = ca.reshape(ctx.param(self.a), 3, 1)
        ab = ca.reshape(ctx.param(self.b), 3, 1) - a
        h = ca.fmax(0.0, ca.fmin(1.0, ca.dot(p - a, ab) / ca.fmax(ca.dot(ab, ab), 1e-18)))
        return smooth_norm(p - a - h * ab) - ctx.param(self.radius)


class BoxDistance(SurfaceDistance):
    """Signed distance [m] from a point to the surface of a box with its sides along the base
    frame's axes: ``center`` [m] and ``half_sizes`` [m], the half of its three sides; negative
    inside."""

    def __init__(
        self, point: Coordinate, center: Any, half_sizes: Any, scope: Scope = "episode"
    ) -> None:
        super().__init__(point)
        self.center = as_param(center, "center", unit="m", scope=scope, bounds=(-np.inf, np.inf))
        self.half_sizes = as_param(
            half_sizes, "half_sizes", unit="m", scope=scope, bounds=(0.0, np.inf)
        )

    def params(self) -> dict[str, Param]:
        """Centre and half sides."""
        return {"center": self.center, "half_sizes": self.half_sizes}

    def surface(self, p: Any, ctx: Context) -> Any:
        """The distance to the box, or minus the distance to its nearest face from inside."""
        c = ca.reshape(ctx.param(self.center), 3, 1)
        h = ca.reshape(ctx.param(self.half_sizes), 3, 1)
        q = ca.fabs(p - c) - h
        return smooth_norm(ca.fmax(q, 0.0)) + ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0.0)


class CylinderDistance(SurfaceDistance):
    """Signed distance [m] from a point to the surface of a solid cylinder: ``center`` [m], the
    ``axis`` along its length (any length, a direction), ``radius`` [m] and ``half_height`` [m];
    negative inside."""

    def __init__(
        self,
        point: Coordinate,
        center: Any,
        axis: Any,
        radius: Any,
        half_height: Any,
        scope: Scope = "episode",
    ) -> None:
        super().__init__(point)
        free = (-np.inf, np.inf)
        self.center = as_param(center, "center", unit="m", scope=scope, bounds=free)
        self.axis = as_param(axis, "axis", scope=scope, bounds=free)
        self.radius = as_param(radius, "radius", unit="m", scope=scope, bounds=(0.0, np.inf))
        self.half_height = as_param(
            half_height, "half_height", unit="m", scope=scope, bounds=(0.0, np.inf)
        )

    def params(self) -> dict[str, Param]:
        """Centre, axis, radius and half height."""
        return {
            "center": self.center,
            "axis": self.axis,
            "radius": self.radius,
            "half_height": self.half_height,
        }

    def surface(self, p: Any, ctx: Context) -> Any:
        """The distance to the cylinder, or minus the distance to its nearest surface inside."""
        n = ca.reshape(ctx.param(self.axis), 3, 1)
        n = n / ca.norm_2(n)
        w = p - ca.reshape(ctx.param(self.center), 3, 1)
        z = ca.dot(w, n)
        d = ca.vertcat(
            smooth_norm(w - z * n) - ctx.param(self.radius),
            ca.fabs(z) - ctx.param(self.half_height),
        )
        return smooth_norm(ca.fmax(d, 0.0)) + ca.fmin(ca.fmax(d[0], d[1]), 0.0)
