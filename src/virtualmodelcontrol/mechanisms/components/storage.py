"""Storage components: springs with an energy V(y) ≥ 0 and the force f = −∂V/∂y.

The deflection is y = x − x_ref, so a spring pulls x towards x_ref. Gains are Params: ``stage``
scope (live) for stiffnesses and shapes, ``episode`` for limits and exponents.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.registry import register
from ...core.symbolic import logcosh, quad, smooth_norm
from ...core.units import stiffness_unit
from ..coordinates.base import Context, Coordinate
from .base import Component, scaled


@register("component", "linear_spring")
class LinearSpring(Component):
    """Linear spring, V = ½ yᵀ K y and f = −K y.

    ``stiffness`` K is a scalar, one value per axis, or a matrix (its symmetric part acts)
    [N/m, or N·m/rad on angles].
    """

    kind = "storage"

    def __init__(self, coord: Coordinate, stiffness: Any) -> None:
        super().__init__(coord)
        matrix = np.ndim(getattr(stiffness, "value", stiffness)) == 2
        self.stiffness = self._param(
            "stiffness",
            stiffness,
            unit=stiffness_unit(coord.unit),
            scope="stage",
            bounds=(-np.inf, np.inf) if matrix else (0.0, np.inf),
        )

    def energy(self, ctx: Context, y: Any) -> Any:
        """½ yᵀ K y."""
        return 0.5 * ca.dot(y, scaled(ctx.param(self.stiffness), y))

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−K y (symmetric part of K for a matrix)."""
        K = ctx.param(self.stiffness)
        if K.shape[1] == 1:
            return -K * y
        return -0.5 * (ca.mtimes(K, y) + ca.mtimes(K.T, y))


@register("component", "tanh_spring")
class TanhSpring(Component):
    """Saturating spring, per axis: f = −F tanh(k y / F), within ±F on each axis.

    ``stiffness`` k is the slope at y = 0, a scalar or one value per axis; ``max_force`` F [N].
    """

    kind = "storage"

    def __init__(self, coord: Coordinate, stiffness: Any, max_force: Any) -> None:
        super().__init__(coord)
        self.stiffness = self._param(
            "stiffness", stiffness, unit=stiffness_unit(coord.unit), scope="stage"
        )
        self.max_force = self._param("max_force", max_force, unit="N", scope="stage")

    def energy(self, ctx: Context, y: Any) -> Any:
        """Σ (F²/k) log cosh(k y / F); zero on axes with k = 0."""
        k = ctx.param(self.stiffness) * ca.DM.ones(self.coord.dim, 1)
        F = ctx.param(self.max_force)
        zero = ca.DM.zeros(self.coord.dim, 1)
        return ca.sum1(ca.if_else(k > 0, F**2 / k * logcosh(k * y / F), zero))

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−F tanh(k y / F)."""
        k, F = ctx.param(self.stiffness), ctx.param(self.max_force)
        return -F * ca.tanh(k * y / F)


@register("component", "gaussian_spring")
class GaussianSpring(Component):
    """Repulsive Gaussian, V = A σ² exp(−‖y‖²/2σ²) and f = A exp(−‖y‖²/2σ²) y.

    With y = x − x_obstacle it pushes x away from the obstacle. ``strength`` A [N/m], ``sigma``
    σ in the coordinate's unit.
    """

    kind = "storage"

    def __init__(self, coord: Coordinate, strength: Any, sigma: Any) -> None:
        super().__init__(coord)
        self.strength = self._param(
            "strength", strength, unit=stiffness_unit(coord.unit), scope="stage"
        )
        self.sigma = self._param("sigma", sigma, unit=coord.unit, scope="stage")

    def energy(self, ctx: Context, y: Any) -> Any:
        """A σ² exp(−‖y‖²/2σ²)."""
        A, s = ctx.param(self.strength), ctx.param(self.sigma)
        return A * s**2 * ca.exp(-ca.sumsqr(y) / (2 * s**2))

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """A exp(−‖y‖²/2σ²) y."""
        A, s = ctx.param(self.strength), ctx.param(self.sigma)
        return A * ca.exp(-ca.sumsqr(y) / (2 * s**2)) * y


@register("component", "sigmoid_spring")
class SigmoidSpring(Component):
    """Spring stiffening with distance: f = −k(d) y, k(d) = k_min + (k_max − k_min) σ(α(d − d₀)).

    d is each axis' magnitude (``element_wise``) or the norm ‖y‖. The energy has no closed form
    and is integrated by Gauss–Legendre quadrature with ``nodes`` points; the force is exact.
    """

    kind = "storage"

    def __init__(
        self,
        coord: Coordinate,
        k_min: Any,
        k_max: Any,
        threshold: Any,
        alpha: Any,
        *,
        element_wise: bool = True,
        nodes: int = 32,
    ) -> None:
        super().__init__(coord)
        unit = stiffness_unit(coord.unit)
        self.k_min = self._param("k_min", k_min, unit=unit, scope="stage")
        self.k_max = self._param("k_max", k_max, unit=unit, scope="stage")
        self.threshold = self._param("threshold", threshold, unit=coord.unit, scope="stage")
        inv = f"1/{coord.unit}" if coord.unit else ""
        self.alpha = self._param("alpha", alpha, unit=inv, scope="stage")
        self.element_wise = element_wise
        self.nodes = nodes

    def _stiffness(self, ctx: Context, d: Any) -> Any:
        k0, k1 = ctx.param(self.k_min), ctx.param(self.k_max)
        d0, a = ctx.param(self.threshold), ctx.param(self.alpha)
        return k0 + (k1 - k0) / (1 + ca.exp(-a * (d - d0)))

    def _distance(self, y: Any) -> Any:
        return ca.fabs(y) if self.element_wise else smooth_norm(y)

    def energy(self, ctx: Context, y: Any) -> Any:
        """Σ ∫₀^d k(r) r dr, by quadrature."""
        d = self._distance(y)
        integral = quad(lambda t: self._stiffness(ctx, d * t) * t, 0.0, 1.0, self.nodes)
        return ca.sum1(d**2 * integral)

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−k(d) y."""
        return -self._stiffness(ctx, self._distance(y)) * y


@register("component", "polynomial_spring")
class PolynomialSpring(Component):
    """Spring with stiffness k = K (d / d₀)^n: f = −k y, V = K d^(n+2) / ((n+2) d₀ⁿ).

    d is each axis' magnitude (``element_wise``) or the norm ‖y‖. ``stiffness`` K is the
    stiffness at d = d₀.
    """

    kind = "storage"

    def __init__(
        self,
        coord: Coordinate,
        stiffness: Any,
        order: Any,
        dist_norm: Any,
        *,
        element_wise: bool = True,
    ) -> None:
        super().__init__(coord)
        self.stiffness = self._param(
            "stiffness", stiffness, unit=stiffness_unit(coord.unit), scope="stage"
        )
        self.order = self._param("order", order, unit="", scope="episode")
        self.dist_norm = self._param("dist_norm", dist_norm, unit=coord.unit, scope="stage")
        self.element_wise = element_wise

    def _distance(self, y: Any) -> Any:
        return ca.fabs(y) if self.element_wise else smooth_norm(y)

    def energy(self, ctx: Context, y: Any) -> Any:
        """Σ K d^(n+2) / ((n+2) d₀ⁿ)."""
        K, n, d0 = ctx.param(self.stiffness), ctx.param(self.order), ctx.param(self.dist_norm)
        return ca.sum1(K * self._distance(y) ** (n + 2) / ((n + 2) * d0**n))

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """−K (d / d₀)ⁿ y."""
        K, n, d0 = ctx.param(self.stiffness), ctx.param(self.order), ctx.param(self.dist_norm)
        return -K * (self._distance(y) / d0) ** n * y


@register("component", "limit_spring")
class LimitSpring(Component):
    """Zero force inside [lower, upper] and linear outside, per axis (a joint-limit spring).

    V = ½ k (max(0, lower − y)² + max(0, y − upper)²).
    """

    kind = "storage"

    def __init__(self, coord: Coordinate, stiffness: Any, lower: Any, upper: Any) -> None:
        super().__init__(coord)
        self.stiffness = self._param(
            "stiffness", stiffness, unit=stiffness_unit(coord.unit), scope="stage"
        )
        free = (-np.inf, np.inf)
        self.lower = self._param("lower", lower, unit=coord.unit, scope="episode", bounds=free)
        self.upper = self._param("upper", upper, unit=coord.unit, scope="episode", bounds=free)

    def energy(self, ctx: Context, y: Any) -> Any:
        """½ k (max(0, lower − y)² + max(0, y − upper)²)."""
        k = ctx.param(self.stiffness)
        below = ca.fmax(0, ctx.param(self.lower) - y)
        above = ca.fmax(0, y - ctx.param(self.upper))
        return ca.sum1(0.5 * k * (below**2 + above**2))

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """k (max(0, lower − y) − max(0, y − upper))."""
        k = ctx.param(self.stiffness)
        below = ca.fmax(0, ctx.param(self.lower) - y)
        above = ca.fmax(0, y - ctx.param(self.upper))
        return k * (below - above)


@register("component", "gravity")
class Gravity(Component):
    """Gravity on a robot's point masses: V = −Σ m_i gᵀ p_i, f_i = m_i g (a physical component).

    Unlike a spring, V has no lower bound. ``gravity`` g [m/s², base frame] defaults to the
    robot's ``gravity`` Param.
    """

    kind = "storage"

    def __init__(self, robot: Any, gravity: Any = None) -> None:
        from ..coordinates.ops import Stack
        from .inertance import PointMass

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
            "gravity", gravity, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)
        )

    def _weights(self, ctx: Context) -> Any:
        g = ca.reshape(ctx.param(self.gravity), 3, 1)
        return ca.vertcat(*[ctx.param(m.mass) * g for m in self.masses])

    def energy(self, ctx: Context, y: Any) -> Any:
        """−Σ m_i gᵀ p_i."""
        return -ca.dot(self._weights(ctx), y)

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """m_i g on each mass, stacked."""
        return self._weights(ctx)
