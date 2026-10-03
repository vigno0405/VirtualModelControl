"""Component base class: something acting on one coordinate, of one of four kinds."""

from __future__ import annotations

from typing import Any, ClassVar

import casadi as ca
import numpy as np

from ...core.params import Param, Scope, as_param
from ..coordinates.base import Context, Coordinate

KINDS = ("storage", "dissipation", "inertance", "source")


def scaled(gain: Any, x: Any) -> Any:
    """Gain times x: a scalar or per-axis gain multiplies entrywise, a matrix multiplies."""
    return gain * x if gain.shape[1] == 1 else ca.mtimes(gain, x)


class Component:
    """Acts on one coordinate. ``kind`` is storage, dissipation, inertance or source.

    Storage has an energy V(y) and the force f = −∂V/∂y; dissipation has f(y, ẏ) with f·ẏ ≤ 0;
    inertance has an inertia M(y); a source has a metered force f.
    """

    kind: ClassVar[str] = ""

    def __init__(self, coord: Coordinate) -> None:
        self.coord = coord
        self._params: dict[str, Param] = {}

    def _param(
        self,
        name: str,
        value: Any,
        *,
        unit: str,
        scope: Scope,
        bounds: tuple[Any, Any] = (0.0, np.inf),
    ) -> Param:
        param = as_param(value, name, unit=unit, bounds=bounds, scope=scope)
        self._params[name] = param
        return param

    def params(self) -> dict[str, Param]:
        """Params owned by this component, by local name."""
        return dict(self._params)

    def energy(self, ctx: Context, y: Any) -> Any:
        """Stored energy V(y) [J] (storage components)."""
        raise NotImplementedError(f"{type(self).__name__} stores no energy")

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """Force on the coordinate, (dim, 1). Storage defaults to −∂V/∂y by differentiation."""
        if self.kind != "storage":
            raise NotImplementedError(f"{type(self).__name__} defines no force")
        if isinstance(y, ca.MX):
            ym = ca.MX.sym("y", self.coord.dim)
            return ca.substitute(-ca.gradient(self.energy(ctx, ym), ym), ym, y)
        ys = ca.SX.sym("y", self.coord.dim)
        return ca.substitute(-ca.gradient(self.energy(ctx, ys), ys), ys, ca.SX(y))

    def inertance(self, ctx: Context, y: Any) -> Any:
        """Inertia M(y), (dim, dim) (inertance components)."""
        raise NotImplementedError(f"{type(self).__name__} has no inertance")

    def __repr__(self) -> str:
        values = ", ".join(f"{k}={p.value.tolist()}" for k, p in self._params.items())
        return f"{type(self).__name__}({self.coord!r}, {values})"
