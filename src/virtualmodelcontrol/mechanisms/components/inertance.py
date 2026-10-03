"""Inertance components: masses and inertias, with kinetic energy ½ ẏᵀ M ẏ."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.registry import register
from ...core.units import KG, inertance_unit
from ..coordinates.base import Context, Coordinate
from .base import Component


@register("component", "point_mass")
class PointMass(Component):
    """A mass m [kg] at a point coordinate; a ``design`` Param (a physical property)."""

    kind = "inertance"

    def __init__(self, coord: Coordinate, mass: Any) -> None:
        super().__init__(coord)
        self.mass = self._param("mass", mass, unit=KG, scope="design")

    def inertance(self, ctx: Context, y: Any) -> Any:
        """m I."""
        return ctx.param(self.mass) * ca.DM.eye(self.coord.dim)


@register("component", "inertance")
class Inertance(Component):
    """Constant inertance M on a coordinate: a scalar, one value per axis, or a matrix.

    An ``episode`` Param: a virtual flywheel's inertia, say [kg·m² on angles].
    """

    kind = "inertance"

    def __init__(self, coord: Coordinate, inertance: Any) -> None:
        super().__init__(coord)
        matrix = np.ndim(getattr(inertance, "value", inertance)) == 2
        self.inertia = self._param(
            "inertance",
            inertance,
            unit=inertance_unit(coord.unit),
            scope="episode",
            bounds=(-np.inf, np.inf) if matrix else (0.0, np.inf),
        )

    def inertance(self, ctx: Context, y: Any) -> Any:
        """M as a (dim, dim) matrix."""
        M = ctx.param(self.inertia)
        if M.shape[1] == 1:
            return ca.diag(M * ca.DM.ones(self.coord.dim, 1))
        return M
