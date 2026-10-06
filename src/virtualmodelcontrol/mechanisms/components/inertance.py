"""Inertance components: masses and inertias, with kinetic energy ½ ẏᵀ M ẏ."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.registry import register
from ...core.units import KG, KG_M2, inertance_unit
from ..coordinates.base import Context, Coordinate
from ..coordinates.frames import FrameRotation
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


@register("component", "rotational_inertia")
class RotationalInertia(Component):
    """The rotational inertia of a rigid body about the origin of a frame: a 3 by 3 matrix in the
    frame's axes [kg·m²], or its three principal moments. Kinetic energy ½ tr(Ṙ Σ Ṙᵀ) with
    Σ = ½ tr(I) − I, so it acts on the rotation matrix of the frame (a ``FrameRotation``). With a
    ``PointMass`` at the same frame, it makes a rigid body.
    """

    kind = "inertance"

    def __init__(self, rotation: Coordinate, inertia: Any) -> None:
        if not isinstance(rotation, FrameRotation):
            raise ValueError("a rotational inertia acts on the FrameRotation of the body's frame")
        super().__init__(rotation)
        inertia = getattr(inertia, "value", inertia)
        if np.ndim(inertia) == 1:
            inertia = np.diag(inertia)
        free = (-np.inf, np.inf)
        self.inertia = self._param("inertia", inertia, unit=KG_M2, scope="design", bounds=free)

    def inertance(self, ctx: Context, y: Any) -> Any:
        """kron(1, Σ): Σ for each row of the rotation matrix."""
        I = ca.reshape(ctx.param(self.inertia), 3, 3)  # noqa: E741
        sigma = 0.5 * (I[0, 0] + I[1, 1] + I[2, 2]) * ca.DM.eye(3) - I
        return ca.kron(ca.DM.eye(3), sigma)
