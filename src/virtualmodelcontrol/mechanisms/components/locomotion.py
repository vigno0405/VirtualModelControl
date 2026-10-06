"""Locomotion components: springs whose stiffness follows the phase of a gait."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.registry import register
from ...core.units import RAD, stiffness_unit
from ..coordinates.base import Context, Coordinate
from .base import Component


@register("component", "phase_spring")
class PhaseSpring(Component):
    """A spring on a deflection e whose stiffness follows a phase φ, on the coordinate (e, φ).

    V = K (1 + side steer) [ψ(e) + depth cos(φ − peak) ψ_b(e)], with ψ = ψ_b = e²/2. With
    ``limit`` e_max [rad] both saturate, so the force never exceeds K (1 + |steer|) e_max.
    Its force is −∂V/∂(e, φ): the φ entry is the reaction that keeps the loop passive.
    """

    kind = "storage"

    def __init__(
        self,
        coord: Coordinate,
        stiffness: Any,
        depth: Any = 0.0,
        peak: Any = 0.0,
        steer: Any = 0.0,
        side: Any = 1.0,
        limit: Any = None,
    ) -> None:
        super().__init__(coord)
        self.stiffness = self._param(
            "stiffness", stiffness, unit=stiffness_unit(coord.unit), scope="stage"
        )
        self.depth = self._param("depth", depth, unit="", scope="stage", bounds=(0.0, 0.9))
        self.peak = self._param("peak", peak, unit=RAD, scope="stage", bounds=(-np.inf, np.inf))
        self.steer = self._param("steer", steer, unit="", scope="stage", bounds=(-1.0, 1.0))
        self.side = self._param("side", side, unit="", scope="design", bounds=(-1.0, 1.0))
        self.limit = (
            None if limit is None else self._param("limit", limit, unit=coord.unit, scope="episode")
        )

    def energy(self, ctx: Context, y: Any) -> Any:
        """K (1 + side steer) [ψ(e) + depth cos(φ − peak) ψ_b(e)]."""
        e, phase = y[0], y[1]
        K = ctx.param(self.stiffness) * (1 + ctx.param(self.side) * ctx.param(self.steer))
        wave = ctx.param(self.depth) * ca.cos(phase - ctx.param(self.peak))
        if self.limit is None:
            return K * 0.5 * e**2 * (1 + wave)
        e_max2 = ctx.param(self.limit) ** 2
        psi = e_max2 * (ca.sqrt(1 + e**2 / e_max2) - 1)
        psi_b = 0.5 * e**2 * e_max2 / (e**2 + e_max2)
        return K * (psi + wave * psi_b)
