"""Transmission efficiency: the motor torque a robot receives, a polynomial of the commanded one."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.params import ParamSet, as_param
from ..core.registry import register
from ..core.units import NM


@register("efficiency", "polynomial")
class Efficiency:
    """Delivered motor torque as a polynomial of the commanded one, motor by motor:
    τ = c₁ u + c₂ u² + … + cₙ uⁿ.

    ``Efficiency(c1, c2, ...)``: each coefficient is one value for every motor or one per motor.
    ``Efficiency(1.0)``, the default of every transmission, delivers the commanded torque. The
    coefficients are ``design`` Params ``c1`` … ``cn``, to tune or to identify.
    """

    def __init__(self, *coefficients: Any) -> None:
        if not coefficients:
            raise ValueError("an efficiency needs at least its linear coefficient c1")
        self.params = ParamSet()
        for k, c in enumerate(coefficients, start=1):
            unit, bounds = ("", (0.0, 1.0)) if k == 1 else (f"({NM})^{1 - k}", (-np.inf, np.inf))
            self.params.add(as_param(c, f"c{k}", unit=unit, bounds=bounds, scope="design"), f"c{k}")

    @property
    def degree(self) -> int:
        """The highest power of the commanded torque."""
        return len(self.params)

    def delivered(self, u: Any, p: dict[str, Any]) -> Any:
        """Delivered torques for commanded ones ``u``, with the coefficients ``p`` by name."""
        return sum(p[f"c{k}"] * u**k for k in range(1, self.degree + 1))

    def __call__(self, u: ArrayLike) -> np.ndarray:
        """Delivered torques [N·m] at the current coefficients; motors along the last axis."""
        u = np.asarray(u, dtype=float)
        return np.asarray(sum(self.params[f"c{k}"].value * u**k for k in range(1, self.degree + 1)))

    def to_dict(self) -> dict[str, Any]:
        """The coefficients at their current values."""
        values = [p.value.tolist() for p in self.params.values()]
        return {"type": "polynomial", "coefficients": values}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Efficiency:
        """Inverse of ``to_dict``."""
        return cls(*data["coefficients"])


def as_efficiency(value: Any) -> Efficiency:
    """``value`` if it is an ``Efficiency``, else the efficiency with that linear coefficient."""
    if isinstance(value, Efficiency):
        return value
    if isinstance(value, dict):
        return Efficiency.from_dict(value)
    return Efficiency(value)


def coefficients(p: dict[str, Any], prefix: str = "efficiency") -> dict[str, Any]:
    """The efficiency's coefficients among a transmission's Params, by their own names."""
    start = f"{prefix}."
    return {name[len(start) :]: value for name, value in p.items() if name.startswith(start)}
