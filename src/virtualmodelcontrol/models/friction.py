"""Static friction of an actuator: the torque it takes, a smooth function of the applied one."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import ParamSet, as_param
from ..core.registry import register
from ..core.units import NM


@register("friction", "static")
class StaticFriction:
    """The torque τ_f that friction takes from an actuator, as a function of the applied one u.

    ``StaticFriction(breakaway, kinetic, width)`` [N·m], each one value for every motor or one
    per motor. Below the ``breakaway`` torque F the actuator is stuck: friction takes the whole of
    u. Past it the actuator moves and friction takes the ``kinetic`` torque Fc, with the sign of
    u: Fc = 0 when static friction is gone once it moves, Fc = F when friction stays. Around
    |u| = F the change is smooth over ``width`` w > 0, and the whole map is smooth (C^∞), so a
    solver can differentiate it:

        τ_f = G sat_F(u) + (1 − G) Fc u / a,  a = √(u² + w²),  G = ½ (1 − tanh((a − F) / w)),

    with sat_F(u) = ½ (√((u + F)² + w²) − √((u − F)² + w²)), the torque clipped to ±F. It depends
    on the torque only, not on the speed. With F = Fc = 0, the default, τ_f = 0: no friction. The
    parameters are ``design`` Params, to tune or to identify.
    """

    def __init__(self, breakaway: Any = 0.0, kinetic: Any = 0.0, width: Any = 0.01) -> None:
        self.params = ParamSet()
        for name, value in (("breakaway", breakaway), ("kinetic", kinetic), ("width", width)):
            param = as_param(value, name, unit=NM, bounds=(0.0, np.inf), scope="design")
            self.params.add(param, name)

    def torque(self, u: Any, p: dict[str, Any]) -> Any:
        """τ_f for applied torques ``u`` (a CasADi expression); ``p`` has the parameters by name."""
        F, Fc, w = p["breakaway"], p["kinetic"], p["width"]
        a = ca.sqrt(u * u + w * w)
        stuck = ca.sqrt((u + F) ** 2 + w * w) - ca.sqrt((u - F) ** 2 + w * w)
        G = 0.5 * (1.0 - ca.tanh((a - F) / w))
        return G * 0.5 * stuck + (1.0 - G) * Fc * u / a

    def __call__(self, u: ArrayLike) -> np.ndarray:
        """τ_f [N·m] at the current parameters; motors along the last axis."""
        F, Fc, w = (self.params[name].value for name in ("breakaway", "kinetic", "width"))
        return friction_torque(u, F, Fc, w)

    def to_dict(self) -> dict[str, Any]:
        """The parameters at their current values."""
        return {"type": "static", **{k: p.value.tolist() for k, p in self.params.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticFriction:
        """Inverse of ``to_dict``."""
        return cls(data["breakaway"], data.get("kinetic", 0.0), data.get("width", 0.01))


def friction_torque(u: ArrayLike, F: Any, Fc: Any, w: Any) -> np.ndarray:
    """τ_f for numbers: the map of ``StaticFriction`` with breakaway ``F``, kinetic ``Fc`` and
    smoothing ``w``."""
    u = np.asarray(u, dtype=float)
    a = np.sqrt(u * u + w * w)
    stuck = np.sqrt((u + F) ** 2 + w * w) - np.sqrt((u - F) ** 2 + w * w)
    G = 0.5 * (1.0 - np.tanh((a - F) / w))
    return np.asarray(G * 0.5 * stuck + (1.0 - G) * Fc * u / a)


def as_friction(value: Any) -> StaticFriction | None:
    """``value`` if it is a ``StaticFriction``, the one a dict describes, or ``None`` for none."""
    if value is None or isinstance(value, StaticFriction):
        return value
    if isinstance(value, dict):
        return StaticFriction.from_dict(value)
    raise TypeError(f"friction must be a StaticFriction or None, not {type(value).__name__}")
