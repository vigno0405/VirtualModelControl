"""Actuation: how motor angles follow from q, and how motor torques u give τ = B(q) u."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import casadi as ca
import numpy as np

from ..core.params import ParamSet, as_param
from ..core.registry import register
from ..core.units import RAD, M
from .efficiency import as_efficiency, coefficients


class Actuation(Protocol):
    """How motors drive a robot. Implement these to add a transmission (``register("actuation")``).

    ``config_from_motors`` / ``velocity_from_motors`` invert the map when the controller reads
    motors directly; ``allocate`` solves B u = τ.
    """

    params: ParamSet

    def motor_sizes(self, space: Any) -> tuple[int, int]:
        """Numbers of motor angles and motor rates."""
        ...

    def motor_angles(self, q: Any, p: dict[str, Any]) -> Any:
        """Motor angles θ(q) [rad]."""
        ...

    def motor_rates(self, q: Any, v: Any, p: dict[str, Any]) -> Any:
        """Motor rates θ̇ [rad/s]."""
        ...

    def generalized_force(self, u: Any, q: Any, p: dict[str, Any]) -> Any:
        """τ = B(q) u."""
        ...

    def allocate(self, tau: Any, q: Any, p: dict[str, Any]) -> Any:
        """Motor torques u with B(q) u = τ."""
        ...

    def config_from_motors(self, theta: Any, p: dict[str, Any]) -> Any:
        """q from motor angles."""
        ...

    def velocity_from_motors(self, q: Any, theta_dot: Any, p: dict[str, Any]) -> Any:
        """v from motor rates."""
        ...


@register("actuation", "direct")
class Direct:
    """One actuator per generalized coordinate: motor angles θ = q, B = I, u = τ.

    ``efficiency``: the torque each motor delivers for the commanded one, an ``Efficiency`` or
    its linear coefficient (one value or one per motor). It is 1 by default: models identified
    from commanded torques already include it. ``allocate`` never divides by it.
    """

    def __init__(self, efficiency: Any = 1.0) -> None:
        self.efficiency = as_efficiency(efficiency)
        self.params = ParamSet()
        self.params.merge(self.efficiency.params, "efficiency")

    def motor_sizes(self, space: Any) -> tuple[int, int]:
        """Numbers of motor angles and motor rates: nq and nv."""
        return space.nq, space.nv

    def motor_angles(self, q: Any, p: dict[str, Any]) -> Any:
        """θ = q."""
        return q

    def motor_rates(self, q: Any, v: Any, p: dict[str, Any]) -> Any:
        """θ̇ = v."""
        return v

    def generalized_force(self, u: Any, q: Any, p: dict[str, Any]) -> Any:
        """τ = the torques the motors deliver for ``u`` (τ = u at the default efficiency)."""
        return self.efficiency.delivered(u, coefficients(p))

    def allocate(self, tau: Any, q: Any, p: dict[str, Any]) -> Any:
        """u = τ."""
        return tau

    def config_from_motors(self, theta: Any, p: dict[str, Any]) -> Any:
        """q = θ."""
        return theta

    def velocity_from_motors(self, q: Any, theta_dot: Any, p: dict[str, Any]) -> Any:
        """v = θ̇."""
        return theta_dot

    def to_dict(self) -> dict[str, Any]:
        """The efficiency at its current coefficients."""
        return {"type": "direct", "efficiency": self.efficiency.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Direct:
        """Inverse of ``to_dict``."""
        return cls(data.get("efficiency", 1.0))


@register("actuation", "tendons")
class TendonTransmission:
    """Rigid tendons on PCC segments, with motor angle θ = −ΔL / r: θ > 0 pulls a tendon.

    Tendon j of a segment changes length by ΔL_j = Dl − Dx cos δ_j − Dy sin δ_j. Per segment i:
    ``seg{i}.delta`` tendon angles around the section at the base [rad] and ``seg{i}.r`` spool
    radius [m], ``design`` Params. With three tendons per segment the map is invertible.
    ``efficiency``: the torque each motor delivers for the commanded one, as for ``Direct``; 1 by
    default. ``allocate`` never divides by it.
    """

    def __init__(self, delta: Sequence[Any], r: Any, efficiency: Any = 1.0) -> None:
        n = len(delta)
        r_list = list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r] * n
        if len(r_list) != n:
            raise ValueError(f"{n} segments of tendon angles but {len(r_list)} spool radii")
        self.n_segments = n
        self.params = ParamSet()
        for i in range(n):
            key = f"seg{i + 1}.delta"
            free = (-np.inf, np.inf)
            self.params.add(as_param(delta[i], key, unit=RAD, scope="design", bounds=free), key)
            key = f"seg{i + 1}.r"
            self.params.add(
                as_param(r_list[i], key, unit=M, scope="design", bounds=(0, np.inf)), key
            )
        self.efficiency = as_efficiency(efficiency)
        self.params.merge(self.efficiency.params, "efficiency")
        self.n_tendons = [self.params[f"seg{i + 1}.delta"].size for i in range(n)]
        self.n_motors = sum(self.n_tendons)

    def motor_sizes(self, space: Any) -> tuple[int, int]:
        """Numbers of motor angles and motor rates: one per tendon."""
        return self.n_motors, self.n_motors

    def _blocks(self, p: dict[str, Any]) -> list[Any]:
        """∂θ_i/∂q_i = −A_i / r_i per segment (the map is linear in q)."""
        out = []
        for i in range(self.n_segments):
            delta, r = p[f"seg{i + 1}.delta"], p[f"seg{i + 1}.r"]
            rows = [
                ca.horzcat(-ca.cos(delta[j]), -ca.sin(delta[j]), 1) for j in range(delta.numel())
            ]
            out.append(-ca.vertcat(*rows) / r)
        return out

    def tendon_lengths(self, q: Any, p: dict[str, Any]) -> Any:
        """Length change of every tendon, ΔL [m]."""
        out = []
        for i, block in enumerate(self._blocks(p)):
            r = p[f"seg{i + 1}.r"]
            out.append(-r * ca.mtimes(block, q[3 * i : 3 * i + 3]))
        return ca.vertcat(*out)

    def motor_angles(self, q: Any, p: dict[str, Any]) -> Any:
        """θ = −ΔL / r [rad]."""
        blocks = self._blocks(p)
        return ca.vertcat(*[ca.mtimes(B, q[3 * i : 3 * i + 3]) for i, B in enumerate(blocks)])

    def jacobian(self, p: dict[str, Any]) -> Any:
        """∂θ/∂q, constant; the input matrix is B = (∂θ/∂q)ᵀ."""
        return ca.diagcat(*self._blocks(p))

    def motor_rates(self, q: Any, v: Any, p: dict[str, Any]) -> Any:
        """θ̇ = (∂θ/∂q) v."""
        return ca.mtimes(self.jacobian(p), v)

    def generalized_force(self, u: Any, q: Any, p: dict[str, Any]) -> Any:
        """τ = B τ_m with B = (∂θ/∂q)ᵀ and τ_m the torques the motors deliver for ``u``."""
        return ca.mtimes(self.jacobian(p).T, self.efficiency.delivered(u, coefficients(p)))

    def allocate(self, tau: Any, q: Any, p: dict[str, Any]) -> Any:
        """Motor torques u with B u = τ."""
        blocks = self._blocks(p)
        return ca.vertcat(*[ca.solve(B.T, tau[3 * i : 3 * i + 3]) for i, B in enumerate(blocks)])

    def config_from_motors(self, theta: Any, p: dict[str, Any]) -> Any:
        """q from motor angles (the exact inverse of ``motor_angles``)."""
        out, offset = [], 0
        for i, block in enumerate(self._blocks(p)):
            m = self.n_tendons[i]
            out.append(ca.solve(block, theta[offset : offset + m]))
            offset += m
        return ca.vertcat(*out)

    def velocity_from_motors(self, q: Any, theta_dot: Any, p: dict[str, Any]) -> Any:
        """v from motor velocities (the map is linear, so this is the same inverse)."""
        return self.config_from_motors(theta_dot, p)

    def to_dict(self) -> dict[str, Any]:
        """Constructor arguments at the current Param values."""
        n = range(1, self.n_segments + 1)
        return {
            "type": "tendons",
            "delta": [self.params[f"seg{i}.delta"].value.tolist() for i in n],
            "r": [float(self.params[f"seg{i}.r"].value) for i in n],
            "efficiency": self.efficiency.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TendonTransmission:
        """Inverse of ``to_dict``."""
        return cls(data["delta"], data["r"], data.get("efficiency", 1.0))
