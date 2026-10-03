"""Couplings: a model whose joints are driven together, joint angles = matrix · q."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.params import Param, ParamSet
from ...core.registry import get, register
from ...core.space import Euclidean


@register("model", "linear_coupling")
class LinearCoupling:
    """Wraps a joint-space model so that its joint angles are ``coupling`` · q.

    Use it when one motor drives several joints (pulleys, mimic joints): q holds one entry per
    motor. ``coupling`` (n_joints × n_q) is a ``design`` Param; the wrapped model's Params keep
    their names.
    """

    def __init__(self, model: Any, coupling: Any) -> None:
        value = np.atleast_2d(np.asarray(getattr(coupling, "value", coupling), dtype=float))
        if value.shape[0] != model.space.nq:
            raise ValueError(
                f"coupling has {value.shape[0]} rows but the model has {model.space.nq} joints"
            )
        self.model = model
        self.space = Euclidean(value.shape[1])
        self.params = ParamSet()
        self.params.merge(model.params)
        free = (-np.inf, np.inf)
        self.coupling = (
            coupling
            if isinstance(coupling, Param)
            else Param("coupling", value, scope="design", bounds=free)
        )
        self.params.add(self.coupling, "coupling")
        self.sites = tuple(model.sites)
        self.q_unit = getattr(model, "q_unit", "")

    def joint_angles(self, q: Any, p: dict[str, Any]) -> Any:
        """Joint angles of the wrapped model."""
        return ca.mtimes(p["coupling"], q)

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """Frame of the wrapped model at the coupled joint angles."""
        return self.model.frame(self.joint_angles(q, p), at, p)

    def to_dict(self) -> dict[str, Any]:
        """The wrapped model and the coupling matrix."""
        return {
            "type": "linear_coupling",
            "model": self.model.to_dict(),
            "coupling": self.coupling.value.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearCoupling:
        """Inverse of ``to_dict``."""
        inner = data["model"]
        return cls(get("model", inner["type"]).from_dict(inner), data["coupling"])
