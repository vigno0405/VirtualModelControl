"""Kinematic model contract: a space, Params, named sites and frames written in CasADi."""

from __future__ import annotations

from typing import Any, Protocol

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import ParamSet, constants
from ..core.registry import get
from ..core.space import Space


class KinematicModel(Protocol):
    """Frames of a robot as smooth functions of q and the Params.

    ``frame(q, at, p)`` returns (R, position) in the base frame; ``at`` is a site name or, for a
    continuous body, an arc parameter s (possibly symbolic); ``p`` maps the model's Param names
    to CasADi expressions.
    """

    @property
    def space(self) -> Space:
        """Configuration space."""
        ...

    @property
    def params(self) -> ParamSet:
        """Every geometric number, as Params."""
        ...

    @property
    def sites(self) -> tuple[str, ...]:
        """Named points of the model."""
        ...

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """Rotation (3, 3) and position (3, 1) at ``at``."""
        ...


def evaluate_frame(model: KinematicModel, q: ArrayLike, at: Any) -> tuple[np.ndarray, np.ndarray]:
    """Numeric (R, position) at the model's current Param values."""
    R, p = model.frame(ca.DM(np.asarray(q, dtype=float)), at, constants(model.params))
    return np.array(ca.evalf(R)), np.array(ca.evalf(p)).ravel()


def from_dict(data: dict[str, Any], kind: str = "model") -> Any:
    """Build a registered model (or actuation, with ``kind``) from ``to_dict()`` output."""
    return get(kind, data["type"]).from_dict(data)
