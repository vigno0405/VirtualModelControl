"""Joint-space models: independent joints with no frames (controllers that act on joints only)."""

from __future__ import annotations

from typing import Any

from ..core.params import ParamSet
from ..core.registry import register
from ..core.space import Euclidean


@register("model", "joint_space")
class JointSpace:
    """``n`` independent joints [rad, or ``unit``] with no frames: act on ``robot.joint(i)``."""

    def __init__(self, n: int, unit: str = "rad") -> None:
        self.space = Euclidean(n)
        self.params = ParamSet()
        self.sites: tuple[str, ...] = ()
        self.q_unit = unit

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """A joint-space model has no frames."""
        raise KeyError("a JointSpace model has no frames; use robot.joint(i) coordinates")

    def to_dict(self) -> dict[str, Any]:
        """Number of joints and unit."""
        return {"type": "joint_space", "n": self.space.nq, "unit": self.q_unit}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JointSpace:
        """Inverse of ``to_dict``."""
        return cls(data["n"], data.get("unit", "rad"))
