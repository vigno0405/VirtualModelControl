"""References: coordinates held in Params, such as goals and obstacle positions."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...core.params import Param, Scope
from .base import Context, Coordinate


class Ref(Coordinate):
    """A reference value held in a Param; ``stage`` scope, so live at every step by default."""

    def __init__(
        self,
        name: str,
        dim: int | None = None,
        value: Any = None,
        *,
        unit: str = "m",
        scope: Scope = "stage",
    ) -> None:
        if isinstance(value, Param):
            param = value
        else:
            if value is None:
                value = np.zeros(1 if dim is None else dim)
            param = Param(name, np.ravel(value), unit=unit, scope=scope)
        if dim is not None and dim != param.size:
            raise ValueError(f"Ref {name!r}: dim {dim} but the value has {param.size} entries")
        super().__init__(param.size, param.unit)
        self.param = param

    def params(self) -> dict[str, Param]:
        """The reference Param."""
        return {self.param.name: self.param}

    def value(self, ctx: Context) -> Any:
        """The Param as a column."""
        return ca.reshape(ctx.param(self.param), self.dim, 1)
