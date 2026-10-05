"""A gate on an element: its force times a number between 0 and 1."""

from __future__ import annotations

from typing import Any

from ...core.params import Param
from ..coordinates.base import Context
from .base import Component


class Gated(Component):
    """``component`` with its force and energy times a ``gate`` in [0, 1]: 0 removes it, 1 keeps it.

    The gate is a live Param, ``<name>.gate``: a controller can switch the element, and an
    optimization can free it (``optimization.Sparsity`` keeps few).
    """

    def __init__(self, component: Component, gate: Any = 1.0) -> None:
        if component.kind == "inertance":
            raise TypeError("a gate scales forces, and an inertance has none")
        super().__init__(component.coord)
        self.component = component
        self.kind = component.kind
        self.gate = self._param("gate", gate, unit="", scope="stage", bounds=(0.0, 1.0))

    def params(self) -> dict[str, Param]:
        """The element's Params, and the gate."""
        return {**self.component.params(), **super().params()}

    def force(self, ctx: Context, y: Any, yd: Any) -> Any:
        """The gate times the element's force."""
        return ctx.param(self.gate) * self.component.force(ctx, y, yd)

    def energy(self, ctx: Context, y: Any) -> Any:
        """The gate times the element's energy."""
        return ctx.param(self.gate) * self.component.energy(ctx, y)

    def __repr__(self) -> str:
        return f"Gated({self.component!r}, gate={self.gate.value.tolist()})"
