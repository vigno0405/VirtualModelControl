"""Points fixed to a model's frames, at a named site or at an arc parameter s."""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...core.params import Param, as_param
from .base import Context, Coordinate


class FramePoint(Coordinate):
    """Position [m] of a point fixed to a model frame: a named site, or arc parameter ``s``.

    ``s`` and ``offset`` (in the frame, [m]) become ``episode`` Params, folded in at compile time
    unless made live. Give both ``at`` and ``s`` for a continuous part of an assembly.
    """

    def __init__(self, model: Any, at: str | None = None, *, s: Any = None, offset: Any = None):
        if at is None and s is None:
            raise ValueError("give a site name `at`, an arc parameter `s`, or both")
        super().__init__(3, "m")
        self.model = model
        self.site = at
        self.s = None if s is None else as_param(s, "s", bounds=(0.0, 1.0), scope="episode")
        self.offset = (
            None if offset is None else as_param(offset, "offset", unit="m", scope="episode")
        )

    def params(self) -> dict[str, Param]:
        """The arc parameter and the offset, when given."""
        out = {}
        if self.s is not None:
            out["s"] = self.s
        if self.offset is not None:
            out["offset"] = self.offset
        return out

    def value(self, ctx: Context) -> Any:
        """Point position in the model's base frame."""
        if self.s is None:
            at: Any = self.site
        elif self.site is None:
            at = ctx.param(self.s)
        else:
            at = (self.site, ctx.param(self.s))
        R, p = self.model.frame(ctx.q, at, ctx.view(self.model.params))
        if self.offset is not None:
            p = p + ca.mtimes(R, ca.reshape(ctx.param(self.offset), 3, 1))
        return p

    def __repr__(self) -> str:
        where = ", ".join(
            x
            for x in (
                repr(self.site) if self.site else "",
                f"s={self.s.value.tolist()}" if self.s is not None else "",
            )
            if x
        )
        return f"FramePoint({where})"
