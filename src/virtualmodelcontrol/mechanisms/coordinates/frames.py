"""Points fixed to a model's frames, at a named site or at an arc parameter s."""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...core.params import Param, as_param
from .base import Context, Coordinate


class FramePoint(Coordinate):
    """Position [m] of a point fixed to a model frame: a named site, or arc parameter ``s``.

    ``s`` and ``offset`` (in the frame, [m]) become ``episode`` Params, folded in at compile time
    unless made live. Give both ``at`` and ``s`` for a continuous part of an assembly. Without
    ``q`` the model is the robot's, at the robot's configuration; with a coordinate ``q`` it is a
    virtual model, at the configuration ``q`` gives: a virtual state, a reference or a function
    of time.
    """

    def __init__(
        self,
        model: Any,
        at: str | None = None,
        *,
        s: Any = None,
        offset: Any = None,
        q: Coordinate | None = None,
    ):
        if at is None and s is None:
            raise ValueError("give a site name `at`, an arc parameter `s`, or both")
        if q is not None and q.dim != model.space.nq:
            raise ValueError(f"the model has {model.space.nq} coordinates, q has {q.dim}")
        super().__init__(3, "m")
        self.model = model
        self.q = q
        self.site = at
        self.s = None if s is None else as_param(s, "s", bounds=(0.0, 1.0), scope="episode")
        self.offset = (
            None if offset is None else as_param(offset, "offset", unit="m", scope="episode")
        )

    def params(self) -> dict[str, Param]:
        """The arc parameter and the offset, when given, and a virtual model's own Params."""
        out = {}
        if self.q is not None:
            out.update({f"model.{name}": param for name, param in self.model.params.items()})
        if self.s is not None:
            out["s"] = self.s
        if self.offset is not None:
            out["offset"] = self.offset
        return out

    def children(self) -> tuple[Coordinate, ...]:
        """The coordinate that gives a virtual model's configuration, when there is one."""
        return () if self.q is None else (self.q,)

    def value(self, ctx: Context) -> Any:
        """Point position in the model's base frame."""
        q = ctx.q if self.q is None else ctx.value(self.q)
        if self.s is None:
            at: Any = self.site
        elif self.site is None:
            at = ctx.param(self.s)
        else:
            at = (self.site, ctx.param(self.s))
        R, p = self.model.frame(q, at, ctx.view(self.model.params))
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
