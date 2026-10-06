"""Points, rotations and errors of a model's frames, at a named site or at an arc parameter s."""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...core.params import Param, as_param
from ...math import exp_so3, log_so3
from .base import Context, Coordinate


class OnFrame(Coordinate):
    """A coordinate of a model frame: a named site, or arc parameter ``s``, or both.

    Without ``q`` the model is the robot's, at the robot's configuration; with a coordinate ``q``
    it is a virtual model, at the configuration ``q`` gives: a virtual state, a reference or a
    function of time. ``s`` becomes an ``episode`` Param, folded in at compile time unless made
    live. Give both ``at`` and ``s`` for a continuous part of an assembly.
    """

    def __init__(
        self,
        dim: int,
        unit: str,
        model: Any,
        at: str | None = None,
        *,
        s: Any = None,
        q: Coordinate | None = None,
    ):
        if at is None and s is None:
            raise ValueError("give a site name `at`, an arc parameter `s`, or both")
        if q is not None and q.dim != model.space.nq:
            raise ValueError(f"the model has {model.space.nq} coordinates, q has {q.dim}")
        super().__init__(dim, unit)
        self.model = model
        self.q = q
        self.site = at
        self.s = None if s is None else as_param(s, "s", bounds=(0.0, 1.0), scope="episode")

    def params(self) -> dict[str, Param]:
        """The arc parameter, when given, and a virtual model's own Params."""
        out = {}
        if self.q is not None:
            out.update({f"model.{name}": param for name, param in self.model.params.items()})
        if self.s is not None:
            out["s"] = self.s
        return out

    def children(self) -> tuple[Coordinate, ...]:
        """The coordinate that gives a virtual model's configuration, when there is one."""
        return () if self.q is None else (self.q,)

    def frame(self, ctx: Context) -> tuple[Any, Any]:
        """The frame's rotation (3, 3) and position (3, 1) in the model's base frame."""
        q = ctx.q if self.q is None else ctx.value(self.q)
        if self.s is None:
            at: Any = self.site
        elif self.site is None:
            at = ctx.param(self.s)
        else:
            at = (self.site, ctx.param(self.s))
        return self.model.frame(q, at, ctx.view(self.model.params))

    def __repr__(self) -> str:
        where = ", ".join(
            x
            for x in (
                repr(self.site) if self.site else "",
                f"s={self.s.value.tolist()}" if self.s is not None else "",
            )
            if x
        )
        return f"{type(self).__name__}({where})"


class FramePoint(OnFrame):
    """Position [m] of a point fixed to a model frame: a named site, or arc parameter ``s``.

    ``offset`` (in the frame, [m]) becomes an ``episode`` Param. ``q`` makes the model a virtual
    one, as for any coordinate of a frame.
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
        super().__init__(3, "m", model, at, s=s, q=q)
        self.offset = (
            None if offset is None else as_param(offset, "offset", unit="m", scope="episode")
        )

    def params(self) -> dict[str, Param]:
        """The arc parameter and the offset, when given, and a virtual model's own Params."""
        out = super().params()
        if self.offset is not None:
            out["offset"] = self.offset
        return out

    def value(self, ctx: Context) -> Any:
        """Point position in the model's base frame."""
        R, p = self.frame(ctx)
        if self.offset is not None:
            p = p + ca.mtimes(R, ca.reshape(ctx.param(self.offset), 3, 1))
        return p


class FrameRotation(OnFrame):
    """The rotation matrix of a model frame, row by row: 9 entries."""

    def __init__(
        self, model: Any, at: str | None = None, *, s: Any = None, q: Coordinate | None = None
    ):
        super().__init__(9, "", model, at, s=s, q=q)

    def value(self, ctx: Context) -> Any:
        """The rows of R, one after the other."""
        return ca.reshape(self.frame(ctx)[0].T, 9, 1)


class OrientationError(OnFrame):
    """The rotation of a model frame from a goal orientation, as a rotation vector [rad].

    It is the vector φ with exp(φ) = R_goalᵀ R: the frame's orientation relative to the goal, in
    the goal's axes, with an angle below π. ``goal`` is a rotation vector, a ``stage`` Param.
    """

    def __init__(
        self,
        model: Any,
        at: str | None = None,
        *,
        s: Any = None,
        goal: Any = (0.0, 0.0, 0.0),
        q: Coordinate | None = None,
    ):
        super().__init__(3, "rad", model, at, s=s, q=q)
        free = (-float("inf"), float("inf"))
        self.goal = as_param(goal, "goal", unit="rad", scope="stage", bounds=free)

    def params(self) -> dict[str, Param]:
        """The goal, and the arc parameter and a virtual model's Params."""
        return {**super().params(), "goal": self.goal}

    def value(self, ctx: Context) -> Any:
        """log(R_goalᵀ R)."""
        R_goal = exp_so3(ca.reshape(ctx.param(self.goal), 3, 1))
        return log_so3(ca.mtimes(R_goal.T, self.frame(ctx)[0]))


class InFrame(OnFrame):
    """A 3-vector coordinate expressed along the axes of a model frame: Rᵀ c."""

    def __init__(
        self,
        coord: Coordinate,
        model: Any,
        at: str | None = None,
        *,
        s: Any = None,
        q: Coordinate | None = None,
    ):
        if coord.dim != 3:
            raise ValueError(f"a vector in a frame has 3 entries, the coordinate has {coord.dim}")
        super().__init__(3, coord.unit, model, at, s=s, q=q)
        self.coord = coord

    def children(self) -> tuple[Coordinate, ...]:
        """The vector, and the coordinate that gives a virtual model's configuration."""
        return (self.coord, *super().children())

    def value(self, ctx: Context) -> Any:
        """Rᵀ c."""
        return ca.mtimes(self.frame(ctx)[0].T, ctx.value(self.coord))


class FromFrame(InFrame):
    """A 3-vector coordinate given along the axes of a model frame, in the base frame: R c."""

    def value(self, ctx: Context) -> Any:
        """R c."""
        return ca.mtimes(self.frame(ctx)[0], ctx.value(self.coord))
