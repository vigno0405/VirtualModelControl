"""Coordinates built from other coordinates: differences, slices, stacks, projections, norms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import casadi as ca

from ...core.params import Param, Scope, as_param
from ...core.symbolic import smooth_norm
from .base import Context, Coordinate


class Difference(Coordinate):
    """a − b, e.g. a point minus its goal (the deflection of a spring)."""

    def __init__(self, a: Coordinate, b: Coordinate) -> None:
        if a.dim != b.dim:
            raise ValueError(f"cannot subtract a {b.dim}-entry coordinate from a {a.dim}-entry one")
        super().__init__(a.dim, a.unit)
        self.a, self.b = a, b

    def children(self) -> tuple[Coordinate, ...]:
        """The two operands."""
        return (self.a, self.b)

    def value(self, ctx: Context) -> Any:
        """a − b."""
        return ctx.value(self.a) - ctx.value(self.b)


class Slice(Coordinate):
    """Some entries of a coordinate."""

    def __init__(self, coord: Coordinate, index: int | slice | Sequence[int]) -> None:
        all_indices = list(range(coord.dim))
        if isinstance(index, slice):
            indices = all_indices[index]
        elif isinstance(index, int):
            indices = [all_indices[index]]
        else:
            indices = [all_indices[i] for i in index]
        super().__init__(len(indices), coord.unit)
        self.coord, self.indices = coord, indices

    def children(self) -> tuple[Coordinate, ...]:
        """The sliced coordinate."""
        return (self.coord,)

    def value(self, ctx: Context) -> Any:
        """Selected entries."""
        v = ctx.value(self.coord)
        return ca.vertcat(*[v[i] for i in self.indices])


class Stack(Coordinate):
    """Several coordinates stacked into one."""

    def __init__(self, *coords: Coordinate) -> None:
        units = {c.unit for c in coords}
        super().__init__(sum(c.dim for c in coords), units.pop() if len(units) == 1 else "")
        self.coords = coords

    def children(self) -> tuple[Coordinate, ...]:
        """The stacked coordinates."""
        return self.coords

    def value(self, ctx: Context) -> Any:
        """Values stacked in order."""
        return ca.vertcat(*[ctx.value(c) for c in self.coords])


class Projection(Coordinate):
    """Signed length of a coordinate along a direction: y = n̂ᵀ c with n̂ = n / ‖n‖.

    A spring on a projection acts along n only (a cart). The direction is an ``episode`` Param.
    """

    def __init__(self, coord: Coordinate, direction: Any, *, scope: Scope = "episode") -> None:
        super().__init__(1, coord.unit)
        self.coord = coord
        self.direction = as_param(direction, "direction", scope=scope)
        if self.direction.size != coord.dim:
            raise ValueError(f"direction needs {coord.dim} entries, got {self.direction.size}")

    def params(self) -> dict[str, Param]:
        """The direction."""
        return {"direction": self.direction}

    def children(self) -> tuple[Coordinate, ...]:
        """The projected coordinate."""
        return (self.coord,)

    def value(self, ctx: Context) -> Any:
        """n̂ᵀ c."""
        n = ca.reshape(ctx.param(self.direction), self.coord.dim, 1)
        return ca.dot(n, ctx.value(self.coord)) / ca.norm_2(n)


class Norm(Coordinate):
    """Smooth Euclidean length of a coordinate, √(cᵀc + ε)."""

    def __init__(self, coord: Coordinate, eps: float = 1e-18) -> None:
        super().__init__(1, coord.unit)
        self.coord, self.eps = coord, eps

    def children(self) -> tuple[Coordinate, ...]:
        """The measured coordinate."""
        return (self.coord,)

    def value(self, ctx: Context) -> Any:
        """√(cᵀc + ε)."""
        return smooth_norm(ctx.value(self.coord), self.eps)


class Custom(Coordinate):
    """y = fn(child values, Param expressions by name), with ``fn`` written in CasADi operations."""

    def __init__(
        self,
        fn: Callable[..., Any],
        children: Sequence[Coordinate] = (),
        *,
        dim: int,
        unit: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dim, unit)
        self.fn = fn
        self._children = tuple(children)
        self._params = {name: as_param(v, name) for name, v in (params or {}).items()}

    def params(self) -> dict[str, Param]:
        """Params passed to ``fn`` by keyword."""
        return dict(self._params)

    def children(self) -> tuple[Coordinate, ...]:
        """Coordinates passed to ``fn`` in order."""
        return self._children

    def value(self, ctx: Context) -> Any:
        """``fn`` applied to the children's values and the Params."""
        args = [ctx.value(c) for c in self._children]
        kwargs = {name: ctx.param(p) for name, p in self._params.items()}
        return ca.reshape(self.fn(*args, **kwargs), self.dim, 1)
