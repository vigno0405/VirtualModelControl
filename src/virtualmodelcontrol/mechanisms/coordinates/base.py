"""Coordinates: quantities y(q, z, p, t) that components act on, and where they are evaluated."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ...core.params import Binding, Param


class Context:
    """Symbols a coordinate is evaluated on: robot configuration q, virtual positions z, time t.

    Params resolve through ``binding`` (live symbols or folded constants). Values are memoized
    per coordinate object, so a coordinate shared by several components is built once.
    """

    def __init__(
        self,
        q: Any,
        binding: Binding,
        z: Any = None,
        t: Any = None,
        states: Mapping[int, slice] | None = None,
    ) -> None:
        self.q = q
        self.z = z
        self.t = t
        self.binding = binding
        self._states = dict(states or {})
        self._cache: dict[int, Any] = {}

    def param(self, param: Param) -> Any:
        """Expression of a Param."""
        return self.binding(param)

    def view(self, params: Mapping[str, Param]) -> dict[str, Any]:
        """Expressions of a model's Params, keyed by the model's own names."""
        return self.binding.view(params)

    def state(self, state: Coordinate) -> Any:
        """Slice of z holding a virtual state."""
        try:
            return self.z[self._states[id(state)]]
        except KeyError:
            raise KeyError(f"{state!r} is not a state of this system") from None

    def value(self, coord: Coordinate) -> Any:
        """Value of ``coord`` (dim × 1), built once per context."""
        key = id(coord)
        if key not in self._cache:
            self._cache[key] = coord.value(self)
        return self._cache[key]


class Coordinate:
    """A quantity y with ``dim`` entries in ``unit``, computed from q, z, the Params and t.

    Subtracting coordinates gives a ``Difference``; a plain array on either side becomes a
    ``Ref`` (a live reference). Indexing gives a ``Slice``.
    """

    def __init__(self, dim: int, unit: str = "") -> None:
        self.dim = int(dim)
        self.unit = unit

    def value(self, ctx: Context) -> Any:
        """Symbolic value, shape (dim, 1)."""
        raise NotImplementedError

    def params(self) -> dict[str, Param]:
        """Params owned by this coordinate, by local name."""
        return {}

    def children(self) -> tuple[Coordinate, ...]:
        """Coordinates this one is built from."""
        return ()

    def __sub__(self, other: Any) -> Coordinate:
        from .ops import Difference

        return Difference(self, as_coordinate(other, self))

    def __rsub__(self, other: Any) -> Coordinate:
        from .ops import Difference

        return Difference(as_coordinate(other, self), self)

    def __getitem__(self, index: Any) -> Coordinate:
        from .ops import Slice

        return Slice(self, index)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, unit={self.unit!r})"


def as_coordinate(x: Any, like: Coordinate) -> Coordinate:
    """``x`` itself if it is a coordinate, else a live reference shaped like ``like``."""
    if isinstance(x, Coordinate):
        return x
    from .references import Ref

    return Ref("ref", value=x, unit=like.unit)


def walk(coord: Coordinate) -> Iterator[Coordinate]:
    """Every coordinate in the tree of ``coord``, each once, parents first."""
    seen: set[int] = set()
    stack = [coord]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(node.children()))
