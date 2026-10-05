"""Coordinates in a configuration: a name, or a mapping of one registered kind to its arguments.

Each kind is a builder ``(args, scope, path) -> Coordinate`` registered as ``"coordinate"``, so a
plugin adds its own. Builders record which Param holds which value of the file, so the file can
be written back with the Params' current values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..core.registry import get, register
from ..mechanisms import (
    Coordinate,
    Difference,
    Norm,
    PlaneDistance,
    Projection,
    Ref,
    Slice,
    SphereDistance,
    Stack,
)
from ..mechanisms.coordinates.base import as_coordinate
from .spec import Path, check_keys, where


class Scope:
    """What a coordinate is built from: the robot, the coordinates and states known by name, and
    ``record``, the Param behind each value of the file (by its path)."""

    def __init__(
        self, robot: Any, names: dict[str, Coordinate], record: list[tuple[Path, Param]]
    ) -> None:
        self.robot, self.names, self.record = robot, names, record

    def build(self, spec: Any, path: Path) -> Coordinate:
        """The coordinate ``spec``: a known name, or ``{kind: arguments}``."""
        if isinstance(spec, str):
            if spec not in self.names:
                raise KeyError(
                    f"{where(path)}: no coordinate or state named {spec!r}; known: "
                    f"{sorted(self.names)}"
                )
            return self.names[spec]
        if isinstance(spec, dict) and len(spec) == 1:
            kind, args = next(iter(spec.items()))
            return get("coordinate", kind)(args, self, (*path, kind))
        raise ValueError(
            f"{where(path)}: a coordinate is a name or a mapping of one kind to its arguments, "
            f"such as {{point: {{s: 1.0}}}}; got {spec!r}"
        )

    def operand(self, spec: Any, path: Path, like: Coordinate) -> Coordinate:
        """A coordinate, or a plain list: a live reference named ``ref`` shaped like ``like``."""
        if _plain(spec):
            ref = as_coordinate(spec, like)
            if not isinstance(ref, Ref):
                raise TypeError(f"{where(path)}: expected a reference, got {ref!r}")
            self.track(ref.param, path)
            return ref
        return self.build(spec, path)

    def track(self, param: Param, path: Path) -> None:
        """Remember that ``param`` holds the value at ``path``."""
        self.record.append((tuple(path), param))


def _plain(spec: Any) -> bool:
    return isinstance(spec, (list, tuple, int, float, np.ndarray))


@register("coordinate", "point")
def point(args: Any, scope: Scope, path: Path) -> Coordinate:
    """A point of the robot: a site's name, or ``{at, s, offset}`` as in ``robot.point``."""
    if isinstance(args, str):
        return scope.robot.point(args)
    check_keys(args, path, ("at", "s", "offset"))
    p = scope.robot.point(args.get("at"), s=args.get("s"), offset=args.get("offset"))
    if p.s is not None:
        scope.track(p.s, (*path, "s"))
    if p.offset is not None:
        scope.track(p.offset, (*path, "offset"))
    return p


@register("coordinate", "joint")
def joint(args: Any, scope: Scope, path: Path) -> Coordinate:
    """Entries of the robot's q: an index, a list of indices, or ``{start, stop, step}``."""
    if isinstance(args, dict):
        check_keys(args, path, ("start", "stop", "step"), required=("stop",))
        return scope.robot.joint(slice(args.get("start", 0), args["stop"], args.get("step")))
    return scope.robot.joint(args)


@register("coordinate", "state")
def state(args: Any, scope: Scope, path: Path) -> Coordinate:
    """A virtual state of the controller, by name."""
    return scope.build(args, path)


@register("coordinate", "ref")
def ref(args: Any, scope: Scope, path: Path) -> Coordinate:
    """A live reference held in a Param: ``{name, value, unit (m), scope (stage)}``."""
    check_keys(args, path, ("name", "value", "unit", "scope"), required=("name", "value"))
    r = Ref(
        args["name"],
        value=args["value"],
        unit=args.get("unit", "m"),
        scope=args.get("scope", "stage"),
    )
    scope.track(r.param, (*path, "value"))
    return r


@register("coordinate", "difference")
def difference(args: Any, scope: Scope, path: Path) -> Coordinate:
    """a − b; a plain list on either side is a live reference named ``ref``."""
    if not (isinstance(args, list) and len(args) == 2):
        raise ValueError(f"{where(path)}: a difference takes two coordinates, got {args!r}")
    if _plain(args[0]):
        b = scope.build(args[1], (*path, 1))
        return Difference(scope.operand(args[0], (*path, 0), like=b), b)
    a = scope.build(args[0], (*path, 0))
    return Difference(a, scope.operand(args[1], (*path, 1), like=a))


@register("coordinate", "projection")
def projection(args: Any, scope: Scope, path: Path) -> Coordinate:
    """The length of a coordinate along a direction: ``{of, direction, scope (episode)}``."""
    check_keys(args, path, ("of", "direction", "scope"), required=("of", "direction"))
    coord = scope.build(args["of"], (*path, "of"))
    out = Projection(coord, args["direction"], scope=args.get("scope", "episode"))
    scope.track(out.direction, (*path, "direction"))
    return out


@register("coordinate", "norm")
def norm(args: Any, scope: Scope, path: Path) -> Coordinate:
    """The length of a coordinate."""
    return Norm(scope.build(args, path))


@register("coordinate", "slice")
def slice_(args: Any, scope: Scope, path: Path) -> Coordinate:
    """Some entries of a coordinate: ``{of, index}`` with an index or a list of indices."""
    check_keys(args, path, ("of", "index"), required=("of", "index"))
    return Slice(scope.build(args["of"], (*path, "of")), args["index"])


@register("coordinate", "stack")
def stack(args: Any, scope: Scope, path: Path) -> Coordinate:
    """Coordinates stacked into one: a list."""
    if not isinstance(args, list):
        raise ValueError(f"{where(path)}: a stack takes a list of coordinates, got {args!r}")
    return Stack(*(scope.build(spec, (*path, i)) for i, spec in enumerate(args)))


@register("coordinate", "plane_distance")
def plane_distance(args: Any, scope: Scope, path: Path) -> Coordinate:
    """Signed distance from a point to a plane: ``{point, normal, origin (0, 0, 0)}``."""
    check_keys(args, path, ("point", "normal", "origin"), required=("point", "normal"))
    out = PlaneDistance(
        scope.build(args["point"], (*path, "point")),
        args["normal"],
        args.get("origin", (0.0, 0.0, 0.0)),
    )
    scope.track(out.normal, (*path, "normal"))
    if "origin" in args:
        scope.track(out.origin, (*path, "origin"))
    return out


@register("coordinate", "sphere_distance")
def sphere_distance(args: Any, scope: Scope, path: Path) -> Coordinate:
    """Signed distance from a point to a sphere's surface: ``{point, center, radius}``."""
    check_keys(args, path, ("point", "center", "radius"), required=("point", "center", "radius"))
    out = SphereDistance(
        scope.build(args["point"], (*path, "point")), args["center"], args["radius"]
    )
    scope.track(out.center, (*path, "center"))
    scope.track(out.radius, (*path, "radius"))
    return out
