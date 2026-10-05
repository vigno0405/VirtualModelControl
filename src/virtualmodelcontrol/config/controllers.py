"""Controllers in a configuration: a registered template, or virtual states and elements.

An element names a registered component ``type``; ``coordinate`` is the coordinate it acts on
(left out for components that take the robot, such as gravity compensation) and every other key
is a keyword argument of the component, such as its stiffness.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..core.params import Param
from ..core.registry import get
from ..mechanisms import Component, Coordinate, Mechanism
from .coordinates import Scope
from .spec import Location, check_keys, template, where


def build_controller(
    spec: Any,
    name: str,
    robot: Mechanism,
    coordinates: dict[str, Coordinate],
    record: list[tuple[Location, Param]],
    path: Location,
    *,
    named: bool = False,
) -> Mechanism:
    """The controller mechanism ``name`` of ``spec``; ``named`` allows the key ``name``."""
    if isinstance(spec, dict) and "name" in spec and not named:
        raise ValueError(f"{where(path)}: this controller is named by its key; remove `name`")
    if isinstance(spec, dict) and "template" in spec:
        args = {k: v for k, v in spec.items() if k != "name"}
        mechanism = template("controller", args, path)(robot)
        mechanism.name = name
        return mechanism
    check_keys(spec, path, ("name", "states", "elements"))
    mechanism = Mechanism(name)
    states: dict[str, Coordinate] = {}
    for key, state in (spec.get("states") or {}).items():
        check_keys(state, (*path, "states", key), ("dim", "unit", "initial"))
        if key in coordinates:
            raise ValueError(f"{where((*path, 'states', key))}: a coordinate is named {key!r} too")
        states[key] = mechanism.add_state(key, **state)
    scope = Scope(robot, {**coordinates, **states}, record)
    for key, element in (spec.get("elements") or {}).items():
        mechanism.add(key, build_element(element, scope, (*path, "elements", key)))
    return mechanism


def build_element(spec: Any, scope: Scope, path: Location) -> Component:
    """A component of a registered ``type``, on its ``coordinate`` or on the robot."""
    check_keys(spec, path, None, required=("type",))
    factory = get("component", spec["type"])
    kwargs = {k: v for k, v in spec.items() if k not in ("type", "coordinate")}
    takes_robot = next(iter(inspect.signature(factory).parameters)) == "robot"
    if takes_robot and "coordinate" in spec:
        raise ValueError(f"{where(path)}: {spec['type']} acts on the robot; remove `coordinate`")
    if not takes_robot and "coordinate" not in spec:
        raise ValueError(f"{where(path)}: {spec['type']} needs the `coordinate` it acts on")
    first = scope.robot if takes_robot else scope.build(spec["coordinate"], (*path, "coordinate"))
    try:
        component = factory(first, **kwargs)
    except TypeError as exc:
        raise TypeError(f"{where(path)}: {exc}") from None
    params = {**component.coord.params(), **component.params()}
    for key in kwargs:
        if key in params:
            scope.track(params[key], (*path, key))
    return component
