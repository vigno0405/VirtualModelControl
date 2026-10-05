"""Reading a configuration: where an entry sits in the file, its keys, and registered templates."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..core.registry import get

Location = tuple[Any, ...]
"""Where an entry sits in the configuration: its keys and list indices, from the top."""


def where(path: Location) -> str:
    """A path as it reads in the file, such as ``controller.elements.reach.on``."""
    out = ""
    for key in path:
        out += f"[{key}]" if isinstance(key, int) else f".{key}" if out else str(key)
    return out or "the configuration"


def check_keys(
    spec: Any, path: Location, allowed: Iterable[str] | None, required: Iterable[str] = ()
) -> dict[str, Any]:
    """``spec`` as a mapping whose keys are all ``allowed`` (any, for None) and include the
    ``required`` ones."""
    if not isinstance(spec, dict):
        raise ValueError(f"{where(path)}: expected a mapping, got {spec!r}")
    if allowed is not None:
        allowed = list(allowed)
        unknown = [key for key in spec if key not in allowed]
        if unknown:
            raise ValueError(f"{where(path)}: unknown key {unknown[0]!r}; allowed: {allowed}")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"{where(path)}: missing {missing[0]!r}")
    return spec


def template(kind: str, spec: Any, path: Location) -> Callable[..., Any]:
    """A registered template, called with the arguments the file gives it: ``spec`` is its name,
    or a mapping of ``template`` (the name) and the template's keyword arguments."""
    if isinstance(spec, str):
        name, kwargs = spec, {}
    elif isinstance(spec, dict) and "template" in spec:
        name, kwargs = spec["template"], {k: v for k, v in spec.items() if k != "template"}
    else:
        raise ValueError(
            f"{where(path)}: expected a template's name, or a mapping of `template` and its "
            f"arguments; got {spec!r}"
        )
    factory = get(kind, name)

    def call(*args: Any) -> Any:
        try:
            return factory(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{where(path)}: {exc}") from None

    return call
