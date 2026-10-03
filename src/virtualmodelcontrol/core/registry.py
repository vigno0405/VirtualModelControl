"""Registry: names usable from configuration files, extended by plugins through entry points."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

T = TypeVar("T")

GROUP = "virtualmodelcontrol.plugins"
"""Entry-point group scanned for plugins; loading an entry point runs its ``register`` calls."""

_REGISTRY: dict[str, dict[str, Any]] = {}
_plugins_loaded = False


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class or function decorator: make ``obj`` available as ``get(kind, name)``."""

    def decorate(obj: T) -> T:
        table = _REGISTRY.setdefault(kind, {})
        if name in table and table[name] is not obj:
            raise ValueError(f"a {kind} named {name!r} is already registered")
        table[name] = obj
        return obj

    return decorate


def get(kind: str, name: str) -> Any:
    """Registered object; plugins are loaded on the first miss."""
    if name not in _REGISTRY.get(kind, {}):
        load_plugins()
    table = _REGISTRY.get(kind, {})
    if name not in table:
        raise KeyError(f"no {kind} named {name!r}; known: {sorted(table)}")
    return table[name]


def names(kind: str) -> list[str]:
    """Registered names of one kind, plugins included."""
    load_plugins()
    return sorted(_REGISTRY.get(kind, {}))


def load_plugins() -> None:
    """Import every entry point in the group ``virtualmodelcontrol.plugins`` (once)."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=GROUP):
        ep.load()
