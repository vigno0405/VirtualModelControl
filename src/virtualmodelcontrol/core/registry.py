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
    """Class or function decorator: make ``obj`` available as ``get(kind, name)``.

    Defining the same class again (a script or a notebook cell run twice) replaces it; another
    object under a taken name is refused.
    """

    def decorate(obj: T) -> T:
        table = _REGISTRY.setdefault(kind, {})
        if name in table and table[name] is not obj and not _same_definition(table[name], obj):
            raise ValueError(f"a {kind} named {name!r} is already registered")
        table[name] = obj
        return obj

    return decorate


def _same_definition(old: Any, new: Any) -> bool:
    """True if ``new`` is ``old`` defined again: the same module and qualified name."""
    where = getattr(new, "__module__", None), getattr(new, "__qualname__", None)
    return where[1] is not None and where == (
        getattr(old, "__module__", None),
        getattr(old, "__qualname__", None),
    )


def get(kind: str, name: str) -> Any:
    """Registered object; plugins are loaded on the first miss."""
    failed = ""
    if name not in _REGISTRY.get(kind, {}):
        try:
            load_plugins()
        except ImportError as error:  # a broken plugin must not hide that the name is unknown
            failed = f" ({error})"
    table = _REGISTRY.get(kind, {})
    if name not in table:
        raise KeyError(f"no {kind} named {name!r}; known: {sorted(table)}{failed}")
    return table[name]


def names(kind: str) -> list[str]:
    """Registered names of one kind, plugins included."""
    load_plugins()
    return sorted(_REGISTRY.get(kind, {}))


def load_plugins() -> None:
    """Import every entry point in the group ``virtualmodelcontrol.plugins`` (once).

    A plugin that fails does not stop the others: they load, then its error is raised, and again
    on every later call until it loads.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    failed: list[tuple[Any, Exception]] = []
    for ep in entry_points(group=GROUP):
        try:
            ep.load()
        except Exception as error:
            failed.append((ep, error))
    if failed:
        first, cause = failed[0]
        raise ImportError(
            f"plugin {getattr(first, 'name', first)!r} failed to load: {cause}"
        ) from cause
    _plugins_loaded = True
