"""Parameters: named numbers with units, bounds and scopes, packed into one flat vector."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

Scope = Literal["fixed", "design", "episode", "stage"]
SCOPES: tuple[str, ...] = ("fixed", "design", "episode", "stage")


class Param:
    """A named number (scalar, vector or matrix) with a unit, bounds and a scope.

    The scope is the time scale on which the value may change: ``fixed`` (never), ``design``
    (with the hardware), ``episode`` (between runs) or ``stage`` (at every step).
    """

    __slots__ = ("_value", "bounds", "name", "scale", "scope", "unit")

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        *,
        unit: str = "",
        bounds: tuple[Any, Any] = (-np.inf, np.inf),
        scale: float = 1.0,
        scope: Scope = "fixed",
    ) -> None:
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        value = np.array(value, dtype=float)
        if value.ndim > 2:
            raise ValueError(f"Param {name!r} must be a scalar, vector or matrix")
        self.name = name
        self._value = value
        self.unit = unit
        self.bounds = (bounds[0], bounds[1])
        self.scale = float(scale)
        self.scope = scope

    @property
    def value(self) -> np.ndarray:
        """Current value, as a float array of fixed shape."""
        return self._value

    @value.setter
    def value(self, value: ArrayLike) -> None:
        new = np.array(value, dtype=float)
        if new.shape != self._value.shape:
            if new.size != self._value.size:
                raise ValueError(
                    f"Param {self.name!r} has shape {self._value.shape}, got a value of shape "
                    f"{new.shape}"
                )
            new = new.reshape(self._value.shape)
        self._value = new

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the value: () for a scalar, (n,) for a vector, (r, c) for a matrix."""
        return self._value.shape

    @property
    def size(self) -> int:
        """Number of entries in the value."""
        return self._value.size

    def __repr__(self) -> str:
        unit = f", unit={self.unit!r}" if self.unit else ""
        return f"Param({self.name!r}, {self._value.tolist()!r}{unit}, scope={self.scope!r})"


def as_param(
    value: Any,
    name: str,
    *,
    unit: str = "",
    bounds: tuple[Any, Any] = (-np.inf, np.inf),
    scope: Scope = "fixed",
) -> Param:
    """Return ``value`` if it already is a Param, else wrap it in a new Param."""
    if isinstance(value, Param):
        return value
    return Param(name, value, unit=unit, bounds=bounds, scope=scope)


class ParamSet(Mapping[str, Param]):
    """Ordered, named collection of Params; packs their values into one flat vector.

    A Param object appears once: adding it again under another name keeps the first name.
    Matrices are packed column by column, as CasADi stores them.
    """

    def __init__(self, params: Iterable[Param] = ()) -> None:
        self._params: dict[str, Param] = {}
        self._names: dict[int, str] = {}
        for param in params:
            self.add(param)

    def add(self, param: Param, name: str | None = None, *, rename: bool = False) -> str:
        """Add ``param`` under ``name`` (default: its own name) and return the name used.

        With ``rename``, a name already taken gets a numeric suffix instead of raising.
        """
        if id(param) in self._names:
            return self._names[id(param)]
        name = param.name if name is None else name
        if name in self._params:
            if not rename:
                raise ValueError(f"duplicate parameter name {name!r}")
            k = 2
            while f"{name}{k}" in self._params:
                k += 1
            name = f"{name}{k}"
        self._params[name] = param
        self._names[id(param)] = name
        return name

    def merge(self, other: ParamSet, prefix: str = "") -> None:
        """Add every Param of ``other``, with names prefixed by ``prefix.``."""
        for name, param in other.items():
            self.add(param, f"{prefix}.{name}" if prefix else name)

    def __getitem__(self, name: str) -> Param:
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(f"no parameter {name!r}; known: {list(self._params)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __repr__(self) -> str:
        return f"ParamSet({list(self._params)})"

    def name_of(self, param: Param) -> str:
        """Name under which ``param`` is stored."""
        return self._names[id(param)]

    def has(self, param: Param) -> bool:
        """True if this exact Param object is in the set."""
        return id(param) in self._names

    def select(self, patterns: Iterable[str] = (), scopes: Iterable[str] = ()) -> list[str]:
        """Names matching any glob pattern (``ctrl.*.stiffness``) or having any of the scopes."""
        patterns, scopes = list(patterns), set(scopes)
        return [
            name
            for name, param in self._params.items()
            if param.scope in scopes or any(fnmatch.fnmatchcase(name, pat) for pat in patterns)
        ]

    def size(self, names: Iterable[str] | None = None) -> int:
        """Total number of entries of the named Params (default: all)."""
        return sum(self[n].size for n in self._names_or_all(names))

    def vector(self, names: Iterable[str] | None = None) -> np.ndarray:
        """Values of the named Params (default: all) packed into one flat vector."""
        parts = [self[n].value.ravel(order="F") for n in self._names_or_all(names)]
        return np.concatenate(parts) if parts else np.zeros(0)

    def set_vector(self, x: ArrayLike, names: Iterable[str] | None = None) -> None:
        """Unpack a flat vector into the named Params (default: all)."""
        x = np.asarray(x, dtype=float).ravel()
        names = self._names_or_all(names)
        if x.size != self.size(names):
            raise ValueError(f"expected {self.size(names)} values, got {x.size}")
        offset = 0
        for n in names:
            param = self[n]
            param.value = x[offset : offset + param.size].reshape(param.shape, order="F")
            offset += param.size

    def to_dict(self) -> dict[str, Any]:
        """Values by name, as plain floats and lists."""
        return {name: param.value.tolist() for name, param in self._params.items()}

    def update(self, values: Mapping[str, ArrayLike]) -> None:
        """Set values by name."""
        for name, value in values.items():
            self[name].value = value

    def _names_or_all(self, names: Iterable[str] | None) -> list[str]:
        return list(self._params) if names is None else list(names)


def _shaped(block: Any, shape: tuple[int, ...]) -> Any:
    return ca.reshape(block, shape[0], shape[1]) if len(shape) == 2 else block


class Binding:
    """CasADi expressions for the Params of a set.

    Live Params are slices of one symbol ``p``; the others are folded in as constants at their
    current values.
    """

    def __init__(self, params: ParamSet, live: Iterable[str] = (), symbol: Any = ca.SX) -> None:
        live = set(live)
        unknown = live - set(params)
        if unknown:
            raise KeyError(f"unknown live parameters {sorted(unknown)}")
        self.params = params
        self.live = [name for name in params if name in live]
        self.p = symbol.sym("p", params.size(self.live))
        self._expr: dict[int, Any] = {}
        offset = 0
        for name, param in params.items():
            if name in live:
                block = self.p[offset : offset + param.size]
                offset += param.size
            else:
                block = ca.DM(param.value.ravel(order="F"))
            self._expr[id(param)] = _shaped(block, param.shape)

    def __call__(self, param: Param) -> Any:
        """Expression of ``param``."""
        try:
            return self._expr[id(param)]
        except KeyError:
            raise KeyError(f"{param!r} is not in this binding") from None

    def view(self, params: Mapping[str, Param]) -> dict[str, Any]:
        """Expressions of a model's Params, keyed by the model's own names."""
        return {name: self(param) for name, param in params.items()}

    def values(self) -> np.ndarray:
        """Current values of the live Params, packed like ``p``."""
        return self.params.vector(self.live)


def constants(params: Mapping[str, Param]) -> dict[str, Any]:
    """Current values of Params as CasADi constants, keyed by name (a numeric view)."""
    return {name: _shaped(ca.DM(p.value.ravel(order="F")), p.shape) for name, p in params.items()}
