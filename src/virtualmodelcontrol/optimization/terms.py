"""Terms of a problem: the effort, a coordinate to bring to zero, a coordinate to keep in bounds."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..mechanisms.coordinates.base import Coordinate
from .builder import Builder, param_bounds
from .trajectory import Trajectory


def _per_entry(value: ArrayLike, dim: int, what: str, name: str) -> np.ndarray:
    """``value`` as one number per entry of a coordinate with ``dim`` entries."""
    value = np.asarray(value, dtype=float)
    if value.size not in (1, dim):
        raise ValueError(f"{name!r}: {what} needs 1 or {dim} values, got {value.size}")
    return np.broadcast_to(value.reshape(-1), dim).copy()


def _no_node(name: str, trajectory: Trajectory, t_from: float, t_to: float | None) -> ValueError:
    end = "the end" if t_to is None else f"{t_to:g} s"
    step = trajectory.t[1] - trajectory.t[0] if trajectory.t.size > 1 else 0.0
    return ValueError(
        f"no node lies in the window of {name!r}, from {t_from:g} s to {end}: the nodes are "
        f"{step:g} s apart from 0 to {trajectory.t[-1]:g} s"
    )


class Term:
    """A cost and/or constraints over the nodes of the planned motion; subclass it to add your own.

    ``cost(trajectory)`` returns a scalar CasADi expression (or None). ``constraints(trajectory)``
    returns ``[(expr, lower, upper), ...]`` with ``expr`` a column. List the task coordinates the
    term uses in ``coordinates()``, so the problem knows their Params.
    """

    name = "term"

    def coordinates(self) -> tuple[Coordinate, ...]:
        """The library coordinates this term evaluates."""
        return ()

    def cost(self, trajectory: Trajectory) -> Any:
        """The term's cost, or None."""
        return None

    def constraints(self, trajectory: Trajectory) -> list[tuple[Any, Any, Any]]:
        """The term's constraints, as (expression, lower, upper)."""
        return []

    def build(self, builder: Builder) -> None:
        """Add the term to the problem."""
        trajectory = builder.trajectory
        if trajectory is None:
            raise ValueError(f"add a Collocation or an Equilibrium before the term {self.name!r}")
        cost = self.cost(trajectory)
        if cost is not None:
            builder.cost(self.name, cost)
        rows = self.constraints(trajectory)
        if rows:
            exprs, lowers, uppers = [], [], []
            for expr, lower, upper in rows:
                column = ca.reshape(expr, -1, 1)
                exprs.append(column)
                lowers.append(np.broadcast_to(np.asarray(lower, dtype=float), column.shape[0]))
                uppers.append(np.broadcast_to(np.asarray(upper, dtype=float), column.shape[0]))
            builder.constrain(
                self.name, ca.vertcat(*exprs), np.concatenate(lowers), np.concatenate(uppers)
            )


class Effort(Term):
    """``weight`` × the integral of the squared motor torques over the horizon (trapezoid rule)."""

    def __init__(self, weight: float, *, name: str = "effort") -> None:
        self.weight, self.name = float(weight), name

    def cost(self, trajectory: Trajectory) -> Any:
        """weight · Σ w_k Δt ‖u_k‖², with w_k = ½ at the first and last node."""
        last = len(trajectory.u) - 1
        total = ca.MX(0)
        for k, u in enumerate(trajectory.u):
            total = total + (0.5 if k in (0, last) else 1.0) * trajectory.dt * ca.sumsqr(u)
        return self.weight * total


class Cost(Term):
    """``weight`` × the integral of the squared coordinate from ``t_from`` on: arrive and stay.

    The sum counts every node of the window ``t_from ≤ t ≤ t_to`` fully, the last one by half.
    ``weight`` is a number, or one per entry of the coordinate.
    """

    def __init__(
        self,
        coordinate: Coordinate,
        weight: ArrayLike = 1.0,
        *,
        t_from: float = 0.0,
        t_to: float | None = None,
        name: str = "cost",
    ) -> None:
        self.coordinate, self.t_from, self.t_to, self.name = coordinate, t_from, t_to, name
        self.weight = _per_entry(weight, coordinate.dim, "the weight", name)

    def coordinates(self) -> tuple[Coordinate, ...]:
        """The coordinate."""
        return (self.coordinate,)

    def cost(self, trajectory: Trajectory) -> Any:
        """weight · Σ w_k Δt ‖y_k‖² over the window."""
        nodes = trajectory.window(self.t_from, self.t_to)
        if not nodes:
            raise _no_node(self.name, trajectory, self.t_from, self.t_to)
        values, _ = trajectory.coordinate(self.coordinate)
        weight = ca.DM(self.weight)
        total = ca.MX(0)
        for k in nodes:
            half = 0.5 if k == nodes[-1] else 1.0
            total = total + half * trajectory.dt * ca.dot(weight, values[k] ** 2)
        return total


class Bound(Term):
    """The coordinate stays within ``lower`` and ``upper`` (numbers, or one per entry) at the
    nodes of the window ``t_from ≤ t ≤ t_to``, except the first node when it is the fixed start
    (a periodic motion has none)."""

    def __init__(
        self,
        coordinate: Coordinate,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        *,
        t_from: float = 0.0,
        t_to: float | None = None,
        name: str = "bound",
    ) -> None:
        if lower is None and upper is None:
            raise ValueError("give a lower bound, an upper bound or both")
        self.coordinate, self.t_from, self.t_to, self.name = coordinate, t_from, t_to, name
        dim = coordinate.dim
        self.lower = _per_entry(-np.inf if lower is None else lower, dim, "the lower bound", name)
        self.upper = _per_entry(np.inf if upper is None else upper, dim, "the upper bound", name)
        if np.any(self.lower > self.upper):
            raise ValueError(f"{name!r}: the lower bound is above the upper bound")

    def coordinates(self) -> tuple[Coordinate, ...]:
        """The coordinate."""
        return (self.coordinate,)

    def constraints(self, trajectory: Trajectory) -> list[tuple[Any, Any, Any]]:
        """The window's values of the coordinate, between the bounds."""
        first = 1 if trajectory.fixed_start else 0
        nodes = [k for k in trajectory.window(self.t_from, self.t_to) if k >= first]
        if not nodes:
            raise _no_node(self.name, trajectory, self.t_from, self.t_to)
        values, _ = trajectory.coordinate(self.coordinate)
        lower, upper = np.tile(self.lower, len(nodes)), np.tile(self.upper, len(nodes))
        return [(ca.vertcat(*[values[k] for k in nodes]), lower, upper)]


class Period(Term):
    """The horizon equals the Param called ``param``: a controller whose reference repeats every
    ``param`` seconds (a function of ``vmc.Time()`` and that Param) has its orbit repeat with it.

    Make the Param free or a parameter to tie the free horizon of a ``Collocation`` to the period.
    """

    def __init__(self, param: str, *, name: str = "period") -> None:
        self.param, self.name = param, name

    def build(self, builder: Builder) -> None:
        """Constrain the horizon to the Param's value."""
        trajectory = builder.trajectory
        if trajectory is None:
            raise ValueError(f"add a Collocation before the term {self.name!r}")
        found = builder.params.select(patterns=[self.param])
        if len(found) != 1:
            raise ValueError(
                f"{self.name!r}: {self.param!r} matches {len(found)} Params, it must be one "
                f"of {list(builder.params)}"
            )
        param = builder.params[found[0]]
        if param.size != 1:
            raise ValueError(
                f"{self.name!r}: a period is one number, {found[0]!r} has {param.size}"
            )
        period = ca.reshape(builder.value(param), 1, 1)
        builder.constrain(self.name, trajectory.horizon - period, 0.0, 0.0)


class Sparsity(Term):
    """``weight`` × the sum of the free Params matching ``patterns``, each of them at least 0.

    This L1 norm drives a gate, or a gain, to exactly 0 when keeping it earns less than ``weight``
    for each unit of it.
    """

    def __init__(self, weight: float, *patterns: str, name: str = "sparsity") -> None:
        self.weight, self.patterns, self.name = float(weight), patterns, name

    def build(self, builder: Builder) -> None:
        """Add the sum of the free Params whose names match."""
        names = [n for n in builder.params.select(patterns=self.patterns) if n in builder.free]
        if not names:
            raise ValueError(f"{self.name!r}: no free Param matches {list(self.patterns)}")
        total = ca.MX(0)
        for name in names:
            if np.any(param_bounds(builder.params[name])[0] < 0):
                raise ValueError(f"{name!r} can be negative: the sum is an L1 norm only above 0")
            total = total + ca.sum1(builder.value(builder.params[name]))
        builder.cost(self.name, self.weight * total)
