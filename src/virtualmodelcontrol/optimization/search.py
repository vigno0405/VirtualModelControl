"""Where the references sit, searched on a grid: one reference at a time, the best plan kept."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .problem import Problem
from .result import Result


def sphere_points(center: ArrayLike, radius: float, step: float) -> np.ndarray:
    """The grid points ``step`` apart inside the ball of ``radius`` around ``center``, the centre
    first, then by distance. Shape (points, len(center))."""
    center = np.asarray(center, dtype=float).ravel()
    n = int(np.floor(radius / step + 1e-9))
    axis = np.arange(-n, n + 1) * step
    offsets = np.array(
        [
            o
            for o in itertools.product(axis, repeat=center.size)
            if np.dot(o, o) <= radius**2 + 1e-12
        ]
    )
    offsets = offsets[np.argsort(np.linalg.norm(offsets, axis=1), kind="stable")]
    return center + offsets


@dataclass
class Search:
    """The best plan found, the references that gave it, and every point tried."""

    result: Result
    references: dict[str, np.ndarray]
    candidates: list[dict[str, Any]]


def search_references(
    problem: Problem,
    anchors: Mapping[str, ArrayLike],
    radius: float,
    step: float,
    *,
    passes: int = 1,
    progress: Callable[[int, Result | None, str, dict[str, Any] | None], None] | None = None,
) -> Search:
    """Search the parameters in ``anchors`` (name → point) over the grid points of the balls of
    ``radius`` [m] around them, ``step`` [m] apart.

    Each parameter in turn takes every point of its ball while the others keep their best so far;
    the converged plan with the lowest cost wins. The program is built once and every point is
    another solve, warm-started from the best plan so far. The search repeats until a pass
    changes nothing, at most ``passes`` times. A point whose solve fails just loses. It returns the
    best converged plan, or the plan with every parameter at its anchor when none converged.
    ``progress(solved, best, stage, candidate)`` is called before and after every point.
    """
    names = list(anchors)
    if not names:
        raise ValueError("give at least one anchor: a parameter and the point to search around")
    parameters = problem.build().parameters
    unknown = [name for name in names if name not in parameters]
    if unknown:
        raise KeyError(
            f"{unknown} are not parameters of the problem (see Problem.parameter); "
            f"they are {list(parameters)}"
        )
    grids = [sphere_points(anchors[name], radius, step) for name in names]
    choice = (0,) * len(names)
    solved: dict[tuple[int, ...], Result | None] = {}
    errors: dict[tuple[int, ...], str] = {}
    candidates: list[dict[str, Any]] = []
    best: tuple[int, ...] | None = None

    def cost(c: tuple[int, ...]) -> float:
        result = solved[c]
        return result.cost if result is not None and result.converged else np.inf

    def report(stage: str, candidate: dict[str, Any] | None = None) -> None:
        if progress is not None:
            progress(len(solved), None if best is None else solved[best], stage, candidate)

    for _ in range(passes):
        start = choice
        for i, name in enumerate(names):
            for j in range(len(grids[i])):
                c = (*choice[:i], j, *choice[i + 1 :])
                if c in solved:
                    continue
                refs = {n: grids[g][c[g]] for g, n in enumerate(names)}
                stage = f"{name}, point {j + 1}/{len(grids[i])}"
                report(f"{stage}: solving")
                t0 = time.perf_counter()
                try:
                    solved[c] = problem.solve(refs, None if best is None else solved[best])
                except Exception as error:  # a point that crashes just loses
                    solved[c], errors[c] = None, repr(error)
                result = solved[c]
                if cost(c) < (np.inf if best is None else cost(best)):
                    best = c
                candidate = {
                    "references": {n: r.tolist() for n, r in refs.items()},
                    "status": errors[c] if result is None else result.status,
                    "cost": None if result is None else result.cost,
                    "iterations": None if result is None else result.iterations,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
                candidates.append(candidate)
                report(stage, candidate)
            if best is not None:
                choice = best
        if choice == start:
            break

    final = best if best is not None else (0,) * len(names)
    result = solved[final]
    if result is None:
        raise RuntimeError(f"no grid point could be solved: {errors[final]}")
    points = {n: grids[g][final[g]] for g, n in enumerate(names)}
    return Search(result=result, references=points, candidates=candidates)
