"""Problem: a system, the Params to optimize, and the blocks that make the program."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.params import ParamSet
from ..mechanisms.coordinates.base import walk
from ..system import VirtualMechanismSystem
from .builder import Builder, param_bounds
from .nlp import NLP
from .result import Result
from .solver import IterationCallback, create_solver

Progress = Callable[[int, float, dict[str, np.ndarray], np.ndarray], "bool | None"]
WARM_START = ("q", "v", "a", "params")


class Problem:
    """Optimize the Params of a ``VirtualMechanismSystem`` over a planned motion.

    Add a ``Collocation`` (or an ``Equilibrium``) and terms (``Effort``, ``Cost``, ``Bound``).
    ``free`` chooses the Params
    to optimize (bounds and scale come from the Param) and ``parameter`` the ones set at every
    solve (references, say). The program is built at the first ``solve`` and kept: the values of
    those Params and the bounds of the free ones are read again at every solve, every other Param
    stays at its value when the program was built. ``options`` overrides IPOPT's defaults
    (``optimization.IPOPT``); changing ``options`` later makes the next solve create the solver
    again.
    """

    def __init__(self, system: VirtualMechanismSystem, *, options: Mapping[str, Any] | None = None):
        self.system = system
        self.options: dict[str, Any] = dict(options or {})
        self.params = ParamSet()
        for name, param in system.params.items():
            self.params.add(param, name)
        self._blocks: list[Any] = []
        self._free: list[str] = []
        self._parameters: list[str] = []
        self._nlp: NLP | None = None
        self._solver: Any = None
        self._solver_options: dict[str, Any] = {}
        self._callback: IterationCallback | None = None
        self._progress: Progress | None = None
        self._failure: BaseException | None = None
        self._iteration = 0

    def add(self, block: Any) -> Any:
        """Add a ``Collocation``, an ``Equilibrium`` or a term (see ``Term``); returns it, named
        uniquely."""
        if any(block is b for b in self._blocks):
            raise ValueError(f"{block.name!r} is in the problem already; add a new term")
        motion = getattr(block, "motion", False)
        has_motion = any(getattr(b, "motion", False) for b in self._blocks)
        if motion and has_motion:
            raise ValueError("a problem has one Collocation or Equilibrium")
        if not motion and not has_motion:
            raise ValueError(f"add the Collocation or Equilibrium before the term {block.name!r}")
        taken = {b.name for b in self._blocks}
        name, k = block.name, 2
        while name in taken:
            name, k = f"{block.name}{k}", k + 1
        block.name = name
        for coord in block.coordinates():
            for node in walk(coord):
                for local, param in node.params().items():
                    self.params.add(param, f"{name}.{local}", rename=True)
        self._blocks.append(block)
        self._invalidate()
        return block

    def free(self, *patterns: str) -> list[str]:
        """Optimize the Params with these names (glob patterns such as ``ctrl.*.stiffness``)."""
        names = self._select(patterns)
        for name in names:
            if name in self._parameters:
                raise ValueError(f"{name!r} is a parameter already")
            if name not in self._free:
                self._free.append(name)
        self._invalidate()
        return names

    def parameter(self, *patterns: str) -> list[str]:
        """Make the Params with these names inputs of each solve (``solve(references=...)``)."""
        names = self._select(patterns)
        for name in names:
            if name in self._free:
                raise ValueError(f"{name!r} is free already")
            if name not in self._parameters:
                self._parameters.append(name)
        self._invalidate()
        return names

    def build(self) -> NLP:
        """The program, built at the first call and kept."""
        if self._nlp is None:
            builder = Builder(self.system, self.params, self._free, self._parameters)
            for block in self._blocks:
                block.build(builder)
            self._nlp = builder.finish()
        return self._nlp

    def initial_guess(self, warm_start: Result | Mapping[str, Any] | None = None) -> np.ndarray:
        """The solver's starting point (scaled): the robot at rest at ``q0`` with the free Params
        at their values, or ``warm_start``, a previous ``Result`` or a dict with ``q``, ``v``,
        ``a`` (nodes × size) and ``params``."""
        nlp = self.build()
        variables, x = nlp.variables, nlp.x0.copy()
        for name in nlp.free:
            key = f"param:{name}"
            x[variables.slices[key]] = variables.scaled(key, self.params[name].value.ravel("F"))
        if warm_start is None:
            return x
        if isinstance(warm_start, Result):
            get = lambda key: getattr(warm_start, key)  # noqa: E731
        elif isinstance(warm_start, Mapping):
            unknown = sorted(set(warm_start) - set(WARM_START))
            if unknown:
                raise ValueError(f"warm_start has {unknown}; its keys are {list(WARM_START)}")
            get = warm_start.get
        else:
            raise TypeError("warm_start is a Result or a dict of q, v, a and params")
        for key, shape in nlp.trajectory.shapes.items():
            value = get(key)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != shape:
                    raise ValueError(f"warm start {key!r} has shape {value.shape}, not {shape}")
                x[variables.slices[key]] = variables.scaled(key, value)
        for name, value in (get("params") or {}).items():
            if name in nlp.free:
                key = f"param:{name}"
                value = np.asarray(value, dtype=float).reshape(self.params[name].shape)
                x[variables.slices[key]] = variables.scaled(key, value.ravel("F"))
        return x

    def solve(
        self,
        references: Mapping[str, ArrayLike] | None = None,
        warm_start: Result | Mapping[str, Any] | None = None,
        *,
        progress: Progress | None = None,
    ) -> Result:
        """Run the solver. ``references`` gives the parameters' values (the others keep their
        current ones). ``progress(iteration, cost, params, q)`` is called at every iteration;
        return True from it to stop (status ``User_Requested_Stop``). An exception raised in it
        stops the solve and is raised again here.
        """
        nlp = self.build()
        refs, p = self._references(nlp, references)
        x0 = self.initial_guess(warm_start)
        lbx, ubx = self._bounds(nlp)
        solver = self._solver_for(nlp)
        self._progress, self._failure, self._iteration = progress, None, 0
        start = time.perf_counter()
        try:
            solution = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=nlp.lbg, ubg=nlp.ubg)
        finally:
            failure, self._progress, self._failure = self._failure, None, None
        if failure is not None:
            raise failure
        seconds = time.perf_counter() - start
        stats = solver.stats()
        x = np.array(solution["x"]).ravel()
        found = nlp.unpack(x)
        g = np.array(solution["g"]).ravel()
        violation = max(
            float(np.max(nlp.lbg - g, initial=0.0)), float(np.max(g - nlp.ubg, initial=0.0))
        )
        parts = nlp.costs(x, p) if nlp.cost_names else ()
        parts = parts if isinstance(parts, (tuple, list)) else (parts,)
        return Result(
            t=nlp.trajectory.t.copy(),
            q=found["q"],
            v=found["v"],
            a=found["a"],
            u=np.array(nlp.outputs["u"](x, p)).T,
            blend=nlp.trajectory.blend.copy(),
            params=found["params"],
            references=refs,
            cost=float(solution["f"]),
            costs={name: float(part) for name, part in zip(nlp.cost_names, parts, strict=True)},
            status=str(stats["return_status"]),
            iterations=int(stats.get("iter_count", 0)),
            seconds=seconds,
            violation=violation,
        )

    def _select(self, patterns: tuple[str, ...]) -> list[str]:
        names: list[str] = []
        for pattern in patterns:
            if not isinstance(pattern, str):
                raise TypeError(
                    f"give the names as separate strings, not a {type(pattern).__name__}"
                )
            found = self.params.select(patterns=[pattern])
            if not found:
                raise KeyError(f"no Param matches {pattern!r}; the names are {list(self.params)}")
            names += [name for name in found if name not in names]
        return names

    def _references(
        self, nlp: NLP, references: Mapping[str, ArrayLike] | None
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        given = dict(references or {})
        unknown = sorted(set(given) - set(nlp.parameters))
        if unknown:
            raise KeyError(
                f"{unknown} are not parameters of this problem (declare them with "
                f"Problem.parameter); the parameters are {list(nlp.parameters)}"
            )
        values: dict[str, np.ndarray] = {}
        p = np.zeros(nlp.p.numel())
        for name, where in nlp.parameters.items():
            param = self.params[name]
            value = np.array(given.get(name, param.value), dtype=float)
            if value.size != param.size:
                raise ValueError(f"{name!r} needs {param.size} values, got {value.size}")
            if value.shape != param.shape:
                value = value.reshape(param.shape)
            values[name] = value
            p[where] = value.ravel(order="F")
        return values, p

    def _bounds(self, nlp: NLP) -> tuple[np.ndarray, np.ndarray]:
        """The bounds of the variables, with the free Params' current bounds."""
        lbx, ubx = nlp.lbx.copy(), nlp.ubx.copy()
        for name in nlp.free:
            key = f"param:{name}"
            where, scale = nlp.variables.slices[key], nlp.variables.scales[key]
            lower, upper = param_bounds(self.params[name])
            lbx[where], ubx[where] = lower / scale, upper / scale
        return lbx, ubx

    def _solver_for(self, nlp: NLP) -> Any:
        if self._solver is None or self._solver_options != self.options:
            self._callback = IterationCallback(
                "progress", nlp.x.numel(), nlp.g.numel(), self._on_iteration
            )
            self._solver = create_solver(nlp.problem, self.options, self._callback)
            self._solver_options = dict(self.options)
        return self._solver

    def _on_iteration(self, x: np.ndarray, cost: float) -> bool:
        """One IPOPT iteration: tell the user's callback; True stops the solver."""
        iteration, self._iteration = self._iteration, self._iteration + 1
        if self._progress is None or self._failure is not None:
            return self._failure is not None
        found = self.build().unpack(x)
        try:
            return bool(self._progress(iteration, cost, found["params"], found["q"]))
        except BaseException as error:  # stop the solver cleanly, raise the error after it
            self._failure = error
            return True

    def _invalidate(self) -> None:
        self._nlp = None
        self._solver = None
