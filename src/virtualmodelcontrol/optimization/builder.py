"""What a block adds to: variables, constraints and costs, and the system's compiled functions."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ..compiler import compile as compile_system
from ..core.params import Binding, Param
from ..dynamics import compile_dynamics
from ..mechanisms.coordinates.base import Context, Coordinate, walk
from ..mechanisms.coordinates.joints import State
from .nlp import NLP
from .trajectory import Trajectory
from .variables import Variables

OPTS = {"cse": True}


def param_bounds(param: Param) -> tuple[np.ndarray, np.ndarray]:
    """The Param's lower and upper bounds, one entry per entry of its value (column by column)."""
    flat = []
    for bound in param.bounds:
        b = np.asarray(bound, dtype=float)
        flat.append(np.broadcast_to(b.reshape(()) if b.size == 1 else b, param.shape).ravel("F"))
    return flat[0], flat[1]


def literal(name: str) -> str:
    """A Param name as a glob pattern that matches only itself."""
    return "".join(f"[{c}]" if c in "*?[" else c for c in name)


class Builder:
    """Assembles one problem. A Param is a decision variable (``free``), an input of the program
    (``parameters``, set at every solve) or a constant at its current value."""

    def __init__(self, system: Any, params: Any, free: list[str], parameters: list[str]) -> None:
        self.system = system
        self.params = params
        self.free = free
        self.parameters = parameters
        declared = [literal(name) for name in (*free, *parameters)]
        self.compiled = compile_system(system, runtime=declared)
        self.dynamics = compile_dynamics(system.robot, declared, system.actuation)
        if self.compiled.z0.size:
            raise NotImplementedError("controllers with virtual states cannot be planned yet")
        self.variables = Variables()
        self.trajectory: Trajectory | None = None
        self._constraints: list[tuple[str, Any, Any, Any]] = []
        self._costs: list[tuple[str, Any]] = []
        self._outputs: dict[str, Any] = {}
        self._values: dict[int, Any] = {}
        self._free_ids = {id(params[name]) for name in free}
        self._free_shapes: dict[str, tuple[int, ...]] = {}
        self._made_free = False
        self._p_slices: dict[str, slice] = {}
        size = sum(params[name].size for name in parameters)
        self._p = ca.MX.sym("p", size)
        offset = 0
        for name in parameters:
            param = params[name]
            self._p_slices[name] = slice(offset, offset + param.size)
            self._values[id(param)] = self._p[offset : offset + param.size]
            offset += param.size
        self._functions: dict[int, tuple[ca.Function, list[str]]] = {}

    def value(self, param: Param) -> Any:
        """The Param as a column: a decision variable, an input or a constant."""
        if id(param) in self._free_ids:
            self.make_free()
        if id(param) in self._values:
            return self._values[id(param)]
        return ca.DM(param.value.ravel(order="F"))

    def make_free(self) -> None:
        """Create the decision variables of the free Params, in the order they were declared."""
        if self._made_free:
            return
        self._made_free = True
        for name in self.free:
            param = self.params[name]
            shape = param.shape
            lower, upper = param_bounds(param)
            self._values[id(param)] = self.variables.add(
                f"param:{name}",
                param.size,
                lower,
                upper,
                param.value.ravel(order="F"),
                param.scale,
            )
            self._free_shapes[name] = shape

    def pack(self, params: Any, live: list[str]) -> Any:
        """The vector ``p`` of a compiled function: its live Params, in its order."""
        parts = [self.value(params[name]) for name in live]
        return ca.vertcat(*parts) if parts else ca.DM.zeros(0, 1)

    def constrain(self, name: str, expr: Any, lower: Any, upper: Any) -> None:
        """Add the rows ``lower ≤ expr ≤ upper`` as a group called ``name``."""
        if any(name == existing for existing, *_ in self._constraints):
            raise ValueError(f"a constraint group named {name!r} already exists; rename the term")
        self._constraints.append((name, expr, lower, upper))

    def cost(self, name: str, expr: Any) -> None:
        """Add a term ``expr`` (a scalar) to the cost, under ``name``."""
        self._costs.append((name, expr))

    def output(self, name: str, expr: Any) -> None:
        """Make ``expr`` available as a function of (x, p) after the solve."""
        self._outputs[name] = expr

    def evaluate(
        self, coord: Coordinate, qs: list[Any], vs: list[Any], ts: Any
    ) -> tuple[list[Any], list[Any]]:
        """Value and rate of a coordinate at the given nodes."""
        fn, live = self._coordinate(coord)
        p = self.pack(self.params, live)
        ys, yds = [], []
        for q, v, t in zip(qs, vs, ts, strict=True):
            y, yd = fn(q, v, p, float(t))
            ys.append(y)
            yds.append(yd)
        return ys, yds

    def _coordinate(self, coord: Coordinate) -> tuple[ca.Function, list[str]]:
        """y(q, v, p, t) and ẏ of a coordinate, built once."""
        key = id(coord)
        if key not in self._functions:
            if any(isinstance(node, State) for node in walk(coord)):
                raise ValueError(
                    "a task coordinate can depend on the robot, the Params and time only, "
                    "not on a controller's virtual states"
                )
            space = self.system.robot.model.space
            q, v, t = ca.SX.sym("q", space.nq), ca.SX.sym("v", space.nv), ca.SX.sym("t")
            binding = Binding(self.params, [*self.free, *self.parameters])
            y = Context(q, binding, t=t).value(coord)
            Jq = ca.mtimes(ca.jacobian(y, q), space.velocity_map(q))
            yd = ca.mtimes(Jq, v) + ca.jacobian(y, t)
            fn = ca.Function("coordinate", [q, v, binding.p, t], [y, yd], OPTS)
            self._functions[key] = (fn, binding.live)
        return self._functions[key]

    def finish(self) -> NLP:
        """The program: variables, parameters, cost and constraints, as built."""
        if self.trajectory is None:
            raise ValueError("add a Collocation to the problem before solving it")
        self.make_free()
        variables = self.variables
        rows: dict[str, slice] = {}
        g, lbg, ubg, start = [], [], [], 0
        for name, expr, lower, upper in self._constraints:
            n = int(expr.shape[0])
            g.append(expr)
            lbg.append(np.broadcast_to(np.asarray(lower, dtype=float), n))
            ubg.append(np.broadcast_to(np.asarray(upper, dtype=float), n))
            rows[name] = slice(start, start + n)
            start += n
        x, p = variables.x, self._p
        f = sum((expr for _, expr in self._costs), ca.MX(0))
        names = [name for name, _ in self._costs]
        outputs = {name: ca.Function(name, [x, p], [expr]) for name, expr in self._outputs.items()}
        return NLP(
            x=x,
            p=p,
            f=f,
            g=ca.vertcat(*g),
            lbx=variables.lower,
            ubx=variables.upper,
            x0=variables.init,
            lbg=np.concatenate(lbg),
            ubg=np.concatenate(ubg),
            variables=variables,
            constraints=rows,
            costs=ca.Function("costs", [x, p], [expr for _, expr in self._costs]),
            cost_names=names,
            outputs=outputs,
            free=dict(self._free_shapes),
            parameters=dict(self._p_slices),
            trajectory=self.trajectory,
        )
