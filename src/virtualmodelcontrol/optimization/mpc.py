"""MPC: the plan as a receding-horizon controller of a virtual mechanism's Params."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .problem import Problem
from .result import Result
from .shooting import Shooting
from .tank import TankBudget


class MPC:
    """Plans the Params of a controller over a horizon again at every step, and applies the first.

    ``problem`` holds a ``Shooting`` (with the Params that ``steps``) and the terms. The program
    is built once: the measured state, the controller's Params now, the tank's level and the
    references are the parameters of each solve. ``step`` solves from the measured state,
    starting from the previous plan shifted by ``shift`` intervals, and sets the plan's Params on
    the controller through ``controller.set``, so that a ``Tank`` around it bounds what changes.
    A plan that took ``latency`` [s] to compute is applied as of that time (its value in the
    interval the time falls in), since the robot has moved on by then. With ``rti`` the solver
    takes one SQP iteration per step (the real-time iteration) instead of solving to convergence.
    The controller's time counts from the start of each plan.
    """

    def __init__(self, problem: Problem, *, shift: int = 1, rti: bool = False) -> None:
        blocks = problem._blocks
        shooting = next((b for b in blocks if isinstance(b, Shooting)), None)
        if shooting is None:
            raise ValueError("an MPC plans with a Shooting: add one to the problem")
        if shift < 1:
            raise ValueError("shift is at least 1 interval")
        self.problem, self.shift = problem, int(shift)
        if rti:
            problem.solver = "rti"
        self._start = (f"{shooting.name}.q0", f"{shooting.name}.v0")
        self._level = next((f"{b.name}.level" for b in blocks if isinstance(b, TankBudget)), None)
        problem.parameter(*self._start, *shooting.steps, *([self._level] if self._level else []))
        self.nlp = problem.build()
        stepped = self.nlp.trajectory.stepped  # type: ignore[attr-defined]
        self.names = list(stepped)
        self.intervals = len(self.nlp.trajectory.t) - 1
        self.dt = float(self.nlp.trajectory.dt)
        self.result: Result | None = None
        self.interval = 0
        self._warm: dict[str, Any] | None = None

    def reset(self) -> None:
        """Forget the previous plan: the next solve starts from the measured state."""
        self._warm, self.result = None, None

    def step(
        self,
        controller: Any,
        q: ArrayLike,
        v: ArrayLike,
        references: dict[str, ArrayLike] | None = None,
        *,
        latency: float | None = None,
    ) -> float:
        """Plan from the measured state ``(q, v)`` and apply it to ``controller`` (or a ``Tank``).

        ``references`` are the values of the problem's other parameters. ``latency`` [s] is how
        late the plan will be applied: the measured solve time by default. Returns the jump of
        the controller's energy [J] that was applied, as the adaptation laws do.
        """
        q, v = np.asarray(q, dtype=float).ravel(), np.asarray(v, dtype=float).ravel()
        live = controller.live_params()
        refs: dict[str, Any] = {**(references or {}), self._start[0]: q, self._start[1]: v}
        refs.update({name: live[name] for name in self.names})
        if self._level is not None:
            refs[self._level] = controller.level
        if self._warm is None:
            rows = self.intervals
            self._warm = {
                "q": np.tile(q, (rows + 1, 1)),
                "v": np.tile(v, (rows + 1, 1)),
                "steps": {name: np.stack([live[name]] * rows) for name in self.names},
            }
        result = self.problem.solve(refs, warm_start=self._warm)
        late = result.seconds if latency is None else latency
        self.interval = min(int(late / self.dt), self.intervals - 1)
        self.result = result
        self._warm = self._shifted(result)
        return float(result.apply(controller, self.interval) or 0.0)

    def _shifted(self, plan: Result) -> dict[str, Any]:
        """The plan one control period on: its nodes and intervals moved up, the last repeated."""
        nodes = np.minimum(np.arange(self.intervals + 1) + self.shift, self.intervals)
        rows = np.minimum(np.arange(self.intervals) + self.shift, self.intervals - 1)
        return {
            "q": plan.q[nodes],
            "v": plan.v[nodes],
            "steps": {name: value[rows] for name, value in plan.steps.items()},
            "params": plan.params,
        }
