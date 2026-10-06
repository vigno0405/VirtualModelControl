"""MPC: the plan as a receding-horizon controller of a virtual mechanism's Params."""

from __future__ import annotations

import threading
import time
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
    The controller's time counts from the start of each plan. ``start`` and ``poll`` do the same
    in a thread, so that a control loop is not held while a plan is made (a solve lets the other
    threads run). The solver is created with the MPC, so set the problem's solver and options
    before it.
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
        problem.warm_up()  # a thread only runs the solver, it does not create it
        stepped = self.nlp.trajectory.stepped  # type: ignore[attr-defined]
        self.names = list(stepped)
        self.intervals = len(self.nlp.trajectory.t) - 1
        self.dt = float(self.nlp.trajectory.dt)
        self.result: Result | None = None
        self.interval = 0
        self._warm: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._pending: Result | None = None
        self._error: BaseException | None = None
        self._started = 0.0

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
        result = self._solve(self._inputs(controller, q, v, references))
        return self._apply(controller, result, result.seconds if latency is None else latency)

    def start(
        self,
        controller: Any,
        q: ArrayLike,
        v: ArrayLike,
        references: dict[str, ArrayLike] | None = None,
    ) -> bool:
        """Start planning in a thread, from the measured state, so that a control loop goes on
        meanwhile. Returns False, and does nothing, while the last plan is still being made."""
        if self.busy:
            return False
        inputs = self._inputs(controller, q, v, references)
        self._pending, self._error, self._started = None, None, time.perf_counter()
        self._thread = threading.Thread(target=self._run, args=(inputs,), daemon=True)
        self._thread.start()
        return True

    @property
    def busy(self) -> bool:
        """True while a plan started with ``start`` is being made."""
        return self._thread is not None and self._thread.is_alive()

    def poll(self, controller: Any, *, latency: float | None = None) -> float | None:
        """Apply the plan started with ``start`` once it is ready, and return the jump of the
        controller's energy [J]; None while it is not. ``latency`` [s] is the time since the start
        by default, which is how late the plan is: it is applied as of that time."""
        if self._thread is None or self._thread.is_alive():
            return None
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise self._error
        late = time.perf_counter() - self._started if latency is None else latency
        assert self._pending is not None
        return self._apply(controller, self._pending, late)

    def _run(self, inputs: tuple[dict[str, Any], dict[str, Any]]) -> None:
        try:
            self._pending = self._solve(inputs)
        except BaseException as error:  # raised again by poll, in the control loop
            self._error = error

    def _inputs(
        self, controller: Any, q: ArrayLike, v: ArrayLike, references: dict[str, ArrayLike] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """What a solve needs from the controller and the measurement, read at this moment."""
        q, v = np.asarray(q, dtype=float).ravel(), np.asarray(v, dtype=float).ravel()
        live = {name: np.array(value) for name, value in controller.live_params().items()}
        refs: dict[str, Any] = {**(references or {}), self._start[0]: q, self._start[1]: v}
        refs.update({name: live[name] for name in self.names})
        if self._level is not None:
            refs[self._level] = controller.level
        return refs, live

    def _solve(self, inputs: tuple[dict[str, Any], dict[str, Any]]) -> Result:
        refs, live = inputs
        if self._warm is None:
            q, v, rows = refs[self._start[0]], refs[self._start[1]], self.intervals
            self._warm = {
                "q": np.tile(q, (rows + 1, 1)),
                "v": np.tile(v, (rows + 1, 1)),
                "steps": {name: np.stack([live[name]] * rows) for name in self.names},
            }
        result = self.problem.solve(refs, warm_start=self._warm)
        self.result = result
        self._warm = self._shifted(result)
        return result

    def _apply(self, controller: Any, result: Result, late: float) -> float:
        """Set the plan's Params as of ``late`` seconds into it."""
        self.interval = min(int(late / self.dt), self.intervals - 1)
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
