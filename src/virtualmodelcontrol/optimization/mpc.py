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
    interval the time falls in), since the robot has moved on by then. A plan whose solve did not
    converge is not applied and the next solve starts from the measured state, unless ``rti``;
    ``applied`` says whether the last plan was. With ``rti`` the solver takes one SQP iteration
    per step (the real-time iteration) instead of solving to convergence, sets the problem's
    solver to ``"rti"``, and a later ``MPC`` of the same problem without it puts the previous
    solver back.
    A controller with virtual states is planned from the state it runs in (``controller.z``):
    give the ``Shooting`` a ``z0``. The controller's time counts from the start of each plan.
    ``start`` and ``poll`` do the same in a thread, so that a control loop is not held while a
    plan is made (a solve lets the other threads run); ``start`` waits until the last plan has
    been polled, and ``reset`` drops a plan that is being made. The solver is created with the
    MPC, so set the problem's solver and options before it.
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
            if problem.solver != "rti":
                problem._solver_before_rti = problem.solver
            problem.solver = "rti"
        elif problem.solver == "rti" and problem._solver_before_rti is not None:
            problem.solver, problem._solver_before_rti = problem._solver_before_rti, None
        self._start = (f"{shooting.name}.q0", f"{shooting.name}.v0")
        self._z = None if shooting._z0 is None else f"{shooting.name}.z0"
        self._level = next((f"{b.name}.level" for b in blocks if isinstance(b, TankBudget)), None)
        extra = [name for name in (self._z, self._level) if name]
        problem.parameter(*self._start, *shooting.steps, *extra)
        self.nlp = problem.build()
        if self.nlp.trajectory.z and self.nlp.trajectory.z[0].shape[0] and self._z is None:
            raise ValueError(
                "the controller has virtual states: give the Shooting z0, the state it runs from"
            )
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
        self._epoch = 0  # counts the resets: a plan made before one is dropped
        self._lock = threading.Lock()  # a plan is stored, or dropped by a reset, whole
        self.applied = False

    def reset(self) -> None:
        """Forget the previous plan, also one that is being made: the next solve starts from the
        measured state."""
        with self._lock:
            self._warm, self.result, self._pending = None, None, None
            self._epoch += 1
        if self._thread is not None and not self._thread.is_alive():  # a plan nobody polled
            self._thread.join()
            self._thread, self._error = None, None

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
        result = self._solve(self._inputs(controller, q, v, references), self._epoch)
        return self._apply(controller, result, result.seconds if latency is None else latency)

    def start(
        self,
        controller: Any,
        q: ArrayLike,
        v: ArrayLike,
        references: dict[str, ArrayLike] | None = None,
    ) -> bool:
        """Start planning in a thread, from the measured state, so that a control loop goes on
        meanwhile. Returns False, and does nothing, while the last plan is being made or has not
        been polled yet."""
        if self._thread is not None:
            return False
        inputs = self._inputs(controller, q, v, references)
        self._pending, self._error, self._started = None, None, time.perf_counter()
        self._thread = threading.Thread(target=self._run, args=(inputs, self._epoch), daemon=True)
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
        pending, self._pending = self._pending, None
        if pending is None:  # a reset dropped it
            return None
        return self._apply(controller, pending, late)

    def _run(self, inputs: tuple[dict[str, Any], dict[str, Any]], epoch: int) -> None:
        try:
            result = self._solve(inputs, epoch)
            with self._lock:
                self._pending = result if epoch == self._epoch else None
        except BaseException as error:  # raised again by poll, in the control loop
            self._error = error

    def _inputs(
        self, controller: Any, q: ArrayLike, v: ArrayLike, references: dict[str, ArrayLike] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """What a solve needs from the controller and the measurement, read at this moment."""
        q, v = np.asarray(q, dtype=float).ravel(), np.asarray(v, dtype=float).ravel()
        live = {name: np.array(value) for name, value in controller.live_params().items()}
        refs: dict[str, Any] = {**(references or {}), self._start[0]: q, self._start[1]: v}
        if self._z is not None:
            refs[self._z] = np.array(controller.z)
        refs.update({name: live[name] for name in self.names})
        if self._level is not None:
            refs[self._level] = controller.level
        return refs, live

    def _solve(self, inputs: tuple[dict[str, Any], dict[str, Any]], epoch: int) -> Result:
        refs, live = inputs
        warm = self._warm
        if warm is None:
            q, v, rows = refs[self._start[0]], refs[self._start[1]], self.intervals
            warm = {
                "q": np.tile(q, (rows + 1, 1)),
                "v": np.tile(v, (rows + 1, 1)),
                "steps": {name: np.stack([live[name]] * rows) for name in self.names},
            }
            if self._z is not None:
                warm["z"] = np.tile(refs[self._z], (rows + 1, 1))
        result = self.problem.solve(refs, warm_start=warm)
        with self._lock:
            if epoch == self._epoch:  # not after a reset: the plan is of the past
                self.result = result
                usable = result.converged or self.problem.solver == "rti"
                self._warm = self._shifted(result) if usable else None
        return result

    def _apply(self, controller: Any, result: Result, late: float) -> float:
        """Set the plan's Params as of ``late`` seconds into it, if it converged (or is a
        real-time iteration)."""
        self.interval = min(int(late / self.dt), self.intervals - 1)
        self.applied = result.converged or self.problem.solver == "rti"
        if not self.applied:
            return 0.0
        return float(result.apply(controller, self.interval) or 0.0)

    def _shifted(self, plan: Result) -> dict[str, Any]:
        """The plan one control period on: its nodes and intervals moved up, the last repeated."""
        nodes = np.minimum(np.arange(self.intervals + 1) + self.shift, self.intervals)
        rows = np.minimum(np.arange(self.intervals) + self.shift, self.intervals - 1)
        shifted = {
            "q": plan.q[nodes],
            "v": plan.v[nodes],
            "steps": {name: value[rows] for name, value in plan.steps.items()},
            "params": plan.params,
        }
        if plan.z.size:
            shifted["z"] = plan.z[nodes]
        return shifted
