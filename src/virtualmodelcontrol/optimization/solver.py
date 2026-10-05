"""The nonlinear solver: IPOPT with the options that suit collocation problems, and its progress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np

CONVERGED = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
"""IPOPT's two successful endings."""

IPOPT: dict[str, Any] = {
    # One flat expression graph: an iteration costs about a tenth of the nested function calls.
    "expand": True,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",  # no licence banner in the terminal
    "print_time": 0,
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-6,
    # Also stop once the plan obeys every constraint and the cost has stopped changing for 10
    # iterations: a saturated spring leaves the cost almost flat in its stiffness, and IPOPT
    # would creep toward the optimality tolerance for hundreds of iterations.
    "ipopt.acceptable_tol": 1e-2,
    "ipopt.acceptable_constr_viol_tol": 1e-4,
    "ipopt.acceptable_obj_change_tol": 1e-4,
    "ipopt.acceptable_iter": 10,
    # A quasi-Newton Hessian: the exact one differentiates the kinematics twice. Gradients stay
    # exact.
    "ipopt.hessian_approximation": "limited-memory",
}
"""Default IPOPT options; ``Problem(options=...)`` overrides single entries."""


class IterationCallback(ca.Callback):
    """Calls ``fn(x, cost)`` once per IPOPT iteration, with the scaled variables; the solver
    stops when it returns True."""

    def __init__(
        self, name: str, nx: int, ng: int, fn: Callable[[np.ndarray, float], bool | None]
    ) -> None:
        ca.Callback.__init__(self)
        self._nx, self._ng, self._fn = nx, ng, fn
        self.construct(name, {})

    def get_n_in(self) -> int:
        """All of nlpsol's outputs."""
        return int(ca.nlpsol_n_out())

    def get_n_out(self) -> int:
        """One output: non-zero asks the solver to stop."""
        return 1

    def get_sparsity_in(self, i: int) -> Any:
        """Shapes of nlpsol's outputs; only the variables and the cost are read."""
        name = ca.nlpsol_out(i)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self._nx)
        if name in ("g", "lam_g"):
            return ca.Sparsity.dense(self._ng)
        return ca.Sparsity(0, 0)

    def eval(self, arg: Any) -> list[Any]:
        """One iteration."""
        return [1 if self._fn(np.array(arg[0]).ravel(), float(arg[1])) else 0]


def create_solver(
    nlp: dict[str, Any],
    options: dict[str, Any],
    callback: IterationCallback | None = None,
) -> Any:
    """IPOPT on ``nlp`` (x, p, f, g), with ``options`` over the defaults."""
    opts = {**IPOPT, **options}
    if callback is not None:
        opts["iteration_callback"] = callback
    return ca.nlpsol("solver", "ipopt", nlp, opts)
