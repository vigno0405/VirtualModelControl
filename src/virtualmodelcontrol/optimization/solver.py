"""The nonlinear solver: IPOPT and the other presets, and the progress of a solve."""

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


QUIET = {
    "qrqp": {"print_iter": False, "print_header": False, "print_info": False},
    "osqp": {"osqp": {"verbose": False}},
    "qpoases": {"printLevel": "none"},
}
"""Options that silence a QP solver of ``sqp``; the ones not listed here print what they like."""

SQP: dict[str, Any] = {
    "expand": True,
    "print_time": 0,
    "print_header": False,
    "print_iteration": False,
    "print_status": False,
    "max_iter": 100,
    "tol_pr": 1e-6,
    "tol_du": 1e-4,
    "qpsol": "qrqp",
    "qpsol_options": QUIET["qrqp"],
    # The exact Hessian of a plan is not convex: its negative eigenvalues are clipped.
    "convexify_strategy": "eigen-clip",
    "convexify_margin": 1e-6,
}
"""Default options of ``sqp``: CasADi's sequential quadratic programming, exact Hessians."""

FATROP: dict[str, Any] = {"expand": True, "print_time": 0, "fatrop.print_level": 0}
"""Default options of ``fatrop``, which treats the program as one general problem here."""

PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "ipopt": ("ipopt", IPOPT),
    "ipopt-exact": ("ipopt", {**IPOPT, "ipopt.hessian_approximation": "exact"}),
    "sqp": ("sqpmethod", SQP),
    "rti": ("sqpmethod", {**SQP, "max_iter": 1, "tol_pr": 1e-12, "tol_du": 1e-12}),
    "fatrop": ("fatrop", FATROP),
}
"""Solver presets by name: the CasADi plugin and its default options. ``rti`` is one SQP
iteration, the real-time iteration of a receding-horizon controller."""


def available(solver: str) -> bool:
    """True if this CasADi build has the plugin of the preset ``solver`` and its QP solver."""
    plugin, options = PRESETS[solver]
    qp = options.get("qpsol")
    return bool(ca.has_nlpsol(plugin)) and (qp is None or bool(ca.has_conic(qp)))


def create_solver(
    nlp: dict[str, Any],
    options: dict[str, Any],
    callback: IterationCallback | None = None,
    solver: str = "ipopt",
) -> Any:
    """The preset ``solver`` (a key of ``PRESETS``) on ``nlp`` (x, p, f, g), with ``options`` over
    its defaults."""
    if solver not in PRESETS:
        raise ValueError(f"solver takes one of {list(PRESETS)}, got {solver!r}")
    plugin, defaults = PRESETS[solver]
    opts = {**defaults, **options}
    if "qpsol" in options and "qpsol_options" not in options:
        opts["qpsol_options"] = QUIET.get(options["qpsol"], {})
    if callback is not None:
        opts["iteration_callback"] = callback
    return ca.nlpsol("solver", plugin, nlp, opts)


def status_of(stats: dict[str, Any]) -> str:
    """The solver's ending as IPOPT words it: FATROP reports a number and a success flag."""
    status = stats["return_status"]
    if isinstance(status, str):
        return status
    return "Solve_Succeeded" if stats.get("success") else f"Failed_{status}"
