"""The outcome of a solve: the planned motion, the values found, and how to apply them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .solver import CONVERGED


@dataclass
class Result:
    """A planned motion and the Params found.

    ``t`` [s], ``q``, ``v``, ``a`` and the motor torques ``u`` are per node (rows), ``blend`` is
    the weight of the new controller. ``params`` holds the free Params at their optimum and
    ``references`` the values the parameters had in this solve, both by name. ``cost`` is the
    total, ``costs`` the part of each term, ``violation`` the largest violation of a constraint.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    u: np.ndarray
    blend: np.ndarray
    params: dict[str, np.ndarray]
    references: dict[str, np.ndarray]
    cost: float
    costs: dict[str, float]
    status: str
    iterations: int
    seconds: float
    violation: float

    @property
    def converged(self) -> bool:
        """True if the solver ended with a solution (maybe an acceptable one, not an optimum)."""
        return self.status in CONVERGED

    def apply(self, target: Any) -> float | None:
        """Set the free Params and the references into ``target``.

        ``target`` is a running controller (its live Params change through ``set``, and the exact
        energy jump [J] is returned), or a ``VirtualMechanismSystem`` or ``ParamSet`` (the Params
        take the values; the next controller compiled from it has them). Params the target does
        not have, such as the target inside a ``Cost``, are left out; one that is not live in a
        running controller raises KeyError (compile it with ``runtime``).
        """
        values = {**self.references, **self.params}
        if hasattr(target, "compiled") and hasattr(target, "set"):
            known = target.compiled.params
            return float(target.set({n: v for n, v in values.items() if n in known}))
        params = target if hasattr(target, "items") else target.params
        for name, value in values.items():
            if name in params:
                params[name].value = value
        return None
