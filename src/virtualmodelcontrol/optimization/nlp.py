"""The assembled nonlinear program, ready for any solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .trajectory import Trajectory
from .variables import Variables


@dataclass
class NLP:
    """min f(x, p) subject to lbg ≤ g(x, p) ≤ ubg and lbx ≤ x ≤ ubx.

    ``x`` holds the scaled variables, ``p`` the parameters (``parameters`` gives their slices).
    ``constraints`` names the groups of rows of ``g``; ``costs`` maps (x, p) to each term's cost
    and ``outputs`` to other results, such as the motor torques ``u``.
    """

    x: Any
    p: Any
    f: Any
    g: Any
    lbx: np.ndarray
    ubx: np.ndarray
    x0: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray
    variables: Variables
    constraints: dict[str, slice]
    costs: ca.Function
    cost_names: list[str]
    outputs: dict[str, ca.Function]
    free: dict[str, tuple[int, ...]]
    parameters: dict[str, slice]
    trajectory: Trajectory

    @property
    def problem(self) -> dict[str, Any]:
        """The dict CasADi's ``nlpsol`` takes."""
        return {"x": self.x, "p": self.p, "f": self.f, "g": self.g}

    def functions(self) -> tuple[ca.Function, ca.Function]:
        """The cost f(x, p) and the constraints g(x, p) as CasADi functions."""
        return (
            ca.Function("f", [self.x, self.p], [self.f]),
            ca.Function("g", [self.x, self.p], [self.g]),
        )

    def unpack(self, x: np.ndarray) -> dict[str, Any]:
        """Physical ``q``, ``v``, ``a`` (nodes × size), ``params`` (free Params by name, each in
        its shape) and ``steps`` (the Params of a shooting that change at every interval, by name,
        each as intervals × its shape) from a scaled solution vector."""
        x = np.asarray(x, dtype=float).ravel()
        out: dict[str, Any] = {
            key: self.variables.value(x, key).reshape(shape)
            for key, shape in self.trajectory.shapes.items()
        }
        out["steps"] = {}
        for name, (count, *shape) in getattr(self.trajectory, "stepped", {}).items():
            rows = self.variables.value(x, f"step:{name}").reshape(count, -1)
            out["steps"][name] = np.stack([row.reshape(shape, order="F") for row in rows])
        out["params"] = {
            name: self.variables.value(x, f"param:{name}").reshape(shape, order="F")
            for name, shape in self.free.items()
        }
        return out
