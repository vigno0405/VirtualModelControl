"""Virtual states in a plan: a controller's own states, as unknowns advanced as it advances them."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .builder import Builder

NO_Z = ca.DM.zeros(0, 1)


def start_of(compiled: Any, given: Any, what: str) -> np.ndarray:
    """The virtual state a controller starts from: ``given`` (positions, then velocities), or the
    state it is compiled with."""
    if given is None:
        return np.array(compiled.z0, dtype=float)
    z = np.asarray(given, dtype=float).ravel()
    if z.size != compiled.z0.size:
        raise ValueError(f"{what} needs {compiled.z0.size} entries (positions, then velocities)")
    return z


def nodes(builder: Builder, name: str, start: np.ndarray, count: int) -> list[Any]:
    """The virtual state at ``count`` nodes: unknowns that start from ``start``, or empty
    columns when the controller has no virtual states."""
    size = start.size
    if not size:
        return [NO_Z] * count
    z = builder.variables.add(name, count * size, -np.inf, np.inf, np.tile(start, count))
    return [z[k * size : (k + 1) * size] for k in range(count)]


def advance(z: Any, zdot: Any, w: Any) -> Any:
    """The virtual state ``w`` [s] on: the controller's own semi-implicit Euler (velocities
    first), the same as ``VMCController.step`` and ``vmc.sim.rollout``."""
    half = z.shape[0] // 2
    velocity = z[half:] + w * zdot[half:]
    return ca.vertcat(z[:half] + w * velocity, velocity)


def blended_law(builder: Builder, compiled: Any, hold: tuple[Any, Any, np.ndarray] | None) -> Any:
    """``law(q, v, z, z_old, t, w, p)`` gives the motor torques of the controller with weight
    ``w`` and of the one in place (compiled, live Params, virtual state) with weight ``1 - w``,
    and the rates of both virtual states, at the state of the plant."""

    def law(q: Any, v: Any, z: Any, z_old: Any, tk: Any, w: float, p: Any) -> tuple[Any, Any, Any]:
        u, zdot = builder.command(compiled, q, v, z, p, tk)
        if hold is None:
            return u, zdot, NO_Z
        u_old, zdot_old = builder.command(hold[0], q, v, z_old, hold[1], tk)
        return (u if w >= 1.0 else w * u + (1.0 - w) * u_old), zdot, zdot_old

    return law
