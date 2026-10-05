"""A robot's diagonal stiffness and damping, fitted to logged motion under known motor torques."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import casadi as ca
import numpy as np

from ..core.params import constants
from ..dynamics import compile_dynamics
from ..models.actuation import Direct

MOVING = 0.01
"""A motor moves, for the friction column, above this fraction of the fastest rate in the run."""


def fit_stiffness_damping(
    robot: Any,
    runs: Iterable[Mapping[str, Any]],
    *,
    baseline: int = 50,
    smoothing: int = 51,
    stride: int = 1,
    runtime: Iterable[str] = (),
    friction: bool = False,
) -> tuple[np.ndarray, ...]:
    """Diagonal stiffness K and damping D that best explain logged runs, with K, D ≥ 0.

    ``robot`` holds the known parts (masses, gravity), not the stiffness and damping. Each run
    is a ``RunLog``, or has ``t`` [s], ``q``, ``v`` and the motor torques (``motor_torque`` or
    ``u``), one row per sample, and optionally
    ``train``, the samples to fit, and ``p``, its values of the Params matching ``runtime`` (its
    gravity, say). Everything is relative to the run's first ``baseline``
    samples, at rest, so constant offsets drop out; in motor torques,

        A (K Δq + D v) = Δu − A (M a + h − (g(q) − g(q₀))),

    with A the allocation of generalized forces to motors. v is smoothed (Savitzky-Golay,
    ``smoothing`` samples); ``stride`` keeps every stride-th sample. With ``friction``, each motor
    also loses a static friction torque F ≥ 0 against its motion, F sign(θ̇), and ``F`` [N·m] is
    returned after K and D. A motor that stands still has none: step and settle runs hardly show
    friction, which then hides in K.
    """
    from scipy.optimize import lsq_linear
    from scipy.signal import savgol_filter

    dyn = compile_dynamics(robot, runtime)
    actuation = robot.actuation if robot.actuation is not None else Direct()
    space = robot.model.space
    q_s, tau_s = ca.SX.sym("q", space.nq), ca.SX.sym("tau", space.nv)
    allocated = actuation.allocate(tau_s, q_s, constants(actuation.params))
    allocation = ca.Function("allocation", [q_s], [ca.jacobian(allocated, tau_s)])
    v_s = ca.SX.sym("v", space.nv)
    rates = actuation.motor_rates(q_s, v_s, constants(actuation.params))
    motor_rates = ca.Function("motor_rates", [q_s, v_s], [rates])
    rows, rhs = [], []
    for run in runs:
        run = run.arrays() if hasattr(run, "arrays") else run  # a RunLog, or a mapping
        t = np.asarray(run["t"], dtype=float).ravel()
        torque = run["motor_torque"] if "motor_torque" in run else run["u"]
        q, v, u = (np.asarray(x, dtype=float) for x in (run["q"], run["v"], torque))
        p = np.asarray(run.get("p", dyn.live_values()), dtype=float)
        dt = float(np.median(np.diff(t)))
        v = savgol_filter(v, min(smoothing, len(t) - (1 - len(t) % 2)), 3, axis=0)
        a = np.gradient(v, dt, axis=0)
        keep = np.asarray(run.get("train", np.ones(len(t))), dtype=bool).ravel()
        idx = np.flatnonzero(keep & (np.arange(len(t)) % stride == 0))
        q0, u0 = q[:baseline].mean(axis=0), u[:baseline].mean(axis=0)
        zero, no_torque = np.zeros(space.nv), np.zeros(u.shape[1])
        r0 = np.array(dyn.residual(q0, zero, zero, no_torque, p, 0.0)).ravel()  # −g(q₀)
        m = len(idx)
        residual = dyn.residual.map(m)(
            q[idx].T, v[idx].T, a[idx].T, np.zeros((u.shape[1], m)), np.tile(p, (m, 1)).T, t[idx]
        )
        known = np.array(residual).T - r0  # M a + h − (g(q) − g(q₀)), one row per sample
        A = np.array(allocation.map(m)(q[idx].T)).T.reshape(m, space.nv, -1).transpose(0, 2, 1)
        columns = [A * (q[idx] - q0)[:, None, :], A * v[idx][:, None, :]]
        if friction:
            rate = np.array(motor_rates.map(m)(q[idx].T, v[idx].T)).T
            sign = np.sign(rate) * (np.abs(rate) > MOVING * np.abs(rate).max())
            columns.append(sign[:, :, None] * np.eye(u.shape[1]))
        rows.append(np.concatenate(columns, axis=2))
        rhs.append(u[idx] - u0 - np.einsum("kmn,kn->km", A, known))
    A_all = np.concatenate(rows).reshape(-1, rows[0].shape[2])
    x = lsq_linear(A_all, np.concatenate(rhs).ravel(), bounds=(0.0, np.inf), method="trf").x
    nv = space.nv
    return (x[:nv], x[nv : 2 * nv], x[2 * nv :]) if friction else (x[:nv], x[nv:])
