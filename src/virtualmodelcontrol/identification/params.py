"""Any Params of a robot, fitted to logged motion by least squares on its dynamics residual."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..dynamics import compile_dynamics
from .stiffness import _read


@dataclass
class Fit:
    """What ``fit_params`` found: ``values`` and their standard errors ``std`` [same units] by
    Param name, in each Param's own shape, and the root-mean-square ``rms`` of the residual
    [N·m or N] over the samples. A Param that the data do not determine (the run never excites
    it, or it only appears in a sum with another) has ``std`` ``inf``, and the fit leaves it
    where it started, along every direction that the data do not see."""

    values: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    rms: float


def _blind(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The scale of each unknown (the norm of its column of the Jacobian J, 1 for a zero column)
    and the directions of the scaled unknowns that J does not see, one per column. The columns are
    scaled to one so that unknowns of different units are compared alike."""
    norms = np.linalg.norm(J, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    scaled = J / norms
    w, V = np.linalg.eigh(scaled.T @ scaled)
    return norms, V[:, w <= 1e-12 * max(w[-1], np.finfo(float).tiny)]


def _undetermined(J: np.ndarray) -> np.ndarray:
    """The unknowns whose value the data cannot fix: those that can move along a direction that
    J does not see."""
    return np.linalg.norm(_blind(J)[1], axis=1) > 1e-6


def fit_params(
    robot: Any,
    names: Iterable[str],
    runs: Iterable[Mapping[str, Any]],
    *,
    smoothing: int = 51,
    stride: int = 1,
) -> Fit:
    """The values of the named Params that best explain logged runs, within the Params' bounds.

    ``names`` are the robot's Param names or globs (``"m_*.mass"``); the others keep their values.
    Each run is a ``RunLog``, or has ``t`` [s], ``q``, ``v`` and the motor torques (``motor_torque``
    or ``u``), one row per sample, and optionally ``train`` (the samples to fit) and ``a``. The
    unknowns are the Params in the residual of the robot's own dynamics at the logged motion,
    M(q) a + h(q, v) − f(q, v; θ) − B(q) u. Without ``a``, the acceleration is v smoothed
    (Savitzky-Golay, ``smoothing`` samples) and differentiated, against the torques averaged over
    the step as it was held; ``stride`` keeps every stride-th sample. A residual linear in the
    Params (masses, stiffnesses, dampings, efficiencies) is solved in a step, and any other by
    Gauss-Newton from the Params' current values.
    """
    from scipy.optimize import least_squares
    from scipy.signal import savgol_filter

    chosen = robot.params.select(patterns=names)
    if not chosen:
        raise ValueError(f"no Param of {robot.name!r} matches {list(names)}")
    dyn = compile_dynamics(robot, runtime=[f"{robot.name}.{n}" for n in chosen])
    sizes = {n: dyn.params[n].size for n in dyn.live}
    start = {n: sum(sizes[k] for k in dyn.live[: dyn.live.index(n)]) for n in dyn.live}
    full = [f"{robot.name}.{n}" for n in chosen]
    columns = np.concatenate([np.arange(start[n], start[n] + sizes[n]) for n in full])
    lower, upper = (
        np.concatenate(
            [
                np.broadcast_to(dyn.params[n].bounds[side], dyn.params[n].shape).ravel(order="F")
                for n in full
            ]
        )
        for side in (0, 1)
    )

    space = robot.model.space
    q, v, a = ca.SX.sym("q", space.nq), ca.SX.sym("v", space.nv), ca.SX.sym("a", space.nv)
    u, t = ca.SX.sym("u", dyn.n_u), ca.SX.sym("t")
    p = ca.SX.sym("p", dyn.params.size(dyn.live))
    residual = dyn.residual(q, v, a, u, p, t)
    terms = ca.Function(
        "terms", [q, v, a, u, p, t], [residual, ca.jacobian(residual, p)[:, columns.tolist()]]
    )

    samples = []
    for run in runs:
        rows, ts, qs, vs, us, keep = _read(run)
        if "a" in rows:
            accel = np.asarray(rows["a"], dtype=float)
        else:
            dt = float(np.median(np.diff(ts)))
            vs = savgol_filter(vs, min(smoothing, len(ts) - (1 - len(ts) % 2)), 3, axis=0)
            accel = np.gradient(vs, dt, axis=0)
            # a torque is held over the step after its sample, and the difference of the
            # velocities at t_k sees the mean of the torques before and after it
            us = 0.5 * (us + np.concatenate([us[:1], us[:-1]]))
        idx = np.flatnonzero(keep & (np.arange(len(ts)) % stride == 0))
        samples.append((ts[idx], qs[idx], vs[idx], accel[idx], us[idx]))
    t_all, q_all, v_all, a_all, u_all = (np.concatenate(x) for x in zip(*samples, strict=True))
    m = len(t_all)
    mapped = terms.map(m)
    p0 = dyn.live_values()

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        live = np.tile(p0[:, None], (1, m))
        live[columns] = x[:, None]
        r, J = (np.array(o) for o in mapped(q_all.T, v_all.T, a_all.T, u_all.T, live, t_all))
        k = len(columns)  # J holds the samples side by side: (nv, m k)
        return r.T.ravel(), J.reshape(space.nv, m, k).transpose(1, 0, 2).reshape(-1, k)

    x0 = np.clip(p0[columns], lower, upper)
    found = least_squares(
        lambda x: evaluate(x)[0],
        x0,
        jac=lambda x: evaluate(x)[1],
        bounds=(lower, upper),
        x_scale="jac",
    )
    x = found.x
    r, J = evaluate(x)
    # along a direction the data do not see the solver ends wherever its rounding leaves it, which
    # depends on the machine: take that part of the move back, if it is inside the bounds and the
    # residual hardly feels it (a hundred-millionth of the misfit at the start)
    norms, blind = _blind(J)
    if blind.size:
        step = norms * (x - x0)
        back = x0 + (step - blind @ (blind.T @ step)) / norms
        r_back, r_start = evaluate(back)[0], evaluate(x0)[0]
        if np.all((lower <= back) & (back <= upper)) and (
            np.linalg.norm(r_back - r) <= 1e-8 * np.linalg.norm(r_start)
        ):
            x = back
            r, J = evaluate(x)
    dof = max(len(r) - len(columns), 1)
    cov = (r @ r / dof) * np.linalg.pinv(J.T @ J)
    sigma = np.sqrt(np.diag(cov))
    sigma[_undetermined(J)] = np.inf
    values, std, at = {}, {}, 0
    for name, short in zip(full, chosen, strict=True):
        shape, n = dyn.params[name].shape, dyn.params[name].size
        values[short] = x[at : at + n].reshape(shape, order="F")
        std[short] = sigma[at : at + n].reshape(shape, order="F")
        at += n
    return Fit(values, std, float(np.sqrt(np.mean(r**2))))
