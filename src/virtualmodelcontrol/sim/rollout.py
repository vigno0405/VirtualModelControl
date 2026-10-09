"""Closed-loop rollouts of a virtual mechanism system, compiled with CasADi, and its ODE."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..compiler import compile as compile_system
from ..dynamics import compile_dynamics
from ..system import VirtualMechanismSystem

INTEGRATORS = ("implicit", "rk4", "cvodes")
TOLERANCE = 1e-8  # relative and absolute, of CVODES


def rollout(
    system: VirtualMechanismSystem,
    q0: ArrayLike,
    T: float,
    dt: float,
    *,
    v0: ArrayLike | None = None,
    integrator: str = "implicit",
    max_step: float = 1e-3,
    runtime: Iterable[str] = (),
    p: Any = None,
    z0: ArrayLike | None = None,
    u: ArrayLike | None = None,
    output: Iterable[Any] = (),
) -> dict[str, Any]:
    """Simulate the closed loop for ``T`` [s]: the controller every ``dt``, held, and the robot
    in steps of at most ``max_step`` of ``integrator`` (``implicit``, ``rk4``, ``cvodes``).

    Row k of ``t``, ``q``, ``v``, ``z`` (if the controller has virtual states) and ``u`` (motor
    torques) is the state at the start of step k and the command computed from it, like
    ``run``. ``p`` replaces the live Params (as ``compiled.live_values()``); a CasADi MX symbol
    gives the results as MX expressions to differentiate. With ``u``, motor torques [N·m] with one
    row per step, the robot gets them instead of the controller's command (open loop, as in a
    logged run). ``output`` lists output stages with a ``symbolic`` form
    (``control.StaticFrictionCompensation``), applied to the controller's command as
    ``VMCController(output=)`` applies them.
    """
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator takes one of {INTEGRATORS}, got {integrator!r}")
    compiled, dynamics, where = _compile(system, runtime)
    space = system.robot.model.space
    nq, nv, nz = space.nq, space.nv, compiled.z0.size // 2
    nx, steps = nq + nv, round(T / dt)
    rhs = _rhs(space, dynamics)
    advance = _advance(rhs, dynamics, space, integrator, dt, max_step)

    nu = dynamics.n_u
    given = None if u is None else np.asarray(u, dtype=float)
    if given is not None and given.shape != (steps, nu):
        raise ValueError(f"u needs a row of {nu} torques for each of {steps} steps")
    x, tw = ca.MX.sym("x", nx + 2 * nz), ca.MX.sym("tw", 2 + (0 if given is None else nu))
    pp = ca.MX.sym("p", compiled.live_values().size)
    q, v, z, t, w = x[:nq], x[nq:nx], x[nx:], tw[0], tw[1]
    command, zdot = compiled.law(q, v, z, pp, t)
    for stage in output:
        command = stage.symbolic(command)
    if given is not None:
        command = tw[2:]
    zvel = z[nz:] + w * zdot[nz:]  # the virtual state moves over the time since the last step
    zpos = z[:nz] + w * zvel
    robot = advance(x[:nx], command, pp[where], t)
    control = ca.Function(
        "control_step", [x, tw, pp], [ca.vertcat(robot, zpos, zvel), ca.vertcat(x, command)]
    )

    symbolic = isinstance(p, ca.MX)
    if p is None:
        p = compiled.live_values()
    p = p if symbolic else ca.DM(np.asarray(p, dtype=float).reshape(-1, 1))
    z_start = compiled.z0 if z0 is None else np.asarray(z0, dtype=float)
    v_start = np.zeros(nv) if v0 is None else np.asarray(v0, dtype=float)
    x0 = np.concatenate([np.asarray(q0, dtype=float), v_start, z_start])
    times = dt * np.arange(steps)
    tws = np.vstack([times, np.r_[0.0, np.full(steps - 1, dt)]])  # t_k, and dt (0 at the start)
    if given is not None:
        tws = np.vstack([tws, given.T])
    _, rows = control.mapaccum(steps)(x0, tws, ca.repmat(p, 1, steps))
    if not symbolic:
        rows = np.array(ca.evalf(rows))
    out = {"t": times, "q": rows[:nq, :].T, "v": rows[nq:nx, :].T, "u": rows[nx + 2 * nz :, :].T}
    if nz:
        out["z"] = rows[nx : nx + 2 * nz, :].T
    return out


def ode(
    system: VirtualMechanismSystem, runtime: Iterable[str] = ()
) -> Callable[[float, np.ndarray], np.ndarray]:
    """The closed loop in continuous time, f(t, x) with x = [q, v, z positions, z velocities],
    for SciPy's ``solve_ivp``. The controller is evaluated continuously, not sampled."""
    compiled, dynamics, where = _compile(system, runtime)
    space = system.robot.model.space
    nq, nv, nz = space.nq, space.nv, compiled.z0.size // 2
    nx = nq + nv
    t, x = ca.MX.sym("t"), ca.MX.sym("x", nx + 2 * nz)
    p = ca.DM(compiled.live_values())
    u, zdot = compiled.law(x[:nq], x[nq:nx], x[nx:], p, t)
    xdot = ca.vertcat(_rhs(space, dynamics)(x[:nx], u, p[where], t), zdot)
    f = ca.Function("ode", [t, x], [xdot])
    return lambda t, x: np.array(f(t, x)).ravel()


def _compile(system: VirtualMechanismSystem, runtime: Iterable[str]) -> tuple[Any, Any, list[int]]:
    """The compiled controller, the robot's dynamics and where the dynamics' live Params sit in
    the controller's p."""
    runtime = list(runtime)
    compiled = compile_system(system, runtime)
    dynamics = compile_dynamics(system.robot, runtime, system.actuation)
    slices = compiled.live_slices()
    where = [i for name in dynamics.live for i in range(slices[name].start, slices[name].stop)]
    return compiled, dynamics, where


def _rhs(space: Any, dynamics: Any) -> ca.Function:
    """The robot's ODE, (x, u, p, t) → ẋ with x = [q, v]."""
    nq, nx = space.nq, space.nq + space.nv
    x, t = ca.MX.sym("x", nx), ca.MX.sym("t")
    u, p = ca.MX.sym("u", dynamics.n_u), ca.MX.sym("p", dynamics.live_values().size)
    xdot = ca.vertcat(
        ca.mtimes(space.velocity_map(x[:nq]), x[nq:]), dynamics.forward(x[:nq], x[nq:], u, p, t)
    )
    return ca.Function("rhs", [x, u, p, t], [xdot])


def _advance(
    rhs: ca.Function, dynamics: Any, space: Any, integrator: str, dt: float, max_step: float
) -> ca.Function:
    """The robot over ``dt`` [s] with the command held: (x, u, p, t) → x, x = [q, v]."""
    nq, nu = space.nq, dynamics.n_u
    x, t = ca.MX.sym("x", nq + space.nv), ca.MX.sym("t")
    u, p = ca.MX.sym("u", nu), ca.MX.sym("p", dynamics.live_values().size)
    if integrator in ("implicit", "rk4"):
        n = max(1, math.ceil(dt / max_step - 1e-9))
        h = dt / n
        if integrator == "implicit":
            x_next = ca.vertcat(*dynamics.step(x[:nq], x[nq:], u, p, t, h))
        else:
            k1 = rhs(x, u, p, t)
            k2 = rhs(x + h / 2 * k1, u, p, t + h / 2)
            k3 = rhs(x + h / 2 * k2, u, p, t + h / 2)
            k4 = rhs(x + h * k3, u, p, t + h)
            x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sub = ca.Function("substep", [x, u, p, t], [x_next])
        times = t + h * ca.DM(np.arange(n)).T
        xf = sub.mapaccum(n)(x, ca.repmat(u, 1, n), ca.repmat(p, 1, n), times)[:, -1]
    else:  # CasADi's integrators run over [0, dt]; the time offset and the held inputs are p
        tau, w = ca.MX.sym("tau"), ca.MX.sym("w", nu + p.numel() + 1)
        uw, pw, tw = w[:nu], w[nu:-1], w[-1] + tau
        dae = {"x": x, "p": w, "t": tau, "ode": rhs(x, uw, pw, tw)}
        args = {"x0": x, "p": ca.vertcat(u, p, t)}
        opts = {"abstol": TOLERANCE, "reltol": TOLERANCE, "max_step_size": max_step}
        xf = ca.integrator("integrator", integrator, dae, 0.0, dt, opts)(**args)["xf"]
    return ca.Function("advance", [x, u, p, t], [xf])
