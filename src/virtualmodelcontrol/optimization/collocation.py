"""Collocation: the whole motion at once, with the closed loop as constraints."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..compiler import compile as compile_system
from ..control.blending import blend_weight
from ..core.space import Euclidean, Product
from ..system import VirtualMechanismSystem
from .builder import Builder
from .trajectory import Trajectory
from .virtual import blended_law, nodes, start_of


def _flat(space: Any) -> bool:
    """True for Euclidean spaces and their products, where q + v·dt stays on the space."""
    if isinstance(space, Product):
        return all(_flat(s) for s in space.spaces)
    return isinstance(space, Euclidean)


def _snapshot(initial: Any) -> tuple[Any, np.ndarray, np.ndarray, bool]:
    """The controller in place as it is now: its compiled law, its live Param values, its virtual
    state and whether it is running (a system starts with the plan)."""
    if isinstance(initial, VirtualMechanismSystem):
        compiled = compile_system(initial)
        values = np.array(compiled.live_values(), dtype=float)
        return compiled, values, compiled.z0.copy(), False
    if hasattr(initial, "compiled") and hasattr(initial, "params"):  # a running controller
        z = getattr(initial, "z", initial.compiled.z0)  # where it is now
        params = np.array(initial.params, dtype=float)
        return initial.compiled, params, np.array(z, dtype=float), True
    raise TypeError(
        "initial is a VirtualMechanismSystem (robot and controller) or a running controller, "
        f"not a {type(initial).__name__}"
    )


SCHEMES = ("trapezoid", "hermite-simpson")


def _z_shape(z: np.ndarray, n: int) -> dict[str, tuple[int, int]]:
    """The shape of the virtual state's nodes, when there is one."""
    return {"z": (n, z.size)} if z.size else {}


class Collocation:
    """The closed loop's motion from ``q0`` over ``horizon`` [s] at ``nodes`` equally spaced nodes.

    q, v and a are unknowns at every node, tied by the robot's dynamics (without inverting the
    mass matrix) and the ``scheme`` that integrates them: the trapezoid rule (second order), or
    Hermite-Simpson (fourth order), which also has the acceleration at the middle of every
    interval as an unknown. The motion starts at rest (or at ``v0``).

    ``initial`` is the controller in place, as it is now: a ``VirtualMechanismSystem`` or a
    running ``VMCController`` on the same robot. Its torques fade out as the new ones fade in over
    ``transition`` [s], with the blend of ``SwapController``, and the model starts at rest where
    the robot is, carrying the force that balances it under ``initial``. Both controllers' time
    counts from the first node. ``scales`` are the typical sizes of q, v and a (the solver sees
    values divided by them).

    ``periodic=True`` plans an orbit that repeats: the last node equals the first (q and v), and
    the first is free, so ``q0`` and ``v0`` are only the solver's starting guess. With
    ``free_time=(lower, upper)`` the horizon is an unknown within these bounds, starting at
    ``horizon``; the windows of the terms still read the nodes' times at that starting horizon.
    Neither goes with ``initial``. Start a periodic problem from a guess of the orbit
    (``Problem.solve(warm_start=...)``), not from rest.

    A controller with virtual states (a flywheel, a tank) adds them as unknowns, with their own
    dynamics, starting from ``z0`` (positions, then velocities; the state it is compiled with by
    default). In a periodic motion they repeat too. The controller in place keeps its own, from
    where a running one is now.
    """

    name = "collocation"
    motion = True

    def __init__(
        self,
        q0: ArrayLike,
        horizon: float,
        nodes: int,
        *,
        v0: ArrayLike | None = None,
        initial: Any = None,
        transition: float = 0.0,
        scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
        scheme: str = "trapezoid",
        periodic: bool = False,
        free_time: tuple[float, float] | None = None,
        z0: ArrayLike | None = None,
    ) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"scheme takes one of {SCHEMES}, got {scheme!r}")
        if nodes < 2:
            raise ValueError("a collocation needs at least 2 nodes")
        if horizon <= 0.0:
            raise ValueError("the horizon must be positive")
        if transition < 0.0:
            raise ValueError("the transition cannot be negative")
        if len(scales) != 3 or min(scales) <= 0.0:
            raise ValueError("scales are three positive numbers: the sizes of q, v and a")
        if free_time is not None:
            if np.size(free_time) != 2 or not 0.0 < free_time[0] < free_time[1]:
                raise ValueError("free_time is (lower, upper), with 0 < lower < upper [s]")
            if not free_time[0] <= horizon <= free_time[1]:
                raise ValueError("the horizon is the starting value of free_time: put it inside")
            if initial is not None or transition > 0.0:
                raise ValueError(
                    "free_time cannot go with initial or transition: the blend needs the node "
                    "times as numbers"
                )
        if periodic and initial is not None:
            raise ValueError("a periodic motion repeats: it has no initial controller")
        self.q0 = np.asarray(q0, dtype=float).ravel()
        self.v0 = None if v0 is None else np.asarray(v0, dtype=float).ravel()
        self.horizon, self.nodes = float(horizon), int(nodes)
        self.transition = float(transition)
        self.scales, self.scheme = tuple(float(s) for s in scales), scheme
        self.periodic = bool(periodic)
        self.free_time = None if free_time is None else (float(free_time[0]), float(free_time[1]))
        self.z0 = None if z0 is None else np.asarray(z0, dtype=float).ravel()
        self._hold = None if initial is None else _snapshot(initial)

    def coordinates(self) -> tuple[Any, ...]:
        """No task coordinates."""
        return ()

    def build(self, builder: Builder) -> None:
        """Add the unknowns, the dynamics and integration constraints, and the trajectory."""
        space = builder.plant.model.space
        if not _flat(space):
            raise NotImplementedError(
                "collocation needs a Euclidean configuration space (or a product of them)"
            )
        nq, nv, n = space.nq, space.nv, self.nodes
        v0 = np.zeros(nv) if self.v0 is None else self.v0
        if self.q0.size != nq or v0.size != nv:
            raise ValueError(f"q0 and v0 need {nq} and {nv} entries, got {self.q0.size}, {v0.size}")
        nominal = self.horizon / (n - 1)
        t = np.arange(n) * nominal
        sq, sv, sa = self.scales
        inf = np.inf
        if self.free_time is None:
            horizon: Any = self.horizon
        else:
            lower, upper = self.free_time
            horizon = builder.variables.add("horizon", 1, lower, upper, self.horizon, self.horizon)
        dt = nominal if self.free_time is None else horizon / (n - 1)

        def node_time(k: int, fraction: float = 0.0) -> Any:
            """The time of node k plus a fraction of a step: a number, or the free horizon's."""
            if self.free_time is None:
                return float(t[k] + fraction * dt)
            return (k + fraction) * dt if k or fraction else 0.0

        v_guess = np.zeros(n * nv)
        v_guess[:nv] = v0
        Q = builder.variables.add("q", n * nq, -inf, inf, np.tile(self.q0, n), sq)
        V = builder.variables.add("v", n * nv, -inf, inf, v_guess, sv)
        A = builder.variables.add("a", n * nv, -inf, inf, np.zeros(n * nv), sa)
        qs = [Q[k * nq : (k + 1) * nq] for k in range(n)]
        vs = [V[k * nv : (k + 1) * nv] for k in range(n)]
        accel = [A[k * nv : (k + 1) * nv] for k in range(n)]

        compiled, dynamics = builder.compiled, builder.dynamics
        p_law = builder.pack(compiled.params, compiled.live)
        p_dyn = builder.pack(dynamics.params, dynamics.live)
        hold = self._check_hold(builder)
        blend = np.array([blend_weight(float(tk), self.transition) if hold else 1.0 for tk in t])
        z_new = start_of(compiled, self.z0, "z0")
        z_old = hold[2] if hold else np.zeros(0)
        zn = nodes(builder, "z", z_new, n)  # the new controller's state and the one in place
        zo = nodes(builder, "zh", z_old, n)
        zs = [ca.vertcat(zn[k], zo[k]) for k in range(n)]
        z_start = np.concatenate([z_new, z_old])

        bias = ca.DM.zeros(nv)
        if hold is not None:
            rest = ca.DM.zeros(nv)
            u_rest = builder.command(hold[0], ca.DM(self.q0), rest, z_old, hold[1], 0.0)[0]
            bias = dynamics.residual(ca.DM(self.q0), rest, rest, u_rest, p_dyn, 0.0)

        blended = blended_law(builder, compiled, hold)
        split = z_new.size

        def law(q: Any, v: Any, z: Any, tk: Any, w: float) -> tuple[Any, Any]:
            """The motor torques at (q, v, z, tk) and the rates of the virtual states of both
            controllers, z holding the state of the controller and then of the one in place."""
            u, zdot, zdot_old = blended(q, v, z[:split], z[split:], tk, w, p_law)
            return u, ca.vertcat(zdot, zdot_old)

        us, zds, defects = [], [], []
        for k in range(n):
            u, zd = law(qs[k], vs[k], zs[k], node_time(k), float(blend[k]))
            us.append(u)
            zds.append(zd)
            defects.append(
                dynamics.residual(qs[k], vs[k], accel[k], us[k], p_dyn, node_time(k)) - bias
            )
        if self.scheme == "trapezoid":
            positions = [
                space.difference(qs[k + 1], qs[k]) - 0.5 * dt * (vs[k] + vs[k + 1])
                for k in range(n - 1)
            ]
            velocities = [
                vs[k + 1] - vs[k] - 0.5 * dt * (accel[k] + accel[k + 1]) for k in range(n - 1)
            ]
            states = [zs[k + 1] - zs[k] - 0.5 * dt * (zds[k] + zds[k + 1]) for k in range(n - 1)]
        else:  # Hermite-Simpson: the middle of an interval from the cubic through its ends
            Am = builder.variables.add("am", (n - 1) * nv, -inf, inf, np.zeros((n - 1) * nv), sa)
            positions, velocities, states, middles = [], [], [], []
            for k in range(n - 1):
                a_c, tc = Am[k * nv : (k + 1) * nv], node_time(k, 0.5)
                q_c = 0.5 * (qs[k] + qs[k + 1]) + dt / 8 * (vs[k] - vs[k + 1])
                v_c = 0.5 * (vs[k] + vs[k + 1]) + dt / 8 * (accel[k] - accel[k + 1])
                z_c = 0.5 * (zs[k] + zs[k + 1]) + dt / 8 * (zds[k] - zds[k + 1])
                w_c = blend_weight(tc, self.transition) if hold else 1.0
                u_c, zd_c = law(q_c, v_c, z_c, tc, w_c)
                middles.append(dynamics.residual(q_c, v_c, a_c, u_c, p_dyn, tc) - bias)
                positions.append(
                    space.difference(qs[k + 1], qs[k]) - dt / 6 * (vs[k] + 4 * v_c + vs[k + 1])
                )
                velocities.append(vs[k + 1] - vs[k] - dt / 6 * (accel[k] + 4 * a_c + accel[k + 1]))
                states.append(zs[k + 1] - zs[k] - dt / 6 * (zds[k] + 4 * zd_c + zds[k + 1]))
        if self.periodic:
            wrap = ca.vertcat(space.difference(qs[-1], qs[0]), vs[-1] - vs[0], zn[-1] - zn[0])
            builder.constrain("periodic", wrap, 0.0, 0.0)
        else:
            start = ca.vertcat(qs[0] - self.q0, vs[0] - v0, zs[0] - z_start)
            builder.constrain("start", start, 0.0, 0.0)
        builder.constrain("dynamics", ca.vertcat(*defects), 0.0, 0.0)
        if self.scheme != "trapezoid":
            builder.constrain("middles", ca.vertcat(*middles), 0.0, 0.0)
        builder.constrain("position", ca.vertcat(*positions), 0.0, 0.0)
        builder.constrain("velocity", ca.vertcat(*velocities), 0.0, 0.0)
        if z_start.size:
            builder.constrain("virtual", ca.vertcat(*states), 0.0, 0.0)
        builder.output("u", ca.horzcat(*us))
        builder.output("horizon", ca.MX(horizon))
        builder.trajectory = Trajectory(
            t=t,
            dt=dt,
            q=qs,
            v=vs,
            a=accel,
            u=us,
            blend=blend,
            shapes={"q": (n, nq), "v": (n, nv), "a": (n, nv), **_z_shape(z_new, n)},
            evaluate=builder.evaluate,
            fixed_start=not self.periodic,
            z=zn,
        )

    def _check_hold(self, builder: Builder) -> tuple[Any, Any, np.ndarray] | None:
        """The initial controller's law, its (fixed) live Params and its virtual state, or None."""
        if self._hold is None:
            return None
        compiled, values, z, _ = self._hold
        if compiled.system.robot is not builder.system.robot:
            raise ValueError("the initial controller must control the same robot")
        return compiled, ca.DM(values), z
