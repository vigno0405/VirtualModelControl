"""Shooting: one rollout of the closed loop per interval, joined by continuity constraints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..control.blending import blend_weight
from ..mechanisms.coordinates.references import Ref
from ..sim.rollout import _advance, _rhs
from .builder import Builder, param_bounds
from .collocation import _snapshot
from .trajectory import Trajectory
from .virtual import advance, blended_law, nodes, start_of

INTEGRATORS = ("implicit", "rk4")


@dataclass
class ShootingTrajectory(Trajectory):
    """The nodes of a shooting motion, plus what a term needs of its intervals: ``stepped`` gives
    the shape (the intervals, then the Param's own) of each Param that changes at every interval,
    ``steps``
    its value per interval, ``params`` the live Params of the controller in each interval,
    ``now`` those before the motion starts, and ``dissipated`` the energy [J] that the
    controller's own dampers take in each interval."""

    stepped: dict[str, tuple[int, ...]] = field(default_factory=dict)
    steps: dict[str, list[Any]] = field(default_factory=dict)
    params: list[Any] = field(default_factory=list)
    now: Any = None
    dissipated: list[Any] = field(default_factory=list)


class Shooting:
    """The closed loop's motion from ``q0`` over ``horizon`` [s] in ``nodes - 1`` intervals.

    The unknowns are q and v at the nodes. Each interval is ``substeps`` control steps: at each
    the controller's command is held while the robot's own integrator takes one step, the linearly
    implicit Euler of ``vmc.sim.rollout`` (``integrator="implicit"``, stable for stiff springs),
    or the fourth-order Runge-Kutta (``"rk4"``, for robots that are not stiff). Constraints tie the
    end of every interval to the next node, so the plan is the closed loop of a simulation, from
    ``q0`` and ``v0`` (rest by default), which are parameters of the program when the problem
    declares ``shooting.q0`` and ``shooting.v0``.

    ``steps`` names live Params (glob patterns) that take a new value in every interval, as the
    inputs of a receding-horizon controller: the plan holds one value per interval and ``Result``
    returns them in ``steps``. Declare them as parameters, and each solve starts from the values
    they have now. ``initial`` and ``transition`` swap from the controller in place, as in
    ``Collocation``, except that the model does not carry the force that balances the robot at
    ``q0`` under it: start where it rests. ``scales`` are the typical sizes of q and v.

    A controller with virtual states (a flywheel, a tank) adds them as unknowns at the nodes, and
    the controller advances them as it does in the simulator. They start from the state it is
    compiled with, as after a reset. With ``z0`` (positions, then velocities) the controller is
    already running, as the one an ``MPC`` plans for: ``z0`` is the state its next step reads,
    and a parameter of the program like ``q0`` (``shooting.z0``).
    """

    name = "shooting"
    motion = True

    def __init__(
        self,
        q0: ArrayLike,
        horizon: float,
        nodes: int,
        *,
        v0: ArrayLike | None = None,
        steps: Sequence[str] = (),
        initial: Any = None,
        transition: float = 0.0,
        scales: tuple[float, float] = (1.0, 1.0),
        integrator: str = "implicit",
        substeps: int = 1,
        z0: ArrayLike | None = None,
    ) -> None:
        if integrator not in INTEGRATORS:
            raise ValueError(f"integrator takes one of {INTEGRATORS}, got {integrator!r}")
        if nodes < 2:
            raise ValueError("a shooting needs at least 2 nodes")
        if horizon <= 0.0:
            raise ValueError("the horizon must be positive")
        if transition < 0.0:
            raise ValueError("the transition cannot be negative")
        if substeps < 1:
            raise ValueError("substeps is at least 1")
        if len(scales) != 2 or min(scales) <= 0.0:
            raise ValueError("scales are two positive numbers: the sizes of q and v")
        q0 = np.asarray(q0, dtype=float).ravel()
        v0 = np.zeros(q0.size) if v0 is None else np.asarray(v0, dtype=float).ravel()
        self._q0 = Ref("q0", q0.size, q0, unit="")
        self._v0 = Ref("v0", v0.size, v0, unit="")
        z0 = None if z0 is None else np.asarray(z0, dtype=float).ravel()
        self._z0 = None if z0 is None else Ref("z0", z0.size, z0, unit="")
        self.horizon, self.nodes = float(horizon), int(nodes)
        self.steps, self.transition = tuple(steps), float(transition)
        self.scales, self.integrator, self.substeps = scales, integrator, int(substeps)
        self._hold = None if initial is None else _snapshot(initial)

    def coordinates(self) -> tuple[Any, ...]:
        """The start, as references: Params ``shooting.q0``, ``shooting.v0`` and, with ``z0``,
        ``shooting.z0``."""
        return (self._q0, self._v0) if self._z0 is None else (self._q0, self._v0, self._z0)

    def build(self, builder: Builder) -> None:
        """Add the unknowns, the continuity constraints, the steps and the trajectory."""
        system = builder.system
        space = system.robot.model.space
        nq, nv, n = space.nq, space.nv, self.nodes
        k_int = n - 1
        q_now, v_now = self._q0.param.value, self._v0.param.value
        if q_now.size != nq or v_now.size != nv:
            raise ValueError(
                f"q0 and v0 need {nq} and {nv} entries, got {q_now.size}, {v_now.size}"
            )
        dt = self.horizon / k_int
        t = np.arange(n) * dt
        sq, sv = self.scales
        inf = np.inf
        Q = builder.variables.add("q", n * nq, -inf, inf, np.tile(q_now, n), sq)
        V = builder.variables.add("v", n * nv, -inf, inf, np.tile(v_now, n), sv)
        qs = [Q[k * nq : (k + 1) * nq] for k in range(n)]
        vs = [V[k * nv : (k + 1) * nv] for k in range(n)]

        compiled, dynamics = builder.compiled, builder.dynamics
        stepped, steps = self._add_steps(builder, k_int)
        now = builder.pack(compiled.params, compiled.live)
        law_params = [self._pack(builder, steps, k) for k in range(k_int)]
        p_dyn = builder.pack(dynamics.params, dynamics.live)
        hold = self._check_hold(builder)
        blend = np.array([blend_weight(float(tk), self.transition) if hold else 1.0 for tk in t])
        # a controller that is running goes on, one that is not starts with the plan
        running = self._z0 is not None
        held = self._hold is not None and self._hold[3]
        z_now = None if self._z0 is None else self._z0.param.value
        z_new = start_of(compiled, z_now, "z0")
        z_old = hold[2] if hold else np.zeros(0)
        zn, zo = nodes(builder, "z", z_new, n), nodes(builder, "zh", z_old, n)
        law = blended_law(compiled, hold)

        h = dt / self.substeps
        step = _advance(_rhs(space, dynamics), dynamics, space, self.integrator, h, h)
        us, joins, dissipated = [], [], []
        for k in range(k_int):
            x, taken = ca.vertcat(qs[k], vs[k]), ca.MX(0)
            z, z_prev = zn[k], zo[k]
            for j in range(self.substeps):  # one control step: the command, held, and the robot
                tj = float(t[k] + j * h)
                w = blend_weight(tj, self.transition)
                u, zdot, zdot_old = law(x[:nq], x[nq:], z, z_prev, tj, w, law_params[k])
                if j == 0:
                    us.append(u)
                x = step(x, u, p_dyn, tj)
                # the controller learns the time of a step at the next one: its first step is free
                first = k == 0 and j == 0
                z = advance(z, zdot, 0.0 if first and not running else h)
                z_prev = advance(z_prev, zdot_old, 0.0 if first and not held else h)
                power = compiled.power(x[:nq], x[nq:], z, law_params[k], tj + h)[1]
                taken -= h * power
            dissipated.append(taken)
            joins.append(space.difference(qs[k + 1], x[:nq]))
            joins.append(vs[k + 1] - x[nq:])
            if z_new.size:
                joins.append(zn[k + 1] - z)
            if z_old.size:
                joins.append(zo[k + 1] - z_prev)
        us.append(
            law(qs[-1], vs[-1], zn[-1], zo[-1], float(t[-1]), float(blend[-1]), law_params[-1])[0]
        )
        z_start = z_new if self._z0 is None else builder.value(self._z0.param)
        start = ca.vertcat(
            space.difference(qs[0], builder.value(self._q0.param)),
            vs[0] - builder.value(self._v0.param),
            zn[0] - z_start,
            zo[0] - z_old,
        )
        builder.constrain("start", start, 0.0, 0.0)
        builder.constrain("continuity", ca.vertcat(*joins), 0.0, 0.0)
        accel = [dynamics.forward(qs[k], vs[k], us[k], p_dyn, float(t[k])) for k in range(n)]
        builder.output("u", ca.horzcat(*us))
        builder.output("a", ca.horzcat(*accel))
        builder.output("horizon", ca.MX(self.horizon))  # the fixed horizon, as Result.horizon
        shapes = {"q": (n, nq), "v": (n, nv)}
        builder.trajectory = ShootingTrajectory(
            t=t,
            dt=dt,
            q=qs,
            v=vs,
            a=accel,
            u=us,
            blend=blend,
            shapes={**shapes, **({"z": (n, z_new.size)} if z_new.size else {})},
            evaluate=builder.evaluate,
            z=zn,
            stepped=stepped,
            steps=steps,
            params=law_params,
            now=now,
            dissipated=dissipated,
        )

    def _add_steps(
        self, builder: Builder, k_int: int
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, list[Any]]]:
        """One decision variable per interval for each Param that steps."""
        names: list[str] = []
        for pattern in self.steps:
            found = builder.params.select(patterns=[pattern])
            if not found:
                raise KeyError(
                    f"no Param matches {pattern!r}; the names are {list(builder.params)}"
                )
            names += [name for name in found if name not in names]
        stepped: dict[str, tuple[int, ...]] = {}
        steps: dict[str, list[Any]] = {}
        for name in names:
            param = builder.params[name]
            if name in builder.free:
                raise ValueError(f"{name!r} is free: a Param is free or steps, not both")
            if name not in builder.compiled.live:
                raise ValueError(f"{name!r} steps, so it must be live: give it scope='stage'")
            lower, upper = param_bounds(param)
            size = param.size
            column = builder.variables.add(
                f"step:{name}",
                k_int * size,
                np.tile(lower, k_int),
                np.tile(upper, k_int),
                np.tile(param.value.ravel("F"), k_int),
                param.scale,
            )
            stepped[name] = (k_int, *param.shape)
            steps[name] = [column[k * size : (k + 1) * size] for k in range(k_int)]
        return stepped, steps

    def _pack(self, builder: Builder, steps: dict[str, list[Any]], k: int) -> Any:
        """The live Params of the controller in interval ``k``: its steps, the others as given."""
        compiled = builder.compiled
        parts = [
            steps[name][k] if name in steps else builder.value(compiled.params[name])
            for name in compiled.live
        ]
        return ca.vertcat(*parts) if parts else ca.DM.zeros(0, 1)

    def _check_hold(self, builder: Builder) -> tuple[Any, Any, np.ndarray] | None:
        """The initial controller's law, its (fixed) live Params and its virtual state, or None."""
        if self._hold is None:
            return None
        compiled, values, z, _ = self._hold
        if compiled.system.robot is not builder.system.robot:
            raise ValueError("the initial controller must control the same robot")
        return compiled.law, ca.DM(values), z
