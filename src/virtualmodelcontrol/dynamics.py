"""Robot dynamics assembled from the robot's own components, in the canonical residual form.

M(q) a + h(q, v) − B(q) u − Σ (J G)ᵀ f = 0: inertances give M, the Lagrangian of
T = ½ vᵀ M v gives h, springs, gravity and dampers give f, the actuation gives B.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import casadi as ca

from .core.params import Binding, ParamSet
from .mechanisms.coordinates.base import Context
from .mechanisms.mechanism import Mechanism
from .models.actuation import Direct

OPTS = {"cse": True}


@dataclass
class Dynamics:
    """CasADi functions of a robot's dynamics; p holds its live Params.

    ``forward`` (q, v, u, p, t) → a; ``residual`` (q, v, a, u, p, t) → r; ``mass`` (q, p) → M;
    ``energy`` (q, v, p, t) → (T, V); ``power`` (q, v, u, p, t) → (input, dissipation, source);
    ``motors`` (q, v, p) → (θ, θ̇); ``step`` (q, v, u, p, t, h) → (q⁺, v⁺), a linearly implicit
    Euler step, stable for stiff springs and dampers.
    """

    robot: Mechanism
    params: ParamSet
    live: list[str]
    forward: ca.Function
    residual: ca.Function
    mass: ca.Function
    energy: ca.Function
    power: ca.Function
    motors: ca.Function
    step: ca.Function
    n_u: int

    def live_values(self) -> Any:
        """Current values of the live Params, packed like p."""
        return self.params.vector(self.live)


def compile_dynamics(robot: Mechanism, runtime: Iterable[str] = ()) -> Dynamics:
    """Compile the dynamics of ``robot`` from its components.

    Valid for spaces whose velocity coordinates commute (Euclidean, SO2 and their products).
    ``stage`` Params and those matching ``runtime`` stay live; the rest are folded in.
    """
    if robot.model is None:
        raise ValueError(f"robot {robot.name!r} needs a kinematic model")
    actuation = robot.actuation if robot.actuation is not None else Direct()
    params = ParamSet()
    params.merge(robot.params, robot.name)
    if robot.actuation is None:
        params.merge(actuation.params, robot.name)
    binding = Binding(params, params.select(patterns=runtime, scopes=["stage"]))
    space = robot.model.space
    pa = binding.view(actuation.params)
    n_u = actuation.motor_sizes(space)[1]
    q, v, a = ca.SX.sym("q", space.nq), ca.SX.sym("v", space.nv), ca.SX.sym("a", space.nv)
    u, t, h = ca.SX.sym("u", n_u), ca.SX.sym("t"), ca.SX.sym("h")
    ctx = Context(q, binding, t=t)
    G = space.velocity_map(q)

    M = ca.SX.zeros(space.nv, space.nv)
    f_gen = ca.SX.zeros(space.nv, 1)
    V, P_diss, P_src = ca.SX(0), ca.SX(0), ca.SX(0)
    for name, comp in robot.components.items():
        y = ctx.value(comp.coord)
        J = ca.mtimes(ca.jacobian(y, q), G)
        if comp.kind == "inertance":
            M += ca.mtimes([J.T, comp.inertance(ctx, y), J])
            continue
        yd = ca.mtimes(J, v) + ca.jacobian(y, t)
        f = comp.force(ctx, y, yd)
        f_gen += ca.mtimes(J.T, f)
        if comp.kind == "storage":
            V += comp.energy(ctx, y)
        elif comp.kind == "dissipation":
            P_diss += ca.dot(f, yd)
        elif comp.kind == "source":
            P_src += ca.dot(f, yd)
        else:
            raise ValueError(f"{name}: unknown component kind {comp.kind!r}")

    T = 0.5 * ca.dot(v, ca.mtimes(M, v))
    h_vec = ca.mtimes(ca.jacobian(ca.mtimes(M, v), q), ca.mtimes(G, v)) - ca.mtimes(
        G.T, ca.gradient(T, q)
    )
    tau_u = actuation.generalized_force(u, q, pa)
    rhs = f_gen + tau_u - h_vec
    acc = ca.solve(M, rhs)

    # Linearly implicit Euler: (M − h ∂f/∂v − h² ∂f/∂q G) Δv = h (rhs + h ∂f/∂q G v). Only the
    # component and input forces f are linearized (stiffness and damping); the velocity-squared
    # terms h stay explicit, which keeps the step first order and cheap.
    f_lin = f_gen + tau_u
    dq = ca.mtimes(ca.jacobian(f_lin, q), G)
    dv = ca.jacobian(f_lin, v)
    delta_v = ca.solve(M - h * dv - h**2 * dq, h * (rhs + h * ca.mtimes(dq, v)))
    v_next = v + delta_v
    q_next = space.integrate(q, h * v_next)

    p = binding.p
    theta = actuation.motor_angles(q, pa)
    theta_dot = actuation.motor_rates(q, v, pa)
    return Dynamics(
        robot=robot,
        params=params,
        live=binding.live,
        forward=ca.Function(
            "forward", [q, v, u, p, t], [acc], ["q", "v", "u", "p", "t"], ["a"], OPTS
        ),
        residual=ca.Function(
            "residual",
            [q, v, a, u, p, t],
            [ca.mtimes(M, a) - rhs],
            ["q", "v", "a", "u", "p", "t"],
            ["r"],
            OPTS,
        ),
        mass=ca.Function("mass", [q, p], [M], ["q", "p"], ["M"], OPTS),
        energy=ca.Function("energy", [q, v, p, t], [T, V], ["q", "v", "p", "t"], ["T", "V"], OPTS),
        power=ca.Function(
            "power",
            [q, v, u, p, t],
            [ca.dot(tau_u, v), P_diss, P_src],
            ["q", "v", "u", "p", "t"],
            ["input", "dissipation", "source"],
            OPTS,
        ),
        motors=ca.Function(
            "motors", [q, v, p], [theta, theta_dot], ["q", "v", "p"], ["theta", "theta_dot"], OPTS
        ),
        step=ca.Function(
            "step",
            [q, v, u, p, t, h],
            [q_next, v_next],
            ["q", "v", "u", "p", "t", "h"],
            ["q_next", "v_next"],
            OPTS,
        ),
        n_u=n_u,
    )
