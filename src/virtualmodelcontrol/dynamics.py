"""Robot dynamics assembled from the robot's own components, in the canonical residual form.

M(q) a + h(q, v) − B(q) u − Σ (J G)ᵀ f = 0: inertances give M, the Lagrangian of
T = ½ vᵀ M v gives h, springs, gravity and dampers give f, the actuation gives B.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import casadi as ca

from .core.backends import export
from .core.params import Binding, ParamSet
from .mechanisms.coordinates.base import Context
from .mechanisms.mechanism import Mechanism
from .models.actuation import Direct

OPTS = {"cse": True}


@dataclass
class Dynamics:
    """CasADi functions of a robot's dynamics; p holds its live Params.

    ``forward`` (q, v, u, p, t) → a; ``residual`` (q, v, a, u, p, t) → r; ``mass`` (q, p) → M;
    ``energy`` (q, v, p, t) → (T, V); ``power`` (q, v, u, p, t) → (input, dissipation, source)
    (both ``None`` for equations of motion without an energy: see ``needs_energy``);
    ``motors`` (q, v, p) → (θ, θ̇); ``step`` (q, v, u, p, t, h) → (q⁺, v⁺), a linearly implicit
    Euler step, stable for stiff springs and dampers; ``elements`` (q, v, p, t) → the coordinate,
    its rate, the force and the generalized force of each component that is not a mass, in the
    order of ``element_names``.
    """

    robot: Mechanism
    params: ParamSet
    live: list[str]
    forward: ca.Function
    residual: ca.Function
    mass: ca.Function
    energy: ca.Function | None
    power: ca.Function | None
    motors: ca.Function
    step: ca.Function
    elements: ca.Function
    element_names: list[str]
    n_u: int

    def live_values(self) -> Any:
        """Current values of the live Params, packed like p."""
        return self.params.vector(self.live)

    def export(self, backend: str) -> Any:
        """The functions as Python callables of ``backend`` (``"numpy"`` or ``"torch"``), named
        as here, each with its ``source``, which needs no CasADi. The functions that this robot
        does not have (without an energy) are ``None``."""
        names = ("forward", "residual", "mass", "energy", "power", "motors", "step", "elements")
        return export({name: getattr(self, name) for name in names}, backend)


def needs_energy(dynamics: Dynamics, what: str) -> tuple[ca.Function, ca.Function]:
    """The ``energy`` and ``power`` functions of ``dynamics``, or a ``ValueError`` that says what
    needed them: equations of motion given as a function (``models.Equations``) have them only
    when they come with an energy."""
    if dynamics.energy is None or dynamics.power is None:
        raise ValueError(
            f"{what} needs the energy of the robot, and its equations of motion have none: "
            "give models.Equations an energy"
        )
    return dynamics.energy, dynamics.power


def compile_dynamics(
    robot: Mechanism, runtime: Iterable[str] = (), actuation: Any = None
) -> Dynamics:
    """Compile the dynamics of ``robot`` from its components.

    The space gives the velocities' structure: Euclidean, SO2, Quaternion and their products.
    ``stage`` Params and those matching ``runtime`` stay live; the rest are folded in. A robot
    without an actuation gets ``actuation`` (a new ``Direct()`` by default), which a system
    passes so that its Params are the ones the dynamics read.
    """
    if robot.model is None:
        raise ValueError(f"robot {robot.name!r} needs a kinematic model")
    if robot.actuation is not None:
        actuation = robot.actuation
    elif actuation is None:
        actuation = Direct()
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
    equations = getattr(robot.model, "equations", None)

    M = ca.SX.zeros(space.nv, space.nv)
    f_gen = ca.SX.zeros(space.nv, 1)
    V, P_diss, P_src = ca.SX(0), ca.SX(0), ca.SX(0)
    per_component: list[Any] = []
    names: list[str] = []
    for name, comp in robot.components.items():
        y = ctx.value(comp.coord)
        J = ca.mtimes(ca.jacobian(y, q), G)
        if comp.kind == "inertance":
            if equations is not None:
                raise ValueError(
                    f"{name}: a robot with equations of motion has no inertances, "
                    "its masses are in the equations"
                )
            M += ca.mtimes([J.T, comp.inertance(ctx, y), J])
            continue
        yd = ca.mtimes(J, v) + ca.jacobian(y, t)
        f = comp.force(ctx, y, yd)
        f_gen += ca.mtimes(J.T, f)
        per_component += [y, yd, f, ca.mtimes(J.T, f)]
        names.append(name)
        if comp.kind == "storage":
            V += comp.energy(ctx, y)
        elif comp.kind == "dissipation":
            P_diss += ca.dot(f, yd)
        elif comp.kind == "source":
            P_src += ca.dot(f, yd)
        else:
            raise ValueError(f"{name}: unknown component kind {comp.kind!r}")

    tau_u = actuation.generalized_force(u, q, pa)
    if equations is None:
        T = 0.5 * ca.dot(v, ca.mtimes(M, v))
        h_vec = ca.mtimes(ca.jacobian(ca.mtimes(M, v), q), ca.mtimes(G, v)) - ca.mtimes(
            G.T, ca.gradient(T, q)
        )
        h_vec -= space.coadjoint(
            v, ca.mtimes(M, v)
        )  # velocities that do not commute (a spinning body)
        rhs = f_gen + tau_u - h_vec
        acc = ca.solve(M, rhs)
        r = ca.mtimes(M, a) - rhs

        # Linearly implicit Euler: (M − h ∂f/∂v − h² ∂f/∂q G) Δv = h (rhs + h ∂f/∂q G v). Only
        # the component and input forces f are linearized (stiffness and damping); the
        # velocity-squared terms h stay explicit, which keeps the step first order and cheap.
        f_lin = f_gen + tau_u
        dq = ca.mtimes(ca.jacobian(f_lin, q), G)
        dv = ca.jacobian(f_lin, v)
        delta_v = ca.solve(M - h * dv - h**2 * dq, h * (rhs + h * ca.mtimes(dq, v)))
        has_energy = True
    else:  # the equations of motion are the user's: M from r, the whole acceleration linearized
        p_model = binding.view(robot.model.params)
        r = equations.residual(q, v, a, tau_u, f_gen, p_model)
        M = ca.jacobian(r, a)
        if ca.depends_on(M, a):
            raise ValueError("the residual of the equations of motion must be affine in a")
        acc = ca.solve(M, -ca.substitute(r, a, ca.SX.zeros(space.nv)))
        dq = ca.mtimes(ca.jacobian(acc, q), G)
        dv = ca.jacobian(acc, v)
        delta_v = ca.solve(
            ca.SX.eye(space.nv) - h * dv - h**2 * dq, h * (acc + h * ca.mtimes(dq, v))
        )
        has_energy = equations.energy is not None
        if has_energy:
            T, V_model = equations.energy(q, v, p_model)
            V += V_model
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
            [r],
            ["q", "v", "a", "u", "p", "t"],
            ["r"],
            OPTS,
        ),
        mass=ca.Function("mass", [q, p], [M], ["q", "p"], ["M"], OPTS),
        energy=ca.Function("energy", [q, v, p, t], [T, V], ["q", "v", "p", "t"], ["T", "V"], OPTS)
        if has_energy
        else None,
        power=ca.Function(
            "power",
            [q, v, u, p, t],
            [ca.dot(tau_u, v), P_diss, P_src],
            ["q", "v", "u", "p", "t"],
            ["input", "dissipation", "source"],
            OPTS,
        )
        if has_energy
        else None,
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
        elements=ca.Function(
            "elements", [q, v, p, t], per_component, ["q", "v", "p", "t"], [], OPTS
        ),
        element_names=names,
        n_u=n_u,
    )
