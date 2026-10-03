"""Compile a virtual mechanism system into CasADi functions: the control law and its energetics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .core.params import Binding, ParamSet
from .mechanisms.coordinates.base import Context
from .system import VirtualMechanismSystem

ARGS = ["q", "v", "z", "p", "t"]
OPTS = {"cse": True}  # merges the repeated derivatives of shared coordinates


@dataclass
class Compiled:
    """CasADi functions of a compiled system.

    ``law`` (→ u, ż), ``tau``, ``energy`` (→ V, T), ``power`` (→ port, dissipation, source) and
    ``forces`` take (q, v, z, p, t): configuration, velocity, virtual state z = [positions,
    velocities], live Params p and time t [s]. ``fast`` maps one vector [θ, θ̇, z, p, t] of motor
    angles and rates to [u, ż]; ``fast_energy`` maps it to V + T.
    """

    system: VirtualMechanismSystem
    params: ParamSet
    live: list[str]
    law: ca.Function
    tau: ca.Function
    energy: ca.Function
    power: ca.Function
    forces: ca.Function
    fast: ca.Function
    fast_energy: ca.Function
    component_names: list[str]
    z0: np.ndarray
    n_motors: tuple[int, int]
    n_u: int

    def live_values(self) -> np.ndarray:
        """Current values of the live Params, packed like p."""
        return self.params.vector(self.live)

    def live_slices(self) -> dict[str, slice]:
        """Where each live Param sits in p."""
        out, offset = {}, 0
        for name in self.live:
            size = self.params[name].size
            out[name] = slice(offset, offset + size)
            offset += size
        return out


def compile(system: VirtualMechanismSystem, runtime: Iterable[str] = ()) -> Compiled:
    """Compile ``system`` into CasADi functions.

    ``stage`` Params, and those matching the glob patterns in ``runtime``, stay live inputs; every
    other Param is folded in at its current value (compile again after changing one).
    """
    params = system.params
    live = params.select(patterns=runtime, scopes=["stage"])
    binding = Binding(params, live)
    space, actuation = system.robot.model.space, system.actuation
    states = system.states
    nz = sum(s.dim for s in states)
    q, v, t = ca.SX.sym("q", space.nq), ca.SX.sym("v", space.nv), ca.SX.sym("t")
    z = ca.SX.sym("z", 2 * nz)
    zpos, zvel = z[:nz], z[nz:]
    slices, offset = {}, 0
    for state in states:
        slices[id(state)] = slice(offset, offset + state.dim)
        offset += state.dim
    ctx = Context(q, binding, z=zpos, t=t, states=slices)
    G = space.velocity_map(q)

    tau = ca.SX.zeros(space.nv, 1)
    fz = ca.SX.zeros(nz, 1)
    Mz = ca.SX.zeros(nz, nz)
    V, P_diss, P_src = ca.SX(0), ca.SX(0), ca.SX(0)
    per_component: list[Any] = []
    names: list[str] = []
    for name, comp in system.components:
        y = ctx.value(comp.coord)
        Jq = ca.mtimes(ca.jacobian(y, q), G)
        Jz = ca.jacobian(y, zpos)
        if comp.kind == "inertance":
            if ca.depends_on(y, ca.vertcat(q, t)):
                raise ValueError(f"{name}: a controller's inertances may act on its states only")
            Mz += ca.mtimes([Jz.T, comp.inertance(ctx, y), Jz])
            continue
        yd = ca.mtimes(Jq, v) + ca.mtimes(Jz, zvel) + ca.jacobian(y, t)
        f = comp.force(ctx, y, yd)
        tau_k = ca.mtimes(Jq.T, f)
        tau += tau_k
        fz += ca.mtimes(Jz.T, f)
        if comp.kind == "storage":
            V += comp.energy(ctx, y)
        elif comp.kind == "dissipation":
            P_diss += ca.dot(f, yd)
        else:
            P_src += ca.dot(f, yd)
        per_component += [y, yd, f, tau_k]
        names.append(name)

    z0 = np.concatenate([s.initial for s in states] + [np.zeros(nz)])
    T = 0.5 * ca.dot(zvel, ca.mtimes(Mz, zvel))
    if nz:
        M0 = np.array(ca.Function("M", [z, binding.p], [Mz])(z0, binding.values()))
        if np.linalg.matrix_rank(M0) < nz:
            raise ValueError("every virtual state needs an inertance (the state mass is singular)")
        coriolis = ca.jtimes(ca.mtimes(Mz, zvel), zpos, zvel) - ca.gradient(T, zpos)
        zdot = ca.vertcat(zvel, ca.solve(Mz, fz - coriolis))
    else:
        zdot = ca.SX.zeros(0, 1)
    u = actuation.allocate(tau, q, binding.view(actuation.params))

    args = [q, v, z, binding.p, t]
    law = ca.Function("law", args, [u, zdot], ARGS, ["u", "zdot"], OPTS)
    energy = ca.Function("energy", args, [V, T], ARGS, ["V", "T"], OPTS)

    # Fast path: motor angles and rates in, through the exact inverse of the transmission.
    n_angles, n_rates = actuation.motor_sizes(space)
    theta, theta_dot = ca.SX.sym("theta", n_angles), ca.SX.sym("theta_dot", n_rates)
    pa = binding.view(actuation.params)
    qa = actuation.config_from_motors(theta, pa)
    va = actuation.velocity_from_motors(qa, theta_dot, pa)
    x = ca.vertcat(theta, theta_dot, z, binding.p, t)
    ua, zdot_a = law(qa, va, z, binding.p, t)
    Va, Ta = energy(qa, va, z, binding.p, t)

    return Compiled(
        system=system,
        params=params,
        live=binding.live,
        law=law,
        tau=ca.Function("tau", args, [tau], ARGS, ["tau"], OPTS),
        energy=energy,
        power=ca.Function(
            "power",
            args,
            [ca.dot(tau, v), P_diss, P_src],
            ARGS,
            ["port", "dissipation", "source"],
            OPTS,
        ),
        forces=ca.Function("forces", args, per_component, OPTS),
        fast=ca.Function("fast", [x], [ca.vertcat(ua, zdot_a)], ["x"], ["out"], OPTS),
        fast_energy=ca.Function("fast_energy", [x], [Va + Ta], ["x"], ["E"], OPTS),
        component_names=names,
        z0=z0,
        n_motors=(n_angles, n_rates),
        n_u=int(u.numel()),
    )
