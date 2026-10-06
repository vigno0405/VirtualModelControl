"""Shared test helpers: a tiny model and numeric evaluation of components."""

import casadi as ca
import numpy as np

from virtualmodelcontrol.core import Binding, Euclidean, Param, ParamSet
from virtualmodelcontrol.mechanisms import Context, walk


class Rod:
    """Straight rod of length L along z, translated by q: frame(q, s) = (I, q + [0, 0, s L])."""

    q_unit = "m"

    def __init__(self, length=1.0):
        self.space = Euclidean(3)
        self.params = ParamSet([Param("L", length, unit="m", scope="design")])

    def frame(self, q, at, p):
        s = 1.0 if at == "tip" else at
        return ca.DM.eye(3), q[0:3] + ca.vertcat(0, 0, s * p["L"])


def component_params(component):
    ps = ParamSet()
    for name, param in component.params().items():
        ps.add(param, f"c.{name}", rename=True)
    for root in component.reads():
        for coord in walk(root):
            for name, param in coord.params().items():
                ps.add(param, f"c.{name}", rename=True)
    return ps


def make_context(params, nq):
    """Context over symbols q (nq) with every Param of ``params`` frozen."""
    return Context(ca.SX.sym("q", nq), Binding(params))


def component_function(component, nq):
    """Function (q, v) -> (Jᵀf, f, y, ẏ, V) with every Param frozen; V is 0 if not storage."""
    q, v = ca.SX.sym("q", nq), ca.SX.sym("v", nq)
    ctx = Context(q, Binding(component_params(component)))
    y = ctx.value(component.coord)
    J = ca.jacobian(y, q)
    yd = ca.mtimes(J, v)
    f = component.force(ctx, y, yd)
    V = component.energy(ctx, y) if component.kind == "storage" else ca.SX(0)
    return ca.Function("component", [q, v], [ca.mtimes(J.T, f), f, y, yd, V])


def evaluate(fn, Q, V=None, output=0):
    """Evaluate ``fn`` row by row over samples Q (and V); returns an (N, ...) array."""
    Q = np.atleast_2d(Q)
    V = np.zeros_like(Q) if V is None else np.atleast_2d(V)
    return np.array([np.array(fn(q, v)[output]).ravel() for q, v in zip(Q, V, strict=True)])
