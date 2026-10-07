"""Shared by the tests of the numpy and PyTorch code of the models."""

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import backends
from virtualmodelcontrol.robots import helyx, turtle

rng = np.random.default_rng(0)
TOLERANCE = 1e-12


def same(got, want, tolerance=TOLERANCE):
    """The array ``got`` equals CasADi's ``want`` to ``tolerance`` of the largest entry."""
    want = np.array(want, dtype=float).reshape(np.shape(got))
    scale = max(float(np.abs(want).max(initial=0.0)), 1.0)
    assert np.abs(np.asarray(got, dtype=float) - want).max(initial=0.0) <= tolerance * scale


def check(function, args, backend="numpy"):
    """Every output of the translated ``function`` at ``args`` agrees with CasADi's."""
    run = backends.translate(function, backend)
    want = function(*args)
    want = want if isinstance(want, (tuple, list)) else (want,)
    got = run(*args)
    got = got if function.n_out() > 1 else (got,)
    assert len(got) == len(want) == function.n_out()
    for g, w in zip(got, want, strict=True):
        same(np.asarray(g), w)
    return run


def operations():
    """One output for each operation the translator knows, over arguments in (0.2, 0.9)."""
    x = ca.SX.sym("x", 4)
    a, b, c, d = (x[i] for i in range(4))
    table = {
        "exp": ca.exp(a), "log": ca.log(b), "sqrt": ca.sqrt(c), "sin": ca.sin(a),
        "cos": ca.cos(b), "tan": ca.tan(c), "asin": ca.asin(a), "acos": ca.acos(b),
        "atan": ca.atan(c), "sinh": ca.sinh(a), "cosh": ca.cosh(b), "tanh": ca.tanh(c),
        "asinh": ca.asinh(a), "acosh": ca.acosh(b + 1), "atanh": ca.atanh(c / 2),
        "log1p": ca.log1p(a), "expm1": ca.expm1(b), "floor": ca.floor(5 * a),
        "ceil": ca.ceil(5 * b), "fabs": ca.fabs(c - 0.5), "sign": ca.sign(d - 0.5),
        "erf": ca.erf(a), "pow": a**b, "constpow": b**3.5, "sq": c**2, "inv": 1 / d,
        "div": a / b, "sub": c - d, "neg": -a, "twice": 2 * b, "fmod": ca.fmod(a + 3, b),
        "fmin": ca.fmin(a, b), "fmax": ca.fmax(c, d), "atan2": ca.atan2(a, b - 0.5),
        "copysign": ca.copysign(c, d - 0.5), "hypot": ca.hypot(a, b),
        "lt": a < b, "le": c <= d, "eq": a == b, "ne": b != c,
        "and": ca.logic_and(a < b, c < d), "or": ca.logic_or(a > b, c > d),
        "not": ca.logic_not(a < b), "ifelse": ca.if_else(a > b, c, d),
    }  # fmt: skip
    return ca.Function("operations", [x], [ca.vertcat(*table.values())]), list(table)


def arm():
    return helyx.add_dynamics(helyx.arm("145-290-290"))


def flywheel():
    cranks = turtle.robot()
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, 0.24))
    ctrl.add("drive", vmc.SpeedRegulator(phi, 0.3, 6.0, 0.5))
    for i, side in enumerate((1.0, -1.0)):
        e = cranks.joint(i) - (phi - (np.pi if i else 0.0))
        spring = vmc.PhaseSpring(vmc.Stack(e, phi), 1.0, depth=0.5, side=side)
        ctrl.add(f"spring{i}", spring)
        ctrl.add(f"damper{i}", vmc.LinearDamper(e, 0.06))
    return vmc.VirtualMechanismSystem(cranks, ctrl)
