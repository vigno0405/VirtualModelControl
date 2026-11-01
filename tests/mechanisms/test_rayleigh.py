"""Dampers are Rayleigh dampers: f = −∂R/∂ẏ with ẏ·∂R/∂ẏ ≥ 0, so f·ẏ ≤ 0, and the power they take
is 2R for a linear one. The library builds the force f and the power f·ẏ, never R itself; these
tests pin that they are the same thing: the Jacobian ∂f/∂ẏ of a damper is symmetric (f is the
gradient of something) and, for every shipped damper but the smooth diode, negative semidefinite
(that something is convex; the smooth diode's is slightly bent where its gate opens, and it
dissipates all the same)."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import component_params, make_context
from underactuated_lab import build
from virtualmodelcontrol.mechanisms import (
    ContactDamper,
    ContactFriction,
    DiodeDamper,
    Joint,
    LinearDamper,
    PlaneDistance,
    TanhDamper,
)

Y3 = Joint(slice(0, 3), unit="m")
MATRIX = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, -0.3], [0.0, -0.3, 1.0]])
FLOOR = PlaneDistance(Y3, [0.0, 0.0, 1.0])

DAMPERS = {
    "linear scalar": (LinearDamper(Y3, 1.5), 3),
    "linear axes": (LinearDamper(Y3, [1.0, 2.0, 3.0]), 3),
    "linear matrix": (LinearDamper(Y3, MATRIX), 3),
    "tanh": (TanhDamper(Y3, 0.33, 0.5), 3),
    "tanh axes": (TanhDamper(Y3, [0.3, 1.0, 2.0], 0.5), 3),
    "diode": (DiodeDamper(Y3, 3.0, sign=1.0), 3),
    "diode smooth": (DiodeDamper(Y3, [1.0, 2.0, 3.0], smoothing=0.05), 3),
    "contact": (ContactDamper(FLOOR, 50.0, 0.01), 1),
    "contact friction": (ContactFriction(FLOOR, 1000.0, 0.4, 1e-3, 0.01), 3),
}


def rate_jacobian(component, dimension):
    """(q, ẏ) → (f, ∂f/∂ẏ) of a damper, with ẏ an input of its own."""
    ctx = make_context(component_params(component), 3)
    rate = ca.SX.sym("rate", dimension)
    f = component.force(ctx, ctx.value(component.coord), rate)
    return ca.Function("damper", [ctx.q, rate], [f, ca.jacobian(f, rate)])


BENT = {"diode smooth"}  # dissipates (f·ẏ ≤ 0) with a dissipation function that is not convex


@pytest.mark.parametrize("name", DAMPERS)
def test_a_damper_is_the_gradient_of_a_dissipation_function(name):
    component, dimension = DAMPERS[name]
    function = rate_jacobian(component, dimension)
    rng = np.random.default_rng(4)
    for _ in range(30):
        q = np.array([*rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.01, 0.002)])  # often in contact
        rate = rng.uniform(-2.0, 2.0, dimension)
        f, H = (np.array(x) for x in function(q, rate))
        H = np.atleast_2d(H)
        scale = max(abs(H).max(), 1e-12)
        assert abs(H - H.T).max() <= 1e-9 * scale  # f = −∂R/∂ẏ for some R
        if name not in BENT:
            assert np.linalg.eigvalsh(0.5 * (H + H.T)).max() <= 1e-9 * scale  # R convex
        assert float(f.ravel() @ rate) <= 1e-12  # so f·ẏ ≤ 0


def test_the_damping_of_a_compiled_robot_is_a_rayleigh_matrix_and_the_power_is_twice_r():
    """J'f = −∂R/∂v with R = ½ vᵀ C v: C is symmetric positive semidefinite, and the power the
    dampers take, f·ẏ summed, is −vᵀ C v = −2R."""
    arm = build("three")
    dynamics = vmc.compile_dynamics(arm)
    live, zero = dynamics.live_values(), np.zeros(3)
    q = np.array([0.5, 0.3, 0.2])

    def generalized(v):  # the velocity-dependent part of the generalized force
        return -np.asarray(dynamics.residual(q, v, zero, np.zeros(2), live, 0.0)).ravel()

    C = -np.column_stack([(generalized(1e-6 * e) - generalized(-1e-6 * e)) / 2e-6
                          for e in np.eye(3)])  # fmt: skip
    np.testing.assert_allclose(C, C.T, atol=1e-8)
    assert np.linalg.eigvalsh(0.5 * (C + C.T)).min() >= -1e-9
    assert np.linalg.eigvalsh(C).max() > 1e-3  # and there is damping to test
    rng = np.random.default_rng(5)
    for v in rng.normal(size=(4, 3)):
        power = float(dynamics.power(q, v, np.zeros(2), live, 0.0)[1])
        assert power == pytest.approx(-v @ C @ v, rel=1e-6, abs=1e-12)
        assert -power == pytest.approx(2 * (0.5 * v @ C @ v), rel=1e-6, abs=1e-12)
