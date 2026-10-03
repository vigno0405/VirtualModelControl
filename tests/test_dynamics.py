"""Robot dynamics from components: lab values, power balance, equilibrium, manifolds."""

from pathlib import Path

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from helpers import Rod  # noqa: F401  (keeps the helpers path import consistent)
from virtualmodelcontrol.core import ParamSet
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parent / "data" / "dynamics.npz")
SIGN = helyx.ENCODER_SIGN["145-290-290"]
rng = np.random.default_rng(11)


def soft_arm(damped=True):
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    if not damped:
        del arm.components["damping"]
    return arm


def test_matches_the_lab_arm_model_at_rest():
    dyn = vmc.compile_dynamics(soft_arm())
    p, zero = dyn.live_values(), np.zeros(9)
    np.testing.assert_allclose(np.array(dyn.mass(zero, p)), DATA["M0"], rtol=1e-12, atol=1e-13)
    for delta, tau, rhs in zip(DATA["delta"], DATA["tau_raw"], DATA["rhs"], strict=True):
        r = np.array(dyn.residual(delta, zero, zero, SIGN * tau, p, 0.0)).ravel()
        np.testing.assert_allclose(-r, rhs, rtol=1e-10, atol=1e-12)


def test_power_balance_of_the_robot():
    dyn = vmc.compile_dynamics(soft_arm())
    p = dyn.live_values()
    for _ in range(5):
        q, v, u = (
            rng.uniform(-0.01, 0.01, 9),
            rng.uniform(-0.05, 0.05, 9),
            rng.normal(size=9) * 0.02,
        )
        a = np.array(dyn.forward(q, v, u, p, 0.0)).ravel()
        h = 1e-7

        def E(s, q=q, v=v, a=a, h=h):
            T, V = dyn.energy(q + s * h * v, v + s * h * a, p, 0.0)
            return float(T) + float(V)

        dEdt = (E(1) - E(-1)) / (2 * h)
        inp, diss, src = (float(x) for x in dyn.power(q, v, u, p, 0.0))
        assert diss <= 0.0
        assert abs(dEdt - (inp + diss + src)) < 1e-6 * max(1.0, abs(inp), abs(diss))


def test_settles_where_the_springs_balance_gravity():
    plant = vmc.sim.ModelPlant(soft_arm())
    plant.advance(8.0)  # slowest mode: K / D ≈ 1.9 per second
    dyn = plant.dynamics
    a = np.array(dyn.forward(plant.q, np.zeros(9), np.zeros(9), plant.p, 0.0)).ravel()
    assert np.abs(plant.v).max() < 1e-6 and np.abs(a).max() < 1e-4
    assert plant.q[2] > 0  # hanging: gravity stretches the first segment


class FreeBody:
    """A planar rigid body: q = (x, y, cos θ, sin θ); centre of mass at ``com``."""

    q_unit = ""

    def __init__(self):
        self.space = vmc.Product(vmc.Euclidean(2), vmc.SO2())
        self.params = ParamSet([vmc.Param("com", [0.0, 0.0, 0.0], unit="m")])
        self.sites = ("com", "rim")

    def frame(self, q, at, p):
        c, s = q[2], q[3]
        R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
        offset = p["com"] if at == "com" else ca.DM([0.2, 0.0, 0.0])
        return R, ca.vertcat(q[0], q[1], 0) + ca.mtimes(R, offset)


def test_free_body_keeps_momentum_on_the_manifold():
    body = vmc.Mechanism("body", model=FreeBody())
    body.add("m", vmc.PointMass(body.point("com"), 1.0))
    body.add("rim", vmc.PointMass(body.point("rim"), 0.5))  # gives the body an inertia
    v0 = np.array([0.3, -0.1, 2.0])
    plant = vmc.sim.ModelPlant(body, v0=v0, max_step=1e-4)
    dyn = plant.dynamics
    M0 = np.array(dyn.mass(plant.q, plant.p))
    plant.advance(1.0)
    assert abs(np.hypot(plant.q[2], plant.q[3]) - 1.0) < 1e-12
    p_lin = lambda M, v: (M @ v)[:2]  # noqa: E731
    M1 = np.array(dyn.mass(plant.q, plant.p))
    np.testing.assert_allclose(p_lin(M1, plant.v), p_lin(M0, v0), atol=1e-3)
