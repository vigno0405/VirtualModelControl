"""Physical properties of every component: f = −∂V/∂y, V ≥ 0, dampers dissipate, saturation."""

import casadi as ca
import numpy as np
import pytest
from scipy.integrate import quad as scipy_quad

import virtualmodelcontrol as vmc
from helpers import Rod, component_function, component_params, evaluate, make_context
from virtualmodelcontrol.core import Param
from virtualmodelcontrol.mechanisms import (
    ConstrainedLinearSpring,
    ConstrainedTanhDamper,
    GaussianSpring,
    GravityCompensation,
    Inertance,
    Joint,
    LimitSpring,
    LinearDamper,
    LinearSpring,
    Mechanism,
    PointMass,
    PolynomialSpring,
    Projection,
    SigmoidSpring,
    TanhDamper,
    TanhSpring,
)

rng = np.random.default_rng(3)
Y3 = Joint(slice(0, 3), unit="m")
K_MATRIX = np.array([[30.0, 5.0, 0.0], [5.0, 20.0, -3.0], [0.0, -3.0, 10.0]])

STORAGE = {
    "linear_scalar": LinearSpring(Y3, 30.0),
    "linear_axes": LinearSpring(Y3, [10.0, 20.0, 30.0]),
    "linear_matrix": LinearSpring(Y3, K_MATRIX),
    "tanh_scalar": TanhSpring(Y3, 65.0, 0.5),
    "tanh_zero_axis": TanhSpring(Y3, [15.0, 0.0, 35.0], 1.5),
    "gaussian": GaussianSpring(Y3, 50.0, 0.06),
    "sigmoid_axes": SigmoidSpring(Y3, 5.0, 50.0, 0.05, 100.0),
    "sigmoid_norm": SigmoidSpring(Y3, 5.0, 50.0, 0.05, 100.0, element_wise=False),
    "polynomial_axes": PolynomialSpring(Y3, 40.0, 2, 0.05),
    "polynomial_norm": PolynomialSpring(Y3, 40.0, 2, 0.05, element_wise=False),
    "limit": LimitSpring(Y3, 2.0, -0.05, 0.05),
}


@pytest.mark.parametrize("name", STORAGE)
def test_storage_force_is_minus_energy_gradient(name):
    fn = component_function(STORAGE[name], 3)
    h = 1e-6
    for y in rng.uniform(-0.2, 0.2, (20, 3)):
        f = np.array(fn(y, np.zeros(3))[1]).ravel()
        V = lambda x: float(fn(x, np.zeros(3))[4])  # noqa: E731
        grad = np.array([(V(y + h * e) - V(y - h * e)) / (2 * h) for e in np.eye(3)])
        np.testing.assert_allclose(f, -grad, rtol=1e-6, atol=1e-8)
        assert V(y) >= 0.0


def test_tanh_spring_with_zero_stiffness_is_inert():
    fn = component_function(TanhSpring(Y3, 0.0, 1.0), 3)
    tau, _, _, _, V = fn([0.1, -0.2, 0.3], np.zeros(3))
    assert float(V) == 0.0 and np.all(np.array(tau) == 0.0)


def test_tanh_on_a_projection_never_exceeds_max_force():
    # The force along any direction stays below F_max, also for oblique directions.
    for _ in range(20):
        n = rng.normal(size=3)
        fn = component_function(TanhSpring(Projection(Y3, n), 500.0, 1.5), 3)
        forces = evaluate(fn, rng.uniform(-10.0, 10.0, (20, 3)))
        assert np.all(np.linalg.norm(forces, axis=1) <= 1.5 * (1 + 1e-12))


def test_sigmoid_energy_matches_numerical_integral():
    spring = SigmoidSpring(Y3, 5.0, 50.0, 0.05, 100.0, element_wise=False)
    fn = component_function(spring, 3)

    def k(r):
        return 5.0 + 45.0 / (1.0 + np.exp(-100.0 * (r - 0.05)))

    for y in rng.uniform(-0.2, 0.2, (10, 3)):
        r = np.linalg.norm(y)
        exact = scipy_quad(lambda rho: k(rho) * rho, 0.0, r, epsabs=1e-14)[0]
        assert abs(float(fn(y, np.zeros(3))[4]) - exact) < 1e-9 * max(exact, 1e-3)


DAMPERS = {
    "linear_scalar": LinearDamper(Y3, 1.5),
    "linear_axes": LinearDamper(Y3, [1.0, 2.0, 3.0]),
    "linear_matrix": LinearDamper(Y3, K_MATRIX / 10.0),
    "tanh_scalar": TanhDamper(Y3, 0.33, 0.5),
    "tanh_axes": TanhDamper(Y3, [0.3, 1.0, 2.0], 0.5),
}


@pytest.mark.parametrize("name", DAMPERS)
def test_dampers_dissipate(name):
    fn = component_function(DAMPERS[name], 3)
    for v in rng.uniform(-2.0, 2.0, (50, 3)):
        _, f, _, yd, _ = fn(np.zeros(3), v)
        assert float(ca.dot(f, yd)) <= 0.0


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        (PointMass(Y3, 0.04), 0.04 * np.eye(3)),
        (Inertance(Y3, [1.0, 2.0, 3.0]), np.diag([1.0, 2.0, 3.0])),
        (Inertance(Joint(0), 0.1), [[0.1]]),
    ],
)
def test_inertances(component, expected):
    ctx = make_context(component_params(component), 3)
    M = component.inertance(ctx, ctx.value(component.coord))
    np.testing.assert_array_equal(np.array(ca.DM(M)), expected)


def test_gravity_compensation_cancels_gravity_on_each_mass():
    robot = Mechanism("arm", model=Rod(0.5))
    gravity = robot.add_param(Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("m1", PointMass(robot.point(s=0.5), 0.1))
    robot.add("m2", PointMass(robot.point(s=1.0), 0.2))
    comp = GravityCompensation(robot)
    assert comp.gravity is gravity  # shared with the robot
    f = np.array(ca.DM(comp.force(make_context(robot.params, 3), None, None))).ravel()
    np.testing.assert_allclose(f, [0, 0, 0.981, 0, 0, 1.962])


def test_gravity_compensation_needs_masses_and_gravity():
    robot = Mechanism("arm", model=Rod())
    with pytest.raises(ValueError, match="no PointMass"):
        GravityCompensation(robot)
    robot.add("m1", PointMass(robot.point(s=1.0), 0.1))
    with pytest.raises(ValueError, match="gravity"):
        GravityCompensation(robot)
    assert GravityCompensation(robot, gravity=[0, 0, -9.81]).gravity.unit == "m/s^2"


def test_constrained_elements_name_their_direction_normal():
    bar = Mechanism("bar", model=vmc.models.JointSpace(3, unit="m"))
    ctrl = Mechanism("ctrl")
    ctrl.add("line", ConstrainedLinearSpring(Y3, 100.0, normal=[0.0, 0.0, 1.0]))
    ctrl.add("drag", ConstrainedTanhDamper(Y3, 1.0, 0.5, normal=[1.0, 0.0, 0.0]))
    names = ["line.stiffness", "line.normal", "drag.damping", "drag.max_force", "drag.normal"]
    assert list(ctrl.params) == names
    assert ctrl.params["line.normal"].scope == "episode"
    system = vmc.VirtualMechanismSystem(bar, ctrl)
    assert "ctrl.line.normal" in vmc.compile(system, runtime=["ctrl.*.normal"]).live
