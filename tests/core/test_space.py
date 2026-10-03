import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.core import SO2, Euclidean, Product

rng = np.random.default_rng(0)


def random_config(space):
    q = space.neutral()
    return space.integrate(q, rng.uniform(-2.0, 2.0, space.nv))


@pytest.mark.parametrize("space", [Euclidean(3), SO2(), Product(Euclidean(2), SO2())])
def test_difference_inverts_integrate(space):
    for _ in range(20):
        q = random_config(space)
        v = rng.uniform(-1.0, 1.0, space.nv)
        np.testing.assert_allclose(space.difference(space.integrate(q, v), q), v, atol=1e-12)


@pytest.mark.parametrize("space", [Euclidean(3), SO2(), Product(Euclidean(2), SO2())])
def test_velocity_map_matches_finite_differences(space):
    q, v, h = random_config(space), rng.normal(size=space.nv), 1e-6
    fd = (space.integrate(q, h * v) - space.integrate(q, -h * v)) / (2 * h)
    np.testing.assert_allclose(space.velocity_map(q) @ v, fd, atol=1e-8)


def test_so2_stays_on_the_unit_circle():
    space, q = SO2(), SO2().neutral()
    for _ in range(1000):
        q = space.integrate(q, rng.uniform(-3, 3, 1))
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_planar_product_dimensions():
    space = Product(Euclidean(2), SO2())
    assert (space.nq, space.nv) == (4, 3)
    assert space.velocity_map(space.neutral()).shape == (4, 3)


def test_symbolic_inputs_give_symbolic_outputs():
    space = Product(Euclidean(2), SO2())
    q, v = ca.SX.sym("q", 4), ca.SX.sym("v", 3)
    q1 = space.integrate(q, v)
    assert isinstance(q1, ca.SX) and q1.shape == (4, 1)
    f = ca.Function("f", [q, v], [q1, space.velocity_map(q)])
    q0, v0 = random_config(space), rng.normal(size=3)
    out, g = f(q0, v0)
    np.testing.assert_allclose(np.array(out).ravel(), space.integrate(q0, v0), atol=1e-14)
    np.testing.assert_allclose(np.array(g), space.velocity_map(q0), atol=1e-14)
