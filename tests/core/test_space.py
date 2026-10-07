import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from virtualmodelcontrol.core import SO2, Euclidean, Product, Quaternion
from virtualmodelcontrol.math import quat_rot

rng = np.random.default_rng(0)


def random_config(space):
    q = space.neutral()
    return space.integrate(q, rng.uniform(-2.0, 2.0, space.nv))


SPACES = [
    Euclidean(3),
    SO2(),
    Product(Euclidean(2), SO2()),
    Quaternion(),
    Product(Euclidean(3), Quaternion(), Euclidean(1)),
]


@pytest.mark.parametrize("space", SPACES)
def test_difference_inverts_integrate(space):
    for _ in range(20):
        q = random_config(space)
        v = rng.uniform(-1.0, 1.0, space.nv)
        np.testing.assert_allclose(space.difference(space.integrate(q, v), q), v, atol=1e-12)


@pytest.mark.parametrize("space", SPACES)
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


def test_a_quaternion_turns_as_a_body_does_and_stays_a_unit_after_many_turns():
    space, q, R = Quaternion(), Quaternion().neutral(), Rotation.identity()
    turned = 0.0
    for _ in range(3000):
        v = rng.uniform(-2.0, 2.0, 3)
        q, R, turned = (
            space.integrate(q, v),
            R * Rotation.from_rotvec(v),
            turned + np.linalg.norm(v),
        )
    assert turned > 100 * np.pi  # many full turns, far past what a rotation vector holds
    np.testing.assert_allclose(quat_rot(q), R.as_matrix(), atol=1e-11)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-13
    G = space.velocity_map(q)
    assert abs(q @ G).max() < 1e-13  # a velocity moves q along the sphere
    np.testing.assert_allclose(G.T @ G, np.eye(3) / 4, atol=1e-13)


def test_the_difference_of_two_rotations_is_the_short_way_and_sees_q_and_minus_q_alike():
    space = Quaternion()
    q = space.integrate(space.neutral(), [0.3, -0.2, 0.5])
    for v in ([3.0, 0.0, 0.0], [0.0, -2.9, 0.1]):  # just below a half turn
        np.testing.assert_allclose(space.difference(space.integrate(q, v), q), v, atol=1e-12)
    far = space.integrate(q, [4.0, 0.0, 0.0])  # 4 rad is 2 pi - 4 the other way
    np.testing.assert_allclose(space.difference(far, q), [4.0 - 2 * np.pi, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(space.difference(-far, q), space.difference(far, q), atol=1e-12)
    np.testing.assert_allclose(space.difference(q, q), 0.0, atol=1e-15)


def test_the_coadjoint_term_is_zero_for_commuting_velocities_and_the_cross_product_for_a_body():
    v, mu = rng.normal(size=6), rng.normal(size=6)
    np.testing.assert_array_equal(Euclidean(6).coadjoint(v, mu), 0.0)
    np.testing.assert_array_equal(SO2().coadjoint(v[:1], mu[:1]), 0.0)
    np.testing.assert_allclose(Quaternion().coadjoint(v[:3], mu[:3]), np.cross(mu[:3], v[:3]))
    mixed = Product(Euclidean(2), Quaternion(), SO2())
    out = mixed.coadjoint(v, mu[:6])
    np.testing.assert_allclose(out[:2], 0.0)
    np.testing.assert_allclose(out[2:5], np.cross(mu[2:5], v[2:5]))
    assert out[5] == 0.0
    sv, smu = ca.SX.sym("v", 3), ca.SX.sym("mu", 3)
    symbolic = ca.Function("f", [sv, smu], [Quaternion().coadjoint(sv, smu)])
    np.testing.assert_allclose(np.array(symbolic(v[:3], mu[:3])).ravel(), np.cross(mu[:3], v[:3]))


def test_integrating_pulls_a_quaternion_that_drifted_back_to_a_unit():
    space = Quaternion()
    drifted = 1.001 * space.integrate(space.neutral(), [0.3, -0.2, 0.5])
    assert abs(np.linalg.norm(space.integrate(drifted, [0.01, 0.0, 0.0])) - 1.0) < 1e-15


def test_the_equations_of_the_manifold_vanish_on_a_configuration_and_not_off_it():
    assert Euclidean(4).on_manifold(ca.DM.ones(4)).shape == (0, 1)  # a flat space has none
    quarter = np.array([np.cos(0.3), np.sin(0.3)])
    assert float(SO2().on_manifold(quarter)[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(SO2().on_manifold(2.0 * quarter)[0]) == pytest.approx(3.0)  # 4 - 1
    unit = Quaternion().integrate(Quaternion().neutral(), [0.3, -0.2, 0.5])
    assert float(Quaternion().on_manifold(unit)[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(Quaternion().on_manifold(3.0 * unit)[0]) == pytest.approx(8.0)  # 9 - 1
    mixed = Product(Euclidean(2), Quaternion(), SO2())
    q = np.concatenate([[0.7, -0.1], unit, quarter])
    np.testing.assert_allclose(np.array(mixed.on_manifold(q)).ravel(), [0.0, 0.0], atol=1e-15)
    q[2:6] *= 2.0
    np.testing.assert_allclose(np.array(mixed.on_manifold(q)).ravel(), [3.0, 0.0], atol=1e-14)
    assert mixed.on_manifold(q).shape == (2, 1)  # a row for the quaternion and one for the angle
