"""Signed distances to a box, a capsule and a cylinder: known values, and the distance property."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import component_function

P = vmc.Joint(slice(0, 3), unit="m")
rng = np.random.default_rng(11)


def distance(coord):
    """The coordinate's value at a point, as a function."""
    fn = component_function(vmc.ContactSpring(coord, 1.0), 3)
    return lambda point: float(fn(point, np.zeros(3))[2])


def box(center=(0.0, 0.0, 0.0), half=(1.0, 2.0, 3.0)):
    return distance(vmc.BoxDistance(P, center, half))


def capsule(a=(0.0, 0.0, 0.0), b=(0.0, 0.0, 1.0), radius=0.2):
    return distance(vmc.CapsuleDistance(P, a, b, radius))


def cylinder(center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=0.5, half_height=1.0):
    return distance(vmc.CylinderDistance(P, center, axis, radius, half_height))


def test_the_distance_to_a_box_from_a_face_an_edge_a_corner_and_inside():
    d = box()
    assert d([2.0, 0.0, 0.0]) == pytest.approx(1.0)  # in front of a face
    assert d([-2.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert d([2.0, 3.0, 0.0]) == pytest.approx(np.sqrt(2.0))  # beside an edge
    assert d([2.0, 3.0, 4.0]) == pytest.approx(np.sqrt(3.0))  # beside a corner
    assert d([0.0, 0.0, 0.0]) == pytest.approx(-1.0)  # inside: the nearest face is 1 m away
    assert d([0.5, 0.0, 0.0]) == pytest.approx(-0.5)
    assert d([0.0, 0.0, 2.0]) == pytest.approx(-1.0)  # nearer the top face than the others
    assert d([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-8)  # on a face
    shifted = box(center=(1.0, -1.0, 0.5))
    assert shifted([3.0, -1.0, 0.5]) == pytest.approx(1.0)
    assert shifted([1.0, -1.0, 0.5]) == pytest.approx(-1.0)


def test_the_distance_to_a_capsule_from_its_side_its_ends_and_inside():
    d = capsule()
    assert d([0.5, 0.0, 0.5]) == pytest.approx(0.3)  # beside the middle
    assert d([0.0, 0.0, 1.5]) == pytest.approx(0.3)  # past the end b: the distance to b, less r
    assert d([0.0, 0.0, -0.5]) == pytest.approx(0.3)
    assert d([0.3, 0.0, 2.0]) == pytest.approx(np.sqrt(0.3**2 + 1.0) - 0.2)  # oblique to the end
    assert d([0.1, 0.0, 0.5]) == pytest.approx(-0.1)  # inside
    assert d([0.0, 0.0, 0.5]) == pytest.approx(-0.2, abs=1e-8)  # on the axis
    tilted = capsule(a=(1.0, 0.0, 0.0), b=(0.0, 1.0, 0.0), radius=0.1)
    assert tilted([1.0, 1.0, 0.0]) == pytest.approx(np.sqrt(0.5) - 0.1)


def test_a_capsule_whose_ends_meet_is_a_sphere_and_has_finite_forces():
    sphere = distance(vmc.SphereDistance(P, [0.1, 0.2, 0.3], 0.4))
    point = capsule(a=(0.1, 0.2, 0.3), b=(0.1, 0.2, 0.3), radius=0.4)
    for x in rng.uniform(-1.0, 1.0, (10, 3)):
        assert point(x) == pytest.approx(sphere(x), abs=1e-8)
    # pressed into by a spring, the push is finite and the sphere's
    spring = vmc.ContactSpring(vmc.CapsuleDistance(P, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 0.4), 1e3)
    reference = vmc.ContactSpring(vmc.SphereDistance(P, [0.1, 0.2, 0.3], 0.4), 1e3)
    inside = [0.2, 0.2, 0.4]
    tau = np.array(component_function(spring, 3)(inside, np.zeros(3))[0]).ravel()
    expected = np.array(component_function(reference, 3)(inside, np.zeros(3))[0]).ravel()
    assert np.isfinite(tau).all() and np.abs(tau).max() > 1.0
    np.testing.assert_allclose(tau, expected, atol=1e-6)


def test_the_distance_to_a_cylinder_from_its_side_its_caps_its_rim_and_inside():
    d = cylinder()
    assert d([1.0, 0.0, 0.0]) == pytest.approx(0.5)  # beside the wall
    assert d([0.0, -1.5, 0.3]) == pytest.approx(1.0)
    assert d([0.0, 0.0, 2.0]) == pytest.approx(1.0)  # above a cap
    assert d([0.0, 0.0, -2.0]) == pytest.approx(1.0)  # below the other
    assert d([0.0, 0.0, -0.9]) == pytest.approx(-0.1)
    assert d([1.0, 0.0, 2.0]) == pytest.approx(np.sqrt(0.5**2 + 1.0))  # beside the rim
    assert d([0.2, 0.0, 0.3]) == pytest.approx(-0.3)  # inside, nearest the wall
    assert d([0.0, 0.0, 0.9]) == pytest.approx(-0.1)  # inside, nearest a cap
    assert d([0.5, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-8)


def test_a_cylinder_can_lie_along_any_axis_of_any_length():
    along_x = cylinder(center=(1.0, 0.0, 0.0), axis=(7.0, 0.0, 0.0))  # the length is not the size
    assert along_x([1.0, 1.0, 0.0]) == pytest.approx(0.5)  # beside the wall
    assert along_x([3.0, 0.0, 0.0]) == pytest.approx(1.0)  # past a cap
    assert along_x([1.0, 0.0, 0.0]) == pytest.approx(-0.5)
    slanted = cylinder(axis=(1.0, 1.0, 0.0), radius=0.5, half_height=1.0)
    assert slanted([0.0, 0.0, 1.5]) == pytest.approx(1.0)  # a point on the third axis: the wall


@pytest.mark.parametrize(
    "shape",
    [
        lambda: box(center=(0.1, -0.2, 0.3)),
        lambda: capsule(a=(0.0, 0.1, 0.0), b=(0.3, 0.1, 0.8), radius=0.2),
        lambda: cylinder(center=(0.1, 0.0, -0.1), axis=(0.3, -0.2, 1.0), radius=0.4),
    ],
)
def test_outside_the_distance_changes_at_the_speed_of_the_point(shape):
    d, h, checked = shape(), 1e-6, 0
    for x in rng.uniform(-3.0, 3.0, (60, 3)):
        if d(x) < 0.05:  # skip the neighbourhood of the surface and the inside
            continue
        gradient = np.array([(d(x + h * e) - d(x - h * e)) / (2 * h) for e in np.eye(3)])
        assert np.linalg.norm(gradient) == pytest.approx(1.0, abs=1e-5)
        checked += 1
    assert checked > 30


def test_the_distances_are_params_that_a_controller_can_change():
    coord = vmc.BoxDistance(P, [0, 0, 0], [1, 2, 3])
    assert set(coord.params()) == {"center", "half_sizes"}
    assert coord.half_sizes.bounds == (0.0, np.inf)
    assert set(vmc.CapsuleDistance(P, [0, 0, 0], [0, 0, 1], 0.2).params()) == {"a", "b", "radius"}
    cyl = vmc.CylinderDistance(P, [0, 0, 0], [0, 0, 1], 0.5, 1.0)
    assert set(cyl.params()) == {"center", "axis", "radius", "half_height"}
    assert all(p.scope == "episode" for p in cyl.params().values())
