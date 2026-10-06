"""Contact friction: a block slides to rest by Coulomb's law, on a table and on a slope."""

from functools import partial

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import SerialChain

G, MASS, K = 9.81, 1.0, 1e4


def block(surface=None, friction=0.4, speed=1e-4, smoothing=0.0):
    """A point mass that slides in three directions, over a surface (the plane z = 0 by default):
    gravity, a contact spring and damper, and the friction, with the spring's stiffness."""
    chain = SerialChain(
        ["prismatic"] * 3,
        axes=np.eye(3).tolist(),
        points=[[0, 0, 0]] * 3,
        sites={"mass": (3, [0, 0, 0])},
    )
    robot = vmc.Mechanism("block", model=chain)
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -G], unit="m/s^2"))
    robot.add("m", vmc.PointMass(robot.point("mass"), MASS))
    robot.add("gravity", vmc.Gravity(robot))
    if surface is None:
        surface = vmc.PlaneDistance(robot.point("mass"), [0.0, 0.0, 1.0], [0, 0, 0])
    else:
        surface = surface(robot.point("mass"))
    k = vmc.Param("k", K, unit="N/m", scope="stage")
    robot.add("floor", vmc.ContactSpring(surface, k, smoothing))
    robot.add("cushion", vmc.ContactDamper(surface, 150.0, smoothing))  # near critical: no bounce
    robot.add("rub", vmc.ContactFriction(surface, k, friction, speed, smoothing))
    return robot


def tilt_normal(angle):
    return [np.sin(angle), 0.0, np.cos(angle)]


def tilted(angle):
    """The plane through the origin that falls towards +x by ``angle``, and the way down it."""
    normal = tilt_normal(angle)
    down = np.array([np.cos(angle), 0.0, -np.sin(angle)])
    return lambda point: vmc.PlaneDistance(point, normal, [0, 0, 0]), down


def test_a_block_pushed_along_a_table_stops_where_coulombs_law_says():
    v0, mu = 2.0, 0.4
    rest = [0.0, 0.0, -MASS * G / K]  # it starts at rest on the contact spring
    plant = vmc.sim.ModelPlant(block(friction=mu), q0=rest, v0=[v0, 0.0, 0.0], max_step=1e-4)
    plant.advance(0.3)
    assert plant.v[0] == pytest.approx(v0 - mu * G * 0.3, rel=2e-3)  # a steady slowing
    plant.advance(2.0)
    assert plant.q[0] == pytest.approx(v0**2 / (2 * mu * G), rel=1e-2)
    assert abs(plant.v[0]) < 1e-3  # it has stopped
    assert plant.q[2] == pytest.approx(-MASS * G / K, rel=1e-3)  # resting on its contact spring


def test_a_block_on_a_slope_accelerates_by_the_slope_and_the_friction():
    angle, mu = np.radians(30.0), 0.2
    surface, down = tilted(angle)
    rest = -MASS * G * np.cos(angle) / K * np.array(tilt_normal(angle))  # on its contact spring
    plant = vmc.sim.ModelPlant(block(surface, mu), q0=rest, v0=0.05 * down, max_step=1e-4)
    plant.advance(0.5)
    expected = G * (np.sin(angle) - mu * np.cos(angle))
    assert plant.v @ down == pytest.approx(0.05 + expected * 0.5, rel=2e-3)


def test_a_block_sticks_where_the_slope_is_gentler_than_the_friction():
    surface, _ = tilted(np.radians(10.0))  # tan 10 degrees is 0.18, below the friction of 0.4
    plant = vmc.sim.ModelPlant(block(surface, 0.4), max_step=1e-4)
    plant.advance(1.0)
    assert np.linalg.norm(plant.v) < 5e-3  # it creeps, far below what a frictionless slope gives


def test_the_friction_force_is_in_the_surface_against_the_motion_and_below_mu_times_the_load():
    mu, depth = 0.5, 2e-3  # [m] into the table
    plant = vmc.sim.ModelPlant(block(friction=mu))
    rng = np.random.default_rng(3)
    for _ in range(20):
        plant.q = np.array([0.1, -0.2, -depth])
        plant.v = rng.normal(0, 1, 3)
        rub = plant.elements()["rub"]["force"]
        assert abs(rub[2]) < 1e-12  # none along the normal
        assert np.linalg.norm(rub) <= mu * K * depth * (1 + 1e-12)
        assert rub @ plant.v < 0  # against the motion
    plant.q, plant.v = np.array([0.0, 0.0, 0.01]), np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(plant.elements()["rub"]["force"], 0.0, atol=1e-15)  # in the air


def test_friction_takes_the_normal_of_each_kind_of_surface():
    mu, v = 0.5, np.array([0.3, -0.4, 0.2])
    side = {"axis": [0, 0, 1.0], "radius": 0.1, "half_height": 0.5}
    cases = {  # a point 2 mm inside the surface, and the direction out of it
        "plane": (partial(vmc.PlaneDistance, normal=[0, 0, 2.0], origin=[0, 0, 0]),
                  [0.3, 0.2, -2e-3], [0, 0, 1]),
        "sphere": (partial(vmc.SphereDistance, center=[0.1, 0, 0], radius=0.2),
                   [0.298, 0, 0], [1, 0, 0]),
        "box": (partial(vmc.BoxDistance, center=[0, 0, 0], half_sizes=[0.1, 0.2, 0.3]),
                [0.098, 0, 0], [1, 0, 0]),
        "cylinder side": (partial(vmc.CylinderDistance, center=[0, 0, 0], **side),
                          [0, 0.098, 0], [0, 1, 0]),
        "cylinder cap": (partial(vmc.CylinderDistance, center=[0, 0, 0], **side),
                         [0, 0, 0.498], [0, 0, 1]),
        "capsule": (partial(vmc.CapsuleDistance, a=[0, 0, 0], b=[0.4, 0, 0], radius=0.05),
                    [0.2, 0.048, 0], [0, 1, 0]),
    }  # fmt: skip
    for name, (surface, at, out) in cases.items():
        plant = vmc.sim.ModelPlant(block(surface, mu, speed=1e-6))
        plant.q, plant.v = np.array(at, dtype=float), v
        n = np.array(out, dtype=float)
        slide = v - n * (n @ v)
        want = -mu * K * 2e-3 * slide / np.linalg.norm(slide)
        np.testing.assert_allclose(plant.elements()["rub"]["force"], want, rtol=1e-4, err_msg=name)


def test_the_friction_has_the_springs_normal_force_whatever_its_smoothing():
    plant = vmc.sim.ModelPlant(block(friction=0.3, smoothing=1e-3, speed=1e-6))
    for depth in (-5e-4, 0.0, 1e-3):  # in the soft edge, at the surface, and a depth
        plant.q, plant.v = np.array([0.0, 0.0, -depth]), np.array([1.0, 0.0, 0.0])
        spring = plant.elements()["floor"]["force"][0]  # the table's push
        rub = plant.elements()["rub"]["force"][0]
        assert rub == pytest.approx(-0.3 * spring, rel=1e-6)


def test_a_dissipation_shows_up_as_the_friction_work_in_the_recorded_run():
    robot = block(friction=0.4)
    nothing = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, vmc.Mechanism("c"))))
    plant = vmc.sim.ModelPlant(robot, v0=[2.0, 0.0, 0.0], max_step=1e-4)
    log = vmc.sim.run(plant, nothing, vmc.sim.SimClock(1e-3), T=1.0, record="robot")
    rows = log.arrays()
    force, rate = rows["robot/rub/force"], rows["robot/rub/ydot"]
    work = -np.sum(np.einsum("ij,ij->i", force, rate)) * 1e-3  # the energy it took
    assert work == pytest.approx(0.5 * MASS * 2.0**2, rel=2e-2)  # all the block's energy
    assert rows["robot/floor/force"][-1][0] == pytest.approx(MASS * G, rel=1e-2)
