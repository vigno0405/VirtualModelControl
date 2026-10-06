"""Contact between two points of a robot: equal and opposite forces, and friction between them."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc

M1, M2, K, WIDTH = 1.0, 2.0, 1e4, 0.1  # [kg], [kg], [N/m], [m]


def pair(friction=0.0, damping=0.0):
    """Two point masses, each free in space, that touch when they are closer than WIDTH. The
    friction acts along the sphere of radius WIDTH around the first, on the second's relative
    position."""
    robot = vmc.Mechanism("pair", model=vmc.models.JointSpace(6, unit="m"))
    a, b = robot.joint(slice(0, 3)), robot.joint(slice(3, 6))
    robot.add("ma", vmc.PointMass(a, M1))
    robot.add("mb", vmc.PointMass(b, M2))
    stiffness = vmc.Param("k", K, unit="N/m", scope="stage")
    robot.add("touch", vmc.ContactSpring(vmc.Norm(b - a) - WIDTH, stiffness))
    if damping:
        robot.add("cushion", vmc.ContactDamper(vmc.Norm(b - a) - WIDTH, damping))
    sphere = vmc.SphereDistance(b - a, [0.0, 0.0, 0.0], WIDTH)
    robot.add("rub", vmc.ContactFriction(sphere, stiffness, friction, 1e-4))
    return robot


def collide(friction, seconds=0.3):
    """The first mass hits the second, which is a little to the side, at 1 m/s."""
    plant = vmc.sim.ModelPlant(
        pair(friction),
        q0=[0.0, 0.0, 0.0, 0.3, 0.04, 0.0],
        v0=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_step=1e-4,
    )
    states = []
    for _ in range(int(seconds / 1e-3)):
        plant.advance(1e-3)
        states.append(np.concatenate([plant.q, plant.v]))
    return np.array(states)


def momentum(states):
    return M1 * states[:, 6:9] + M2 * states[:, 9:12]


def energy(states):
    """Kinetic energy plus the contact spring's."""
    v, depth = (
        states[:, 6:],
        np.maximum(0.0, WIDTH - np.linalg.norm(states[:, 3:6] - states[:, :3], axis=1)),
    )
    kinetic = 0.5 * M1 * np.sum(v[:, :3] ** 2, axis=1) + 0.5 * M2 * np.sum(v[:, 3:] ** 2, axis=1)
    return kinetic + 0.5 * K * depth**2


def test_the_two_points_are_pushed_with_equal_and_opposite_forces():
    plant = vmc.sim.ModelPlant(pair(friction=0.4))
    rng = np.random.default_rng(5)
    for _ in range(10):
        a = rng.normal(0, 0.01, 3)
        direction = rng.normal(0, 1, 3)
        plant.q = np.concatenate([a, a + 0.098 * direction / np.linalg.norm(direction)])
        plant.v = rng.normal(0, 1, 6)
        for part in plant.elements().values():
            if len(part["torque"]) == 6:
                np.testing.assert_allclose(part["torque"][:3], -part["torque"][3:], atol=1e-9)
        assert plant.elements()["touch"]["torque"][3:] @ direction > 0  # it pushes b away


def test_the_friction_between_two_points_is_tangential_and_against_their_sliding():
    mu, depth = 0.5, 2e-3
    plant = vmc.sim.ModelPlant(pair(friction=mu))
    plant.q = np.array([0.0, 0.0, 0.0, WIDTH - depth, 0.0, 0.0])  # b is 2 mm into a's sphere
    plant.v = np.array([0.0, 0.0, 0.0, 0.3, -0.4, 0.2])
    rub = plant.elements()["rub"]["torque"]
    on_b = rub[3:]
    slide = plant.v[3:] - np.array([1.0, 0.0, 0.0]) * plant.v[3]  # b's velocity along the sphere
    assert abs(on_b[0]) < 1e-9  # none along the line between them
    np.testing.assert_allclose(on_b, -mu * K * depth * slide / np.linalg.norm(slide), rtol=1e-4)
    np.testing.assert_allclose(rub[:3], -on_b, atol=1e-12)  # the first feels the opposite


def test_a_collision_with_friction_keeps_the_momentum_and_loses_energy():
    free, rough = collide(0.0), collide(0.4)
    for states in (free, rough):
        np.testing.assert_allclose(momentum(states) - momentum(states)[0], 0.0, atol=1e-6)
    start = energy(free)[0]
    assert energy(free)[-1] > 0.97 * start  # an elastic touch keeps its energy, to the integrator's
    assert energy(rough)[-1] < 0.9 * start  # friction took a good part of it
    assert np.all(np.diff(energy(rough)) < 1e-4)  # and gave none back


def soft_object():
    """Two tips on a line with a soft object between them: its two faces are points with a mass
    each, joined by a spring of rest length 10 cm, and each tip touches a face."""
    robot = vmc.Mechanism("hold", model=vmc.models.JointSpace(4, unit="m"))
    a, left, right, b = (robot.joint(i) for i in range(4))
    for name, coord in zip(("a", "left", "right", "b"), (a, left, right, b), strict=True):
        robot.add(f"m_{name}", vmc.Inertance(coord, 0.5))
        robot.add(f"d_{name}", vmc.LinearDamper(coord, 5.0))
    robot.add("body", vmc.LinearSpring((right - left) - 0.10, K_BODY))
    robot.add("touch_a", vmc.ContactSpring(left - a, K_FACE))
    robot.add("touch_b", vmc.ContactSpring(b - right, K_FACE))
    pull = vmc.Mechanism("pull")
    pull.add("spring", vmc.LinearSpring(b - a, K_PULL))
    return robot, vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, pull)))


K_BODY, K_FACE, K_PULL = 500.0, 2e3, 50.0  # [N/m]


def test_a_soft_object_between_two_tips_is_a_series_of_springs():
    robot, controller = soft_object()
    plant = vmc.sim.ModelPlant(robot, q0=[0.0, 0.05, 0.15, 0.2], max_step=1e-3)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=4.0, record="robot")
    rows = log.arrays()
    stiffness = 1 / (1 / K_BODY + 2 / K_FACE)  # the object and the two faces in series
    gap = stiffness * 0.10 / (stiffness + K_PULL)  # where the series spring balances the pull
    assert rows["q"][-1, 3] - rows["q"][-1, 0] == pytest.approx(gap, rel=1e-3)
    force = K_PULL * gap
    for contact in ("touch_a", "touch_b", "body"):  # the same force through all three
        assert np.abs(rows[f"robot/{contact}/force"][-1][0]) == pytest.approx(force, rel=1e-3)
    assert rows["q"][-1, 2] - rows["q"][-1, 1] == pytest.approx(0.10 - force / K_BODY, rel=1e-3)
