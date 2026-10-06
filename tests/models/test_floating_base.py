"""A free rigid body from point masses, on the free joint: momentum and energy are kept."""

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import SerialChain

MASSES = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 1.5}
POINTS = {"a": [0.2, 0.0, 0.0], "b": [0.0, 0.1, 0.0], "c": [0.0, 0.0, 0.3], "d": [-0.1, -0.1, -0.1]}


def body():
    chain = SerialChain(
        ["free"],
        axes=[None],
        points=[[0.0, 0.0, 0.0]],
        sites={n: (1, p) for n, p in POINTS.items()},
    )
    robot = vmc.Mechanism("body", model=chain)
    for name, mass in MASSES.items():
        robot.add(f"m_{name}", vmc.PointMass(robot.point(name), mass))
    return robot


def momenta(kin, q, v):
    """Total linear momentum, angular momentum about the origin, and kinetic energy."""
    P, L, T = np.zeros(3), np.zeros(3), 0.0
    for name, mass in MASSES.items():
        x, xdot = kin.position(q, name), kin.jacobian(q, name) @ v
        P, L, T = P + mass * xdot, L + mass * np.cross(x, xdot), T + 0.5 * mass * xdot @ xdot
    return P, L, T


def test_a_tumbling_body_keeps_its_momenta_and_its_energy():
    robot = body()
    kin = vmc.Kinematics(robot)
    v0 = np.array([0.1, 0.2, -0.1, 0.8, -1.2, 0.5])
    plant = vmc.sim.ModelPlant(robot, v0=v0, max_step=1e-4)
    start = momenta(kin, plant.q, plant.v)
    plant.advance(1.0)
    end = momenta(kin, plant.q, plant.v)
    assert np.linalg.norm(plant.q[3:]) > 0.5  # it did turn, and never near a rotation of 2 pi
    # the integrator is first order: these are its errors, a missing term would be 1 or 65 %
    np.testing.assert_allclose(end[0], start[0], atol=3e-4)
    np.testing.assert_allclose(end[1], start[1], atol=1e-4)
    assert abs(end[2] / start[2] - 1) < 1e-4
