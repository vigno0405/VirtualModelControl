"""A floating base on a unit quaternion: it turns any number of times, by Euler's equations."""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

import virtualmodelcontrol as vmc
from virtualmodelcontrol.math import quat_rot
from virtualmodelcontrol.models import SerialChain, evaluate_frame, from_dict
from virtualmodelcontrol.testing import check_model

# six masses on the axes of a body, which put its centre at the origin and its principal axes on
# the frame's: inertias 0.16, 0.22 and 0.30 kg m^2 about x, y and z (y is the intermediate axis)
AXES = {"x": (1.0, 0.3), "y": (1.5, 0.2), "z": (2.0, 0.1)}  # mass [kg], distance [m]
INERTIA = np.diag([0.16, 0.22, 0.30])


def body(kind="floating"):
    sites = {}
    for axis, (_, d) in AXES.items():
        for sign in (1, -1):
            point = np.zeros(3)
            point["xyz".index(axis)] = sign * d
            sites[f"{axis}{'+' if sign > 0 else '-'}"] = (1, point)
    chain = SerialChain([kind], axes=[None], points=[[0.0, 0.0, 0.0]], sites=sites)
    robot = vmc.Mechanism("body", model=chain)
    for name in sites:
        robot.add(f"m_{name}", vmc.PointMass(robot.point(name), AXES[name[0]][0]))
    return robot


def test_the_floating_joint_is_the_free_joint_with_a_quaternion_for_the_rotation_vector():
    rotvec, shift = np.array([0.3, -0.5, 0.9]), np.array([0.1, 0.2, 0.3])
    quat = Rotation.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]]
    sites = {"a": (1, [0.2, 0.1, -0.3])}
    floating = SerialChain(["floating"], axes=[None], points=[[0.1, 0, 0]], sites=sites)
    free = SerialChain(["free"], axes=[None], points=[[0.1, 0, 0]], sites=sites)
    assert (floating.space.nq, floating.space.nv) == (7, 6)
    R, p = evaluate_frame(floating, np.concatenate([shift, quat]), "a")
    R0, p0 = evaluate_frame(free, np.concatenate([shift, rotvec]), "a")
    np.testing.assert_allclose(R, R0, atol=1e-14)
    np.testing.assert_allclose(p, p0, atol=1e-14)
    np.testing.assert_allclose(quat_rot(quat), Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-14)


def test_the_jacobian_is_taken_in_the_rates_the_body_turns_with():
    chain = body().model
    kin = vmc.Kinematics(chain)
    q = np.concatenate(
        [[0.1, -0.2, 0.3], Rotation.from_rotvec([0.4, 0.2, -0.7]).as_quat()[[3, 0, 1, 2]]]
    )
    v = np.array([0.3, 0.1, -0.2, 0.5, -0.4, 0.2])
    h = 1e-6
    moved = [kin.position(chain.space.integrate(q, s * h * v), "y+") for s in (1, -1)]
    np.testing.assert_allclose(
        kin.jacobian(q, "y+") @ v, (moved[0] - moved[1]) / (2 * h), atol=1e-8
    )
    assert kin.jacobian(q, "y+").shape == (3, 6)


def euler(t, w):
    """The rotation of a free body about its centre: I w' = (I w) x w."""
    return np.linalg.solve(INERTIA, np.cross(INERTIA @ w, w))


def test_a_body_that_tumbles_for_many_turns_follows_eulers_equations():
    robot = body()
    kin = vmc.Kinematics(robot)
    spin = np.array([0.05, 8.0, 0.1])  # about the intermediate axis: it flips, again and again
    plant = vmc.sim.ModelPlant(robot, v0=np.concatenate([np.zeros(3), spin]), max_step=1e-4)
    times = np.linspace(0.0, 2.0, 21)
    exact = solve_ivp(euler, (0.0, 2.0), spin, t_eval=times, rtol=1e-11, atol=1e-13)
    rates = []
    for t in times:
        plant.advance(t - plant.t)
        rates.append(plant.v[3:].copy())
    np.testing.assert_allclose(rates, exact.y.T, atol=5e-3 * np.abs(spin).max())
    speed = np.linalg.norm(exact.y, axis=0)
    turned = np.sum(0.5 * (speed[1:] + speed[:-1]) * np.diff(times))  # the angle it turned
    assert turned > 2 * 2 * np.pi  # two full turns and more, past where a rotation vector ends
    assert np.ptp(exact.y[0]) > 1.0  # and it flipped about the other axes on the way
    assert abs(np.linalg.norm(plant.q[3:]) - 1.0) < 1e-12
    # the angular momentum, in the frame it is kept in, and the energy
    R = kin.rotation(plant.q, "x+")
    L = R @ (INERTIA @ plant.v[3:])
    L0 = INERTIA @ spin
    np.testing.assert_allclose(L, L0, atol=2e-3 * np.linalg.norm(L0))
    energy = lambda w: 0.5 * w @ INERTIA @ w  # noqa: E731
    assert abs(energy(plant.v[3:]) / energy(spin) - 1.0) < 2e-3


def test_a_floating_base_with_an_arm_keeps_its_momenta_and_its_energy():
    chain = SerialChain(
        ["floating", "revolute"],
        axes=[None, [0.0, 1.0, 0.0]],
        points=[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
        sites={
            "base": (1, [0.0, 0.0, 0.0]),
            "end": (2, [0.15, 0.0, -0.4]),
            "ear": (1, [0.0, 0.2, 0.1]),
        },
    )
    robot = vmc.Mechanism("flyer", model=chain)
    mass = {"base": 3.0, "end": 1.0, "ear": 0.5}
    for name, m in mass.items():
        robot.add(f"m_{name}", vmc.PointMass(robot.point(name), m))
    kin = vmc.Kinematics(robot)
    v0 = np.array([0.2, -0.1, 0.3, 0.5, 2.0, -1.0, 1.5])  # the body moves and spins, the arm swings
    plant = vmc.sim.ModelPlant(robot, v0=v0, max_step=3e-5)  # a first-order step: its error

    def totals(q, v):
        P, L, T = np.zeros(3), np.zeros(3), 0.0
        for name, m in mass.items():
            x, xdot = kin.position(q, name), kin.jacobian(q, name) @ v
            P, L, T = P + m * xdot, L + m * np.cross(x, xdot), T + 0.5 * m * xdot @ xdot
        return P, L, T

    start = totals(plant.q, plant.v)
    plant.advance(1.0)
    end = totals(plant.q, plant.v)
    assert np.linalg.norm(Rotation.from_quat(plant.q[[4, 5, 6, 3]]).as_rotvec()) > 0.1
    np.testing.assert_allclose(end[0], start[0], atol=1.5e-3)  # the errors are 5e-4 at most
    np.testing.assert_allclose(end[1], start[1], atol=1.5e-3)
    assert abs(end[2] / start[2] - 1.0) < 2e-3


def test_the_floating_chain_passes_the_model_contract_and_survives_a_dict_round_trip():
    chain = SerialChain(
        ["floating", "prismatic"],
        axes=[None, [1.0, 0.0, 0.0]],
        points=[[0, 0, 0], [0, 0, 0]],
        sites={"tip": (2, [0.0, 0.0, 0.1])},
    )
    check_model(chain, samples=3, energy=False)
    copy = from_dict(json.loads(json.dumps(chain.to_dict())))
    assert copy.joints == ["floating", "prismatic"] and repr(copy.space) == repr(chain.space)
    q = chain.space.integrate(chain.space.neutral(), np.linspace(-0.4, 0.5, chain.space.nv))
    for a, b in zip(evaluate_frame(copy, q, "tip"), evaluate_frame(chain, q, "tip"), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-15)


def test_a_chain_without_a_floating_joint_keeps_its_flat_space():
    flat = SerialChain(
        ["free", "revolute"],
        axes=[None, [0, 0, 1.0]],
        points=[[0, 0, 0]] * 2,
        sites={"tip": (2, [0, 0, 0])},
    )
    assert repr(flat.space) == "Euclidean(7)"
    assert repr(body().model.space) == "Product(Euclidean(3), Quaternion())"


@pytest.mark.parametrize("n", [1, 2])
def test_neighbouring_flat_joints_after_a_floating_one_share_a_block(n):
    chain = SerialChain(
        ["floating", *["revolute"] * n],
        axes=[None, *[[0, 0, 1.0]] * n],
        points=[[0, 0, 0]] * (n + 1),
        sites={"tip": (n + 1, [0, 0, 0])},
    )
    assert repr(chain.space) == f"Product(Euclidean(3), Quaternion(), Euclidean({n}))"
    assert (chain.space.nq, chain.space.nv) == (7 + n, 6 + n)
