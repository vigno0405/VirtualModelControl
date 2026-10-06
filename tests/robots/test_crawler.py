"""The crawler: a simple floating body on the ground, driven by the flywheel controller's cranks.

Its constants are placeholders, not the lab's turtle: the tests pin what the model does."""

from functools import cache
from typing import NamedTuple

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import controller_of, flywheel
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.robots import turtle

W = turtle.CRAWLER["mass"] * 9.81
DT = 1 / turtle.CONTROL_RATE
OMEGA = 6.0  # [rad/s], the flywheel's commanded speed: a stride in a second
GAINS = dict(K=1.0, C=0.06, Jv=0.24, bv=0.3, delta=np.pi, ramp=0.5)


def pose(left=0.0, right=-np.pi, height=0.04):
    """The crawler's q: at x = y = 0, level, facing +x, with the two cranks at these angles."""
    return np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0, left, right])


def yaw(q):
    """The heading [rad] from the quaternion of the body."""
    w, x, y, z = q[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Run(NamedTuple):
    log: vmc.sim.RunLog
    rows: dict[str, np.ndarray]


@cache
def crawl(omega=OMEGA, T=8.0, depth=None, steer=0.0, **body):
    """The crawler crawling for T seconds under the lab's plain controller (``depth`` None) or
    under the PhaseSpring one, with its energy recorded. Cached: the tests share their runs."""
    robot = turtle.robot()
    if depth is None:
        ctrl = turtle.controller(
            robot,
            stiffness=GAINS["K"],
            damping=GAINS["C"],
            inertia=GAINS["Jv"],
            flywheel_damping=GAINS["bv"],
            speed=omega,
            phase=GAINS["delta"],
            ramp_time=GAINS["ramp"],
        )
    else:
        ctrl = flywheel(robot, speed=omega, depth=depth, steer=steer, **GAINS)
    plant = vmc.sim.ModelPlant(turtle.crawler(**body), q0=pose(), max_step=1e-3)
    clock = vmc.sim.SimClock(DT)
    log = vmc.sim.run(
        plant, controller_of(robot, ctrl), clock, T=T, z0=turtle.initial_state, record=["energy"]
    )
    return Run(log, log.arrays())


def strides(rows, first=2):
    """The steps at which the flywheel completes a turn, from the turn ``first`` on."""
    turns = np.floor(rows["z"][:, 0] / (2 * np.pi)).astype(int)
    return [int(np.argmax(turns >= m)) for m in range(first, turns.max() + 1)]


def per_stride(run):
    """The mean advance [m] and yaw [deg] per stride, from one flywheel turn to the next: the
    zigzag of the gait cancels, so the rest is the drift and the turn."""
    q, steps = run.rows["q"], strides(run.rows)
    heading = np.degrees(np.unwrap([yaw(x) for x in q]))
    return np.diff(q[steps, 0]).mean(), np.diff(heading[steps]).mean()


def flywheel_speed(run):
    """The flywheel's mean speed [rad/s] over whole turns, and the whole turns' steps."""
    steps = strides(run.rows)
    t, phi = run.rows["t"].ravel(), run.rows["z"][:, 0]
    return (phi[steps[-1]] - phi[steps[0]]) / (t[steps[-1]] - t[steps[0]]), steps


def test_the_motors_are_the_two_cranks_and_nothing_drives_the_body():
    robot = turtle.crawler(efficiency=0.5)
    space = robot.model.space
    assert (space.nq, space.nv) == (9, 8)  # body: position and quaternion, then the two cranks
    assert robot.actuation.motor_sizes(space) == (2, 2)
    p = constants(robot.actuation.params)
    tau = robot.actuation.generalized_force(ca.DM([1.0, -2.0]), ca.DM(pose()), p)
    np.testing.assert_allclose(np.array(tau).ravel(), [0, 0, 0, 0, 0, 0, 0.5, -1.0])
    plant = vmc.sim.ModelPlant(robot, q0=pose(0.3, -0.7), v0=[0, 0, 0, 0, 0, 0, 1.5, -2.5])
    reading = plant.read()
    np.testing.assert_allclose(reading["motor_position"], [0.3, -0.7])
    np.testing.assert_allclose(reading["motor_velocity"], [1.5, -2.5])


def test_each_crank_has_its_own_damper_and_inertia():
    v0 = [0, 0, 0, 0, 0, 0, 1.5, -2.5]
    plant = vmc.sim.ModelPlant(turtle.crawler(crank_damping=0.3), q0=pose(0.3, -0.7, 3.0), v0=v0)
    parts = plant.elements()
    for i, side in enumerate(turtle.CRANKS):
        assert parts[f"{side}_damper"]["y"][0] == pytest.approx(pose(0.3, -0.7)[7 + i])
        assert parts[f"{side}_damper"]["ydot"][0] == pytest.approx(v0[6 + i])
        assert parts[f"{side}_damper"]["force"][0] == pytest.approx(-0.3 * v0[6 + i])
    # a crank is a wheel of inertia 2e-3 about its axis: the torque it takes to spin it up
    mass = np.array(plant.dynamics.mass(plant.q, plant.p))
    inertia = turtle.CRAWLER["crank_inertia"]
    assert mass[6, 6] == pytest.approx(inertia, rel=1e-9)  # the left crank, beside the body's
    assert mass[7, 7] == pytest.approx(inertia, rel=1e-9)


def test_a_crank_delivers_its_efficiency_times_the_commanded_torque():
    def push(efficiency):
        """The crawler's acceleration for a torque on the left crank, in free fall."""
        plant = vmc.sim.ModelPlant(turtle.crawler(efficiency=efficiency), q0=pose(height=3.0))
        args = (plant.q, plant.v, [1.0, 0.0], plant.p, 0.0)
        return (
            np.array(plant.dynamics.forward(*args)).ravel()
            - np.array(plant.dynamics.forward(plant.q, plant.v, [0.0, 0.0], plant.p, 0.0)).ravel()
        )

    full, half = push(1.0), push(0.5)
    assert full[6] > 1.0  # the left crank speeds up
    np.testing.assert_allclose(half, 0.5 * full, atol=1e-12)


def test_every_constant_reaches_the_crawler():
    robot = turtle.crawler(
        mass=2.0,
        inertia=(1.0, 2.0, 3.0),
        belly=(0.2, 0.1, 0.05),
        axle=(0.01, 0.15, 0.02),
        crank_radius=0.07,
        crank_inertia=0.004,
        crank_damping=0.3,
        gravity=(0.0, 0.0, -1.6),
        stiffness=2e3,
        ground_damping=30.0,
        friction=0.5,
        belly_friction=0.2,
        slip_speed=0.01,
        smoothing=1e-4,
        efficiency=0.7,
    )
    value = lambda name: robot.params[name].value  # noqa: E731
    parts = robot.components
    assert value("body_mass.mass") == 2.0
    np.testing.assert_array_equal(np.diag(value("body_inertia.inertia")), [1.0, 2.0, 3.0])
    for corner, (x, y) in {"fl": (1, 1), "fr": (1, -1), "bl": (-1, 1), "br": (-1, -1)}.items():
        np.testing.assert_array_equal(
            value(f"body.belly_{corner}.position"), [0.2 * x, 0.1 * y, -0.05]
        )
    np.testing.assert_array_equal(value("left.mount.position"), [0.01, 0.15, 0.02])
    np.testing.assert_array_equal(value("right.mount.position"), [0.01, -0.15, 0.02])
    for side in turtle.CRANKS:
        np.testing.assert_array_equal(value(f"{side}.foot.position"), [0.0, 0.0, -0.07])
        np.testing.assert_array_equal(value(f"{side}.j1.axis"), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(np.diag(value(f"{side}_inertia.inertia")), [0.002, 0.004, 0.002])
        assert parts[f"{side}_damper"].damping.value == 0.3
        assert value(f"{side}.efficiency.c1") == 0.7
    np.testing.assert_array_equal(value("gravity"), [0.0, 0.0, -1.6])
    contacts = [name for name in parts if name.endswith(("_foot", "_fl", "_fr", "_bl", "_br"))]
    assert len(contacts) == 6
    stiffness, damping, edge = (parts[contacts[0]].stiffness, None, parts[contacts[0]].smoothing)
    for name in contacts:
        spring, cushion, rub = parts[name], parts[f"{name}_damper"], parts[f"{name}_friction"]
        assert spring.stiffness is stiffness is rub.stiffness  # one Param: the same normal force
        assert spring.smoothing is edge is cushion.smoothing is rub.smoothing
        assert spring.coord.normal is parts[contacts[0]].coord.normal  # one ground
        assert spring.coord.origin is parts[contacts[0]].coord.origin
        assert cushion.damping.value == 30.0 and rub.speed.value == 0.01
        damping = damping or cushion.damping
        assert cushion.damping is damping
        feet = name.endswith("_foot")
        assert rub.friction.value == (0.5 if feet else 0.2)
    assert stiffness.value == 2e3 and edge.value == 1e-4
    assert parts["left_foot_friction"].friction is parts["right_foot_friction"].friction
    assert parts["belly_fl_friction"].friction is parts["belly_br_friction"].friction
    assert parts["left_foot_friction"].friction is not parts["belly_fl_friction"].friction
    assert turtle.crawler(name="other").name == "other"
    with pytest.raises(TypeError, match="mass_kg"):
        turtle.crawler(mass_kg=1.0)


FEET = ("left_foot", "right_foot")
CORNERS = ("belly_fl", "belly_fr", "belly_bl", "belly_br")


def settle(left, right, T):
    """The crawler dropped 1 cm above its rest, with these crank angles and no torque, after T."""
    plant = vmc.sim.ModelPlant(turtle.crawler(), q0=pose(left, right, 0.05), max_step=1e-3)
    plant.advance(T)
    forces = plant.elements()
    return plant, {name: float(forces[name]["force"][0]) for name in (*FEET, *CORNERS)}


def test_at_rest_with_its_feet_up_the_body_lies_on_its_belly_and_stays_still():
    c = turtle.CRAWLER
    plant, pushes = settle(np.pi, np.pi, 1.5)
    assert plant.q[2] == pytest.approx(c["belly"][2] - W / (4 * c["stiffness"]), rel=1e-6)
    still = plant.q.copy()
    plant.advance(2.0)
    np.testing.assert_allclose(plant.q, still, atol=1e-9)  # no drift
    assert np.abs(plant.v).max() < 1e-6
    for corner in CORNERS:
        assert pushes[corner] == pytest.approx(W / 4, rel=1e-6)
    assert pushes["left_foot"] == 0.0 and pushes["right_foot"] == 0.0


def test_with_both_feet_down_the_body_stands_on_them_at_the_height_the_stiffness_gives():
    c = turtle.CRAWLER
    plant, pushes = settle(0.0, 0.0, 0.6)  # a balance on a line: it holds only for a moment
    assert plant.q[2] == pytest.approx(c["crank_radius"] - W / (2 * c["stiffness"]), rel=1e-6)
    assert pushes["left_foot"] == pytest.approx(W / 2, rel=1e-6)
    assert pushes["right_foot"] == pytest.approx(W / 2, rel=1e-6)
    assert all(pushes[corner] == 0.0 for corner in CORNERS)  # the belly is clear of the ground


def test_with_one_foot_down_the_body_leans_on_it_and_on_the_far_corners_and_stands_still():
    plant, pushes = settle(0.0, np.pi, 2.0)  # the left foot down, the right one up
    still = plant.q.copy()
    plant.advance(2.0)
    np.testing.assert_allclose(plant.q, still, atol=1e-9)
    assert np.abs(plant.v).max() < 1e-6
    assert pushes["belly_fl"] == 0.0 and pushes["belly_bl"] == 0.0  # the left side is lifted
    assert pushes["belly_fr"] == pytest.approx(pushes["belly_br"], rel=1e-6)  # the right is level
    assert sum(pushes.values()) == pytest.approx(W, rel=1e-6)
    assert 0.3 * W < pushes["left_foot"] < 0.45 * W  # about W (0.07 / (0.07 + 0.11)) of the weight


def test_it_crawls_forward_straight_and_level_under_the_lab_controller():
    run = crawl()
    q = run.rows["q"]
    assert q[-1, 0] > 0.5 and abs(q[-1, 1]) < 0.05  # clearly forward, not sideways
    advance, turn = per_stride(run)
    assert advance == pytest.approx(0.135, rel=0.1) and abs(turn) < 0.2  # a stride is 13 cm
    assert 0.029 < q[:, 2].min() and q[:, 2].max() < 0.042  # it never leaves the ground or tips
    assert np.abs(q[:, 4:6]).max() < 0.07  # a rock of a few degrees, no tumbling


def test_its_distance_grows_with_the_flywheel_speed():
    slow, medium, fast = (crawl(omega=w).rows["q"][-1, 0] for w in (3.0, OMEGA, 9.0))
    assert 0.1 < slow < medium < fast
    assert medium > 1.8 * slow and fast > 1.8 * medium
    assert crawl().rows["q"][-1, 0] == pytest.approx(0.746, rel=0.03)  # a golden value


def test_the_modulated_stiffness_crawls_as_well():
    plain, waved = crawl(), crawl(depth=0.5)
    assert waved.rows["q"][-1, 0] > 0.9 * plain.rows["q"][-1, 0]
    assert abs(waved.rows["q"][-1, 1]) < 0.05 and abs(per_stride(waved)[1]) < 0.2


def test_the_flywheel_droops_under_load_by_the_amount_the_cranks_pull_on_it():
    """Equation 14: over whole turns b_v (ω̄ − ⟨φ̇⟩) is the springs' mean torque on the flywheel."""
    omega, bv, Jv = OMEGA, GAINS["bv"], GAINS["Jv"]
    run = crawl()
    speed, steps = flywheel_speed(run)
    assert 0.8 * omega < speed < 0.95 * omega  # it droops by about 14 percent
    t, phid = run.rows["t"].ravel(), run.rows["z"][:, 1]
    span = slice(steps[0], steps[-1])
    load = (
        run.rows["motor_torque"][span].sum(axis=1).mean()
    )  # what the cranks take from the springs
    gain = Jv * (phid[steps[-1]] - phid[steps[0]]) / (t[steps[-1]] - t[steps[0]])
    assert bv * (omega - speed) == pytest.approx(load + gain, rel=0.01)
    # a heavier load droops it more, and carries the body less far
    heavy = crawl(belly_friction=0.3)
    assert flywheel_speed(heavy)[0] < speed
    assert heavy.rows["q"][-1, 0] < run.rows["q"][-1, 0]


def test_the_loop_stays_passive_while_the_body_crawls():
    for run in (crawl(), crawl(depth=0.5), crawl(depth=0.5, steer=0.8)):
        balance = vmc.sim.energy_balance(run.log)
        assert balance["margin"].min() >= 0.0  # never gives more than it holds and is supplied
        assert balance["margin"].max() > 0.05
        assert np.abs(balance["injected"]).max() < 3e-3 * balance["supplied"][-1]  # books close


def test_a_positive_steer_turns_the_body_left_and_a_negative_one_right_by_the_same_amount():
    straight, left, right = (crawl(depth=0.5, steer=u) for u in (0.0, 0.8, -0.8))
    (_, level), (_, turn_left), (_, turn_right) = map(per_stride, (straight, left, right))
    assert abs(level) < 0.2 and turn_left > 5.0 and turn_right < -5.0  # degrees per stride
    assert turn_left == pytest.approx(-turn_right, rel=0.05)  # the same, mirrored
    assert left.rows["q"][-1, 1] > 0.1 and right.rows["q"][-1, 1] < -0.1  # and the path bends
    gentle = per_stride(crawl(depth=0.5, steer=0.4))[1]
    assert 0.3 * turn_left < gentle < 0.8 * turn_left  # a smaller steer turns it less


def test_a_crawler_with_its_cranks_swapped_is_the_mirror_image():
    mirror = crawl(depth=0.5, steer=0.8, axle=(0.0, -0.11, 0.0))
    mine = crawl(depth=0.5, steer=0.8)
    a, b = mine.rows["q"], mirror.rows["q"]
    np.testing.assert_allclose(b[:, [0, 2]], a[:, [0, 2]], atol=1e-6)  # the same forward and up
    np.testing.assert_allclose(b[:, 1], -a[:, 1], atol=1e-6)
    np.testing.assert_allclose([yaw(x) for x in b], -np.array([yaw(x) for x in a]), atol=1e-6)


def test_without_friction_it_does_not_advance():
    run = crawl(friction=0.0, belly_friction=0.0)
    q = run.rows["q"]
    assert np.abs(q[:, :2]).max() < 1e-6  # not a millimetre, not a micrometre
    speed, _ = flywheel_speed(run)
    assert speed > 0.8 * OMEGA  # the cranks do turn, in the air and on the ground
    assert np.ptp(run.rows["motor_position"][:, 0]) > 20.0
    assert crawl().rows["q"][-1, 0] > 0.5  # the same run with friction goes far
