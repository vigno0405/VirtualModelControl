"""More components: a rigid body's rotational inertia, a one-way damper, bounded sources, mass
spread along a continuum."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import SerialChain
from virtualmodelcontrol.robots import helyx

INERTIA = np.array([[0.30, 0.05, 0.02], [0.05, 0.25, 0.04], [0.02, 0.04, 0.40]])


def floating(site="body"):
    chain = SerialChain(["floating"], axes=[None], points=[[0, 0, 0]], sites={site: (1, [0, 0, 0])})
    return vmc.Mechanism("body", model=chain)


def rigid_body(inertia, mass=3.0):
    robot = floating()
    robot.add("mass", vmc.PointMass(robot.point("body"), mass))
    robot.add("spin", vmc.RotationalInertia(vmc.FrameRotation(robot.model, "body"), inertia))
    return robot


def test_a_rigid_body_with_a_full_inertia_matrix_follows_eulers_equations():
    spin = np.array([1.5, 4.0, -0.5])
    robot = rigid_body(INERTIA)
    plant = vmc.sim.ModelPlant(robot, v0=np.concatenate([np.zeros(3), spin]), max_step=1e-4)
    times = np.linspace(0.0, 1.5, 16)
    exact = solve_ivp(
        lambda t, w: np.linalg.solve(INERTIA, np.cross(INERTIA @ w, w)),
        (0.0, 1.5), spin, t_eval=times, rtol=1e-11, atol=1e-13,
    )  # fmt: skip
    rates = []
    for t in times:
        plant.advance(t - plant.t)
        rates.append(plant.v[3:].copy())
    np.testing.assert_allclose(rates, exact.y.T, atol=2e-2)
    assert np.ptp(exact.y[0]) > 0.5  # the axes of the spin move: the full matrix matters
    np.testing.assert_allclose(plant.v[:3], 0.0, atol=1e-12)  # and nothing pushes the centre


def test_the_inertia_of_four_point_masses_and_the_rotational_inertia_move_alike():
    masses = np.array([1.0, 2.0, 3.0, 1.5])
    points = np.array([[0.2, 0.0, 0.1], [0.0, 0.1, -0.1], [-0.1, 0.05, 0.0], [0.0, 0.0, 0.0]])
    points[3] = -(masses[:3] @ points[:3]) / masses[3]  # the centre of mass at the origin
    inertia = sum(
        m * (p @ p * np.eye(3) - np.outer(p, p)) for m, p in zip(masses, points, strict=True)
    )
    chain = SerialChain(
        ["floating"],
        axes=[None],
        points=[[0, 0, 0]],
        sites={f"p{i}": (1, p) for i, p in enumerate(points)},
    )
    cloud = vmc.Mechanism("cloud", model=chain)
    for i, m in enumerate(masses):
        cloud.add(f"m{i}", vmc.PointMass(cloud.point(f"p{i}"), m))
    body = rigid_body(inertia, mass=masses.sum())
    v0 = np.array([0.2, -0.1, 0.3, 1.5, 4.0, -0.5])
    ends = []
    for robot in (cloud, body):
        plant = vmc.sim.ModelPlant(robot, v0=v0, max_step=1e-4)
        plant.advance(1.0)
        ends.append((plant.q.copy(), plant.v.copy()))
    np.testing.assert_allclose(ends[0][1], ends[1][1], atol=1e-7)
    np.testing.assert_allclose(ends[0][0], ends[1][0], atol=1e-7)


def test_the_rotational_inertia_is_a_design_param_taken_as_moments_or_a_matrix():
    robot = rigid_body([0.1, 0.2, 0.3])
    inertia = robot.components["spin"].inertia
    assert inertia.shape == (3, 3) and inertia.unit == "kg*m^2" and inertia.scope == "design"
    np.testing.assert_array_equal(inertia.value, np.diag([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="FrameRotation"):
        vmc.RotationalInertia(robot.point("body"), INERTIA)


def test_gravity_acts_on_the_point_mass_of_a_rigid_body_and_not_on_its_spin():
    robot = rigid_body(INERTIA)
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("gravity", vmc.Gravity(robot))
    plant = vmc.sim.ModelPlant(robot, v0=[0, 0, 0, 0.3, -0.2, 0.1], max_step=1e-4)
    plant.advance(0.5)
    assert plant.q[2] == pytest.approx(-0.5 * 9.81 * 0.25, rel=1e-3)
    np.testing.assert_allclose(plant.v[3:], [0.3, -0.2, 0.1], atol=2e-2)  # no torque from gravity


def joints(n=3):
    return vmc.Mechanism("robot", model=vmc.models.JointSpace(n, unit="m"))


def forces_of(ctrl, robot, v):
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    q, z = np.zeros(len(v)), np.zeros(0)
    return np.array(compiled.tau(q, v, z, compiled.live_values(), 0.0)).ravel()


def test_a_diode_damper_acts_in_one_direction_only_and_never_adds_energy():
    robot = joints()
    v = np.array([0.5, -0.5, 0.0])
    for sign, want in ((1.0, [-1.5, 0.0, 0.0]), (-1.0, [0.0, 3.0 * 0.5, 0.0])):
        ctrl = vmc.Mechanism("ctrl")
        ctrl.add("diode", vmc.DiodeDamper(robot.joint(slice(0, 3)), 3.0, sign=sign))
        np.testing.assert_allclose(forces_of(ctrl, robot, v), want, atol=1e-14)
    rng = np.random.default_rng(0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("diode", vmc.DiodeDamper(robot.joint(slice(0, 3)), [1.0, 2.0, 3.0], smoothing=0.05))
    for _ in range(50):
        velocity = rng.normal(0, 1, 3)
        assert forces_of(ctrl, robot, velocity) @ velocity <= 1e-14  # f . v never positive
    assert abs(forces_of(ctrl, robot, np.array([2.0, 2.0, 2.0]))).min() > 0.5  # it damps going up
    np.testing.assert_allclose(forces_of(ctrl, robot, np.zeros(3)), 0.0, atol=1e-15)


def test_a_source_can_be_bounded_in_force_and_in_power():
    robot = joints(1)
    y = robot.joint(0)

    def push(v, **bounds):
        ctrl = vmc.Mechanism("ctrl")
        ctrl.add("push", vmc.ForceSource(y, [10.0], **bounds))
        return float(forces_of(ctrl, robot, np.array([v]))[0])

    assert push(0.0) == 10.0  # no bound, no change
    assert push(0.0, max_force=100.0) == pytest.approx(100.0 * np.tanh(0.1), rel=1e-12)
    assert push(0.0, max_force=3.0) == pytest.approx(3.0 * np.tanh(10.0 / 3.0), rel=1e-12)
    assert push(0.0, max_force=3.0) < 3.0  # saturated, never above the bound
    assert push(100.0, max_power=50.0) == pytest.approx(10.0 * 50.0 / (50.0 + 1000.0), rel=1e-12)
    assert 100.0 * push(100.0, max_power=50.0) < 50.0  # the power it delivers stays below the bound
    assert push(0.0, max_power=50.0) == 10.0  # at rest the power is zero


def test_mass_spread_along_an_arm_has_the_moments_of_the_continuum():
    arm = helyx.arm("290-145-145", masses=[1e-9] * 3)
    robot = vmc.Mechanism("rod", model=arm.model, actuation=arm.actuation)
    names = robot.add_mass_along("rod", 0.8, 0.25, 0.75, n=3)
    assert names == ["rod1", "rod2", "rod3"]
    kin = vmc.Kinematics(robot)
    parts = [robot.components[name] for name in names]
    mass = np.array([float(c.mass.value) for c in parts])
    z = np.array([kin.position(np.zeros(9), float(c.coord.s.value))[2] for c in parts])
    L = 0.58
    assert mass.sum() == pytest.approx(0.8, rel=1e-14)
    assert mass @ z == pytest.approx(0.8 * L * 0.5, rel=1e-8)  # the middle of 0.25 to 0.75
    second = 0.8 * L**2 * (0.25**2 + 0.25 * 0.75 + 0.75**2) / 3
    assert mass @ z**2 == pytest.approx(second, rel=1e-8)  # exact for a cubic: three nodes


def test_a_rigid_body_on_a_rotational_spring_oscillates_at_the_square_root_of_k_over_i():
    # a free body turns the same whatever the size and sign of its inertia: a spring tells them
    moments, k = np.array([0.2, 0.5, 0.8]), 40.0
    robot = rigid_body(moments)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(vmc.OrientationError(robot.model, "body"), k))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    for axis, inertia in enumerate(moments):
        q0 = vmc.Quaternion().integrate(vmc.Quaternion().neutral(), 0.05 * np.eye(3)[axis])
        plant = vmc.sim.ModelPlant(robot, q0=np.concatenate([np.zeros(3), q0]), max_step=1e-4)
        log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1e-3), T=1.5)
        angle = log.arrays()["q"][:, 4 + axis] * 2  # about the axis, small
        crossings = np.where(np.diff(np.sign(angle)) != 0)[0]
        period = 2 * np.diff(log.arrays()["t"].ravel()[crossings]).mean()  # half periods
        assert period == pytest.approx(2 * np.pi * np.sqrt(inertia / k), rel=0.01)


def test_a_diode_damper_opens_over_its_smoothing_speed():
    robot = joints(1)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("diode", vmc.DiodeDamper(robot.joint(0), 3.0, smoothing=0.2))
    gate = 0.5 * (1 + np.tanh(0.5))  # at a speed of the smoothing
    force = forces_of(ctrl, robot, np.array([0.2]))[0]
    assert force == pytest.approx(-3.0 * gate * 0.2, rel=1e-12)


def test_the_power_bound_counts_motion_in_either_direction():
    robot = joints(1)

    def push(v):
        ctrl = vmc.Mechanism("ctrl")
        ctrl.add("push", vmc.ForceSource(robot.joint(0), [10.0], max_power=50.0))
        return float(forces_of(ctrl, robot, np.array([v]))[0])

    assert push(-100.0) == pytest.approx(push(100.0), rel=1e-12)
    assert push(-100.0) > 0.0 and push(-100.0) * (-100.0) < 0.0  # it brakes, it is not bounded


def test_an_inertance_on_a_difference_is_an_inerter_between_two_masses():
    m1, m2, b, k1, k2 = 2.0, 3.0, 0.7, 40.0, 25.0
    robot = joints(1)
    ctrl = vmc.Mechanism("ctrl")
    z1, z2 = ctrl.add_state("z1", unit="m"), ctrl.add_state("z2", unit="m")
    ctrl.add("m1", vmc.Inertance(z1, m1))
    ctrl.add("m2", vmc.Inertance(z2, m2))
    ctrl.add("inerter", vmc.Inertance(z1 - z2, b))  # a force b (z1'' - z2'') between them
    ctrl.add("k1", vmc.LinearSpring(z1, k1))
    ctrl.add("k2", vmc.LinearSpring(z2 - z1, k2))
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    x = np.array([0.03, -0.05])
    z = np.concatenate([x, np.zeros(2)])  # at rest: no velocity terms
    _, zdot = compiled.law(np.zeros(1), np.zeros(1), z, compiled.live_values(), 0.0)
    acceleration = np.array(zdot).ravel()[2:]
    force = np.array([-k1 * x[0] + k2 * (x[1] - x[0]), -k2 * (x[1] - x[0])])
    mass = np.array([[m1 + b, -b], [-b, m2 + b]])
    np.testing.assert_allclose(acceleration, np.linalg.solve(mass, force), rtol=1e-12)
    assert abs(acceleration[0] - force[0] / m1) > 1e-3  # the inerter is not a mass to ground
