"""Equations of motion from a function: the same arm as a black box and as parts."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control.underactuated import Passivation
from virtualmodelcontrol.dynamics import compile_dynamics
from virtualmodelcontrol.estimation import KalmanFilter, MomentumObserver
from virtualmodelcontrol.models import Equations, FunctionModel, SerialChain
from virtualmodelcontrol.testing import check_model

L1, L2, M1, M2, G = 0.5, 0.4, 1.2, 0.8, 9.81


def add_joint_forces(robot):
    """A spring on the first joint and a damper on both: components, which the equations of the
    black box do not contain."""
    q = robot.joint(slice(0, 2))
    robot.add("spring", vmc.LinearSpring(robot.joint(0) - 0.3, 4.0))
    robot.add("damper", vmc.LinearDamper(q, 0.15))
    return robot


def parts(forces=True):
    """A planar two-link arm from components: point masses at the ends of the links, gravity."""
    chain = SerialChain(
        ["revolute", "revolute"],
        axes=[(0.0, 1.0, 0.0)] * 2,
        points=[(0.0, 0.0, 0.0), (L1, 0.0, 0.0)],
        sites={"m1": (1, (L1, 0.0, 0.0)), "m2": (2, (L1 + L2, 0.0, 0.0))},
    )
    robot = vmc.Mechanism("arm", model=chain)
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -G], unit="m/s^2", scope="design"))
    robot.add("body1", vmc.PointMass(robot.point("m1"), M1))
    robot.add("body2", vmc.PointMass(robot.point("m2"), M2))
    robot.add("weight", vmc.Gravity(robot))
    return add_joint_forces(robot) if forces else robot


def residual(q, v, a, tau, f, p):
    """The textbook equations of the arm, M a + h + dV/dq - tau - f, with the angles taken about
    y from the x axis (z is up: the arm hangs at q = pi / 2)."""
    c1, c2, s2, c12 = ca.cos(q[0]), ca.cos(q[1]), ca.sin(q[1]), ca.cos(q[0] + q[1])
    m22 = M2 * L2**2
    m12 = m22 + M2 * L1 * L2 * c2
    m11 = (M1 + M2) * L1**2 + m22 + 2 * M2 * L1 * L2 * c2
    mass = ca.vertcat(ca.horzcat(m11, m12), ca.horzcat(m12, m22))
    coriolis = M2 * L1 * L2 * s2
    h = ca.vertcat(-coriolis * (2 * v[0] * v[1] + v[1] ** 2), coriolis * v[0] ** 2)
    weight = -G * ca.vertcat((M1 + M2) * L1 * c1 + M2 * L2 * c12, M2 * L2 * c12)  # dV/dq
    return ca.mtimes(mass, a) + h + weight - tau - f


def energy(q, v, p):
    """The kinetic and the potential energy of the arm."""
    c2 = ca.cos(q[1])
    T = 0.5 * (
        ((M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2) * v[0] ** 2
        + 2 * (M2 * L2**2 + M2 * L1 * L2 * c2) * v[0] * v[1]
        + M2 * L2**2 * v[1] ** 2
    )
    V = -G * ((M1 + M2) * L1 * ca.sin(q[0]) + M2 * L2 * ca.sin(q[0] + q[1]))
    return T, V


def frame(q, at, p):
    """The end of the second link."""
    a = q[0] + q[1]
    R = ca.vertcat(
        ca.horzcat(ca.cos(a), 0, ca.sin(a)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(a), 0, ca.cos(a)),
    )
    tip = ca.vertcat(L1 * ca.cos(q[0]) + L2 * ca.cos(a), 0, -L1 * ca.sin(q[0]) - L2 * ca.sin(a))
    return R, tip


def black_box(with_energy=True, forces=True, residual=residual):
    """The same arm with its dynamics from the function above."""
    model = FunctionModel(
        frame, 2, sites=("tip",), equations=Equations(residual, energy if with_energy else None)
    )
    robot = vmc.Mechanism("arm", model=model)
    return add_joint_forces(robot) if forces else robot


def states(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)) for _ in range(n)]


def test_the_black_box_has_the_dynamics_of_the_parts_at_any_state():
    a, b = compile_dynamics(parts()), compile_dynamics(black_box())
    pa, pb = a.live_values(), b.live_values()
    assert np.abs(pa).max() > 0.1  # the spring and the damper have their values
    for q, v, u in states():
        acc = np.array([0.7, -1.1])
        np.testing.assert_allclose(b.mass(q, pb), a.mass(q, pa), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            b.forward(q, v, u, pb, 0.0), a.forward(q, v, u, pa, 0.0), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            b.residual(q, v, acc, u, pb, 0.0), a.residual(q, v, acc, u, pa, 0.0), atol=1e-10
        )
        T, V = a.energy(q, v, pa, 0.0)
        Tb, Vb = b.energy(q, v, pb, 0.0)
        # the parts' storage is the weight and the spring, the box's is its own V and the spring
        np.testing.assert_allclose([float(Tb), float(Vb)], [float(T), float(V)], rtol=1e-12)


def test_the_components_of_a_black_box_still_count_and_are_reported():
    box = black_box()
    dynamics = compile_dynamics(box)
    assert dynamics.element_names == ["spring", "damper"]
    q, v = np.array([0.8, 0.4]), np.array([0.3, -0.2])
    p = dynamics.live_values()
    dissipated = float(dynamics.power(q, v, [0.0, 0.0], p, 0.0)[1])
    assert dissipated == pytest.approx(-0.15 * (0.3**2 + 0.2**2))
    free = compile_dynamics(black_box(forces=False))
    # with nothing but the equations the acceleration is that of the equations alone
    r0 = np.array(free.residual(q, v, [0, 0], [0, 0], [], 0.0))
    want = np.linalg.solve(np.array(free.mass(q, [])), -r0.ravel())
    np.testing.assert_allclose(
        np.array(free.forward(q, v, [0.0, 0.0], [], 0.0)).ravel(), want, rtol=1e-12
    )


def test_a_simulation_of_the_black_box_follows_the_parts_and_closer_with_a_smaller_step():
    gaps = []
    for step in (1e-3, 2.5e-4):
        runs = []
        for robot in (parts(), black_box()):
            plant = vmc.sim.ModelPlant(robot, q0=[0.9, 0.5], v0=[0.0, 0.5], max_step=step)
            plant.write(vmc.Signals(0.0, motor_torque=[0.4, -0.2]))
            plant.advance(1.0)
            runs.append(np.concatenate([plant.q, plant.v]))
        gaps.append(np.abs(runs[0] - runs[1]).max())
        assert np.abs(runs[0][:2] - [0.9, 0.5]).max() > 0.05  # the arm moved
    assert gaps[0] < 5e-3 and gaps[1] < 0.35 * gaps[0]


def test_a_stiff_spring_on_the_black_box_is_stable_at_a_long_step_as_it_is_on_the_parts():
    for robot in (parts(), black_box()):
        robot.add("wall", vmc.LinearSpring(robot.joint(0) - 0.6, 2.0e5))  # stiff: 440 rad/s
        robot.add("brake", vmc.LinearDamper(robot.joint(0), 4.0e3))  # and a damper to match
        plant = vmc.sim.ModelPlant(robot, q0=[0.9, 0.5], max_step=2e-3)
        plant.advance(1.0)
        assert np.abs(plant.q[0] - 0.6) < 0.01 and np.abs(plant.v[0]) < 0.1  # held, not blown


def test_the_energy_of_the_black_box_is_that_of_the_parts_and_the_robot_with_it_never_gains():
    for q, v, _ in states(3):
        a = vmc.sim.ModelPlant(parts(), q0=q, v0=v).energy()
        b = vmc.sim.ModelPlant(black_box(), q0=q, v0=v).energy()
        assert b == pytest.approx(a, rel=1e-12)
    worst = check_model(black_box(), at=["tip"])
    assert worst["energy_growth"] <= 1e-6 and worst["mass_symmetric"] < 1e-12
    assert worst["mass_positive"] == 0.0


def test_check_model_finds_a_mass_that_is_not_symmetric_and_an_energy_that_grows():
    def skewed(q, v, a, tau, f, p):
        return residual(q, v, a, tau, f, p) + ca.vertcat(0.3 * a[1], 0.0)  # M is not symmetric

    with pytest.raises(AssertionError, match="mass_symmetric"):
        check_model(black_box(residual=skewed))

    def pumped(q, v, a, tau, f, p):
        return residual(q, v, a, tau, f, p) - 0.5 * v  # a force along v: it adds energy

    with pytest.raises(AssertionError, match="energy_growth"):
        check_model(black_box(residual=pumped, forces=False))
    with pytest.raises(AssertionError, match="equations"):
        check_model(
            black_box(residual=lambda q, v, a, tau, f, p: a**2 + residual(q, v, a, tau, f, p))
        )


def test_check_model_wants_a_positive_mass_and_lets_a_source_pump_energy_in():
    def negative(q, v, a, tau, f, p):
        return -residual(q, v, a, tau, f, p)  # M is minus the mass matrix

    with pytest.raises(AssertionError, match="mass_positive"):
        check_model(black_box(residual=negative))
    pushed = black_box(forces=False)
    pushed.add("push", vmc.ForceSource(pushed.joint(0), [3.0]))  # does work: the energy grows
    assert check_model(pushed)["mass_positive"] == 0.0


def test_without_an_energy_the_black_box_runs_and_the_tools_that_need_one_refuse():
    box = black_box(with_energy=False)
    dynamics = compile_dynamics(box)
    assert dynamics.energy is None and dynamics.power is None
    plant = vmc.sim.ModelPlant(box, q0=[0.9, 0.5])
    plant.advance(0.1)  # simulating needs none
    assert np.abs(plant.q - [0.9, 0.5]).max() > 1e-4
    with pytest.raises(ValueError, match="needs the energy"):
        plant.energy()
    stage = Passivation(dynamics)
    meas = vmc.Signals(0.0, motor_velocity=[0.1, 0.1], q=[0.9, 0.5], v=[0.1, 0.1])
    with pytest.raises(ValueError, match="passivity correction needs the energy"):
        stage(np.zeros(2), meas)
    with_energy = Passivation(compile_dynamics(black_box()))
    assert np.isfinite(with_energy(np.ones(2), meas)).all()


def test_a_robot_with_equations_has_no_inertances_and_its_residual_is_affine_in_a():
    box = black_box()
    box.add("extra", vmc.Inertance(box.joint(0), 1.0))
    with pytest.raises(ValueError, match="no inertances"):
        compile_dynamics(box)

    def squared(q, v, a, tau, f, p):
        return a**2 - tau

    with pytest.raises(ValueError, match="affine in a"):
        compile_dynamics(black_box(residual=squared, forces=False))


def test_the_params_of_the_model_reach_the_residual_and_stay_live():
    mass = vmc.Param("mass", 2.0, unit="kg", scope="stage")

    def pendulum(q, v, a, tau, f, p):
        return p["mass"] * a - tau - f

    model = FunctionModel(frame, 1, [mass], sites=("tip",), equations=Equations(pendulum))
    robot = vmc.Mechanism("slider", model=model)
    dynamics = compile_dynamics(robot)
    assert dynamics.live == ["slider.mass"]
    np.testing.assert_allclose(
        np.array(dynamics.forward([0.0], [0.0], [3.0], [2.0], 0.0)).ravel(), [1.5]
    )
    np.testing.assert_allclose(
        np.array(dynamics.forward([0.0], [0.0], [3.0], [4.0], 0.0)).ravel(), [0.75]
    )


def controlled(robot):
    """The arm held at (1.0, 0.6) by a spring and a damper."""
    ctrl = vmc.Mechanism("ctrl")
    both = robot.joint(slice(0, 2))
    ctrl.add("hold", vmc.LinearSpring(both - [1.0, 0.6], [[20.0, 0.0], [0.0, 15.0]]))
    ctrl.add("damp", vmc.LinearDamper(both, 2.0))
    return vmc.VirtualMechanismSystem(robot, ctrl)


def test_the_rollout_the_kalman_filter_and_the_plan_run_on_the_black_box():
    reference, box = (controlled(robot) for robot in (parts(), black_box()))
    a = vmc.sim.rollout(reference, [0.9, 0.5], 1.0, 0.001, max_step=0.001)
    b = vmc.sim.rollout(box, [0.9, 0.5], 1.0, 0.001, max_step=0.001)
    assert np.abs(a["q"] - b["q"]).max() < 1e-4 and np.abs(a["q"] - a["q"][0]).max() > 0.05
    filters = [KalmanFilter(system, 0.01) for system in (reference, box)]
    for kf in filters:
        kf.reset([0.9, 0.5])
        kf.predict([0.3, -0.1])
    np.testing.assert_allclose(filters[1].q, filters[0].q, atol=1e-9)
    np.testing.assert_allclose(filters[1].v, filters[0].v, atol=1e-8)
    observers = [MomentumObserver(system, 0.002, 20.0) for system in (reference, box)]
    for k in range(40):  # the same motion, the same force the model does not explain
        for observer in observers:
            observer(a["q"][k], a["v"][k], a["u"][k], 0.002 * k)
    np.testing.assert_allclose(observers[1].r, observers[0].r, atol=1e-8)
    assert np.abs(observers[0].r).max() > 1e-4  # a force the rollout's steps leave, not zero
    plans = []
    for system in (reference, box):
        problem = opt.Problem(system)
        problem.add(opt.Collocation([0.9, 0.5], 1.0, 21, scheme="hermite-simpson"))
        plans.append(problem.solve())
    assert all(plan.converged for plan in plans)
    np.testing.assert_allclose(plans[1].q, plans[0].q, atol=1e-8)
    np.testing.assert_allclose(plans[1].u, plans[0].u, atol=1e-7)
