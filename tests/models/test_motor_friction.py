"""The motors' friction, Coulomb and viscous on their speed, its compensation, and the old names."""

import warnings

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import FrictionCompensation, StaticFrictionCompensation
from virtualmodelcontrol.identification import fit_efficiency
from virtualmodelcontrol.models import Efficiency, JointSpace, StaticFriction
from virtualmodelcontrol.robots import adapt

F, EPS, D = 0.3, 0.02, 0.05  # Coulomb [N·m], its smoothing [rad/s], viscous [N·m·s/rad]
V = 0.03  # the speed [rad/s] past which the compensation fades
TIGHT = {"ipopt.tol": 1e-10, "ipopt.acceptable_iter": 0, "ipopt.constr_viol_tol": 1e-10}


def friction_of(robot, q, v):
    """The torque [N·m] that the robot's friction gives at (q, v): its residual at rest, with
    no command, against the same robot without friction."""
    bare = adapt.add_dynamics(adapt.finger())
    torques = []
    for r in (robot, bare):
        dyn = vmc.compile_dynamics(r)
        a, u = np.zeros(len(q)), np.zeros(dyn.n_u)
        torques.append(np.array(dyn.residual(q, v, a, u, dyn.live_values(), 0.0)).ravel())
    return torques[1] - torques[0]


def test_the_finger_motors_take_coulomb_and_viscous_friction_against_their_speed():
    finger = adapt.add_dynamics(
        adapt.finger(),
        damping=adapt.MOTOR_DAMPING,
        coulomb=adapt.MOTOR_FRICTION,
        inertia=adapt.ROTOR_INERTIA,
    )
    Fc, Dv, eps = np.array(adapt.MOTOR_FRICTION), np.array(adapt.MOTOR_DAMPING), 0.005
    assert adapt.FRICTION_SMOOTHING == eps
    q = np.array([0.3, 0.5])
    for rate in ([0.0, 0.0], [0.002, -0.004], [0.05, -0.1], [2.0, -3.0]):
        rate = np.array(rate)
        expected = -Fc * np.tanh(rate / eps) - Dv * rate
        np.testing.assert_allclose(friction_of(finger, q, rate), expected, atol=1e-12)
        assert friction_of(finger, q, rate) @ rate <= 0.0  # it only takes energy
    dyn = vmc.compile_dynamics(finger)
    bare = vmc.compile_dynamics(adapt.add_dynamics(adapt.finger()))
    extra = np.array(dyn.mass(q, dyn.live_values())) - np.array(bare.mass(q, bare.live_values()))
    np.testing.assert_allclose(extra, adapt.ROTOR_INERTIA * np.eye(2), atol=1e-15)
    wide = adapt.add_dynamics(adapt.finger(), coulomb=0.1, smoothing=0.05)
    np.testing.assert_allclose(
        friction_of(wide, q, np.array([0.05, -0.02])),
        -0.1 * np.tanh(np.array([0.05, -0.02]) / 0.05),
        atol=1e-12,
    )


def test_the_friction_and_the_lab_compensation_are_off_unless_asked_for():
    plain = adapt.add_dynamics(adapt.finger())
    assert not {"coulomb", "rotor", "damping"} & set(plain.components)
    assert adapt.COMPENSATION == (0.20, 0.03) and adapt.HAND_COMPENSATION == (0.1, 0.03)
    finger, hand = adapt.output_stage()[0], adapt.hand_output_stage()[0]
    assert (finger.max_torque, finger.velocity, finger.width) == (0.20, 0.03, 0.0)
    assert (hand.max_torque, hand.velocity, hand.width) == (0.1, 0.03, 0.0)


def test_every_old_friction_name_warns_that_2_0_0_removes_it():
    calls = {
        "StaticFriction": lambda: StaticFriction(0.1),
        "Efficiency(friction=)": lambda: Efficiency(1.0, friction={"breakaway": 0.1}),
        "StaticFrictionCompensation": lambda: StaticFrictionCompensation(0.1),
        "fit_efficiency": lambda: fit_efficiency(
            np.linspace(-1, 1, 41)[:, None], np.linspace(-1, 1, 41)[:, None], friction=True
        ),
        "finger_friction": adapt.finger_friction,
        "hand_friction": adapt.hand_friction,
        "FRICTION": lambda: adapt.FRICTION,
        "HAND_FRICTION": lambda: adapt.HAND_FRICTION,
    }
    for name, call in calls.items():
        with pytest.warns(DeprecationWarning, match="2.0.0") as record:
            call()
        assert len(record) == 1, name  # once, not again from inside the library
        assert record[0].filename == __file__, name  # and pointing at the caller
    with pytest.warns(DeprecationWarning):
        assert adapt.FRICTION == adapt.COMPENSATION
        assert adapt.HAND_FRICTION == adapt.HAND_COMPENSATION
    with pytest.raises(AttributeError, match="NOTHING"):
        adapt.NOTHING  # noqa: B018
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the new names are quiet
        FrictionCompensation(0.1, V, width=0.01)
        adapt.add_dynamics(adapt.finger(), coulomb=adapt.MOTOR_FRICTION)
        adapt.output_stage(), adapt.hand_output_stage()


def spring_to_goal(coulomb=None, goal=1.0, eps=EPS):
    """An inertia on one joint whose motor has Coulomb friction, smoothed over ``eps``, and
    viscous friction, held by a virtual spring to ``goal`` and a damper."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="rad"), actuation=vmc.models.Direct())
    q = robot.joint(0)
    robot.add("inertia", vmc.Inertance(q, 0.05))
    if coulomb is not None:
        robot.add("coulomb", vmc.TanhDamper(robot.motor(0), coulomb / eps, coulomb))
        robot.add("viscous", vmc.LinearDamper(robot.motor(0), D))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(q - vmc.Ref("goal", 1, [goal]), 1.0))
    ctrl.add("damper", vmc.LinearDamper(q, 0.2))
    return vmc.VirtualMechanismSystem(robot, ctrl)


def final_error(system, output=(), T=20.0):
    """How far from the goal the joint ends, with the numeric output stages, as on a robot."""
    plant = vmc.sim.ModelPlant(system.robot, max_step=0.01)
    controller = vmc.VMCController(vmc.compile(system), output=list(output))
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=0.01), T=T).arrays()
    return 1.0 - float(log["q"][-1, 0])


def test_the_joint_stops_short_by_about_F_over_K_and_the_compensation_closes_the_gap():
    assert abs(final_error(spring_to_goal())) < 1e-3
    short = final_error(spring_to_goal(F, eps=0.005))
    assert 0.1 < short < F  # stops near F / K, then only creeps
    for stage in (FrictionCompensation(0.8 * F, V), FrictionCompensation(0.8 * F, V, width=0.01)):
        assert abs(final_error(spring_to_goal(F, eps=0.005), [stage])) < 0.25 * short


def test_the_compensation_takes_a_smooth_sign_and_the_same_map_in_a_plan():
    import casadi as ca

    from virtualmodelcontrol.control.output import PlantState

    F_hat, w = np.array([0.2, 0.1]), 0.02
    stage = FrictionCompensation(F_hat, V, width=w)
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, rate = rng.uniform(-0.3, 0.3, 2), rng.uniform(-0.1, 0.1, 2)
        expected = u + F_hat * np.exp(-((rate / V) ** 2)) * u / np.sqrt(u * u + w * w)
        meas = vmc.Signals(0.0, motor_position=np.zeros(2), motor_velocity=rate)
        np.testing.assert_allclose(stage(u, meas), expected, atol=1e-15)
        got = stage.symbolic(ca.DM(u), PlantState(ca.DM(rate), ca.DM(0.0)))
        np.testing.assert_allclose(np.array(ca.evalf(got)).ravel(), expected, atol=1e-15)
    still = vmc.Signals(0.0, motor_position=np.zeros(1), motor_velocity=np.zeros(1))
    small = np.array([0.005])
    plain = FrictionCompensation(0.2, V)
    assert plain.width == 0.0 and FrictionCompensation.needs_plant
    assert plain(small, still)[0] == pytest.approx(0.205)  # the sign: all of F_hat
    smooth = FrictionCompensation(0.2, V, width=w)
    assert smooth(small, still)[0] == pytest.approx(0.005 + 0.2 * 0.005 / np.hypot(0.005, w))
    assert plain(np.zeros(1), still)[0] == 0.0 == smooth(np.zeros(1), still)[0]
    moving = vmc.Signals(0.0, motor_position=np.zeros(1), motor_velocity=np.array([0.1]))
    assert abs(plain(small, moving)[0] - 0.005) < 1e-5  # faded once the motor moves
    with pytest.raises(ValueError, match="width"):
        plain.symbolic(ca.DM([0.1]), PlantState(ca.DM([0.0]), ca.DM(0.0)))


def test_a_plan_with_the_motor_friction_and_its_compensation_is_the_simulation_step_for_step():
    stage = FrictionCompensation(0.8 * F, V, width=0.02)
    plans = {}
    for name, output in (("bare", []), ("compensated", [stage])):
        system = spring_to_goal(F)
        problem = opt.Problem(system, output=output, options=TIGHT)
        problem.add(opt.Shooting([0.0], 3.0, 31, v0=[0.0], substeps=5))
        plan = problem.solve()
        log = vmc.sim.rollout(system, [0.0], 3.0, 0.02, v0=[0.0], max_step=0.02, output=output)
        assert plan.converged and plan.violation < 1e-8
        np.testing.assert_allclose(plan.q[:-1], log["q"][::5], atol=1e-8)
        np.testing.assert_allclose(plan.u[:-1], log["u"][::5], atol=1e-8)
        plans[name] = plan.q
    free = opt.Problem(spring_to_goal(None))
    free.add(opt.Shooting([0.0], 3.0, 31, v0=[0.0], substeps=5))
    assert np.abs(free.solve().q - plans["bare"]).max() > 0.1  # the friction is in the plan
    assert np.abs(plans["compensated"] - plans["bare"]).max() > 0.02  # and the compensation
    problem = opt.Problem(spring_to_goal(F), output=[FrictionCompensation(0.2, V)])
    problem.add(opt.Shooting([0.0], 1.0, 5, v0=[0.0]))
    with pytest.raises(ValueError, match="width"):
        problem.solve()
