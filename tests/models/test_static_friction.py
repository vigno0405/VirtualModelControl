"""Static friction of an actuator: a smooth map of the applied torque, in every model."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import StaticFrictionCompensation
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.core.registry import get
from virtualmodelcontrol.models import Direct, Efficiency, StaticFriction
from virtualmodelcontrol.models.friction import as_friction
from virtualmodelcontrol.robots import adapt, helyx, planar

F, W = 0.2, 0.01
U = np.linspace(-0.6, 0.6, 1201)


def symbolic(friction, u):
    x = ca.SX.sym("x", len(u))
    f = ca.Function("f", [x], [friction.torque(x, constants(friction.params))])
    return np.array(f(u)).ravel()


def test_the_default_takes_nothing_and_a_transmission_without_friction_is_as_it_was():
    u = np.array([[0.1, -0.2, 0.3], [0.0, 0.05, -0.4]])
    assert np.array_equal(StaticFriction()(u), np.zeros_like(u))
    assert np.array_equal(Efficiency(0.12, friction=StaticFriction())(u), 0.12 * u)  # F = Fc = 0
    plain = Efficiency(0.12)
    assert plain.friction is None and list(plain.params) == ["c1"]
    assert plain.to_dict() == {"type": "polynomial", "coefficients": [0.12]}


def test_below_the_breakaway_torque_it_is_stuck_and_past_it_friction_is_the_kinetic_torque():
    for kinetic in (0.0, 0.05, F):
        tau_f = StaticFriction(F, kinetic, W)(U)
        stuck = np.abs(U) < F - 4 * W
        np.testing.assert_allclose(tau_f[stuck], U[stuck], atol=2e-3)  # it takes all of u
        moving = np.abs(U) > F + 6 * W
        np.testing.assert_allclose(tau_f[moving], kinetic * np.sign(U[moving]), atol=2e-3)
    delivered = U - StaticFriction(F, 0.0, W)(U)
    assert np.abs(delivered[np.abs(U) < F - 4 * W]).max() < 2e-3  # a dead band
    np.testing.assert_allclose(delivered[np.abs(U) > 1.5 * F], U[np.abs(U) > 1.5 * F], atol=2e-3)
    # Fc = F: friction stays, the delivered torque is the command less F
    b = StaticFriction(F, F, W)(np.array([0.5, -0.5]))
    np.testing.assert_allclose(b, [F, -F], atol=2e-3)


def test_it_is_odd_has_the_sign_of_the_command_and_never_exceeds_the_breakaway_torque():
    for friction in (
        StaticFriction(F, 0.0, W),
        StaticFriction(F, 0.07, W),
        StaticFriction(F, F, W),
    ):
        tau_f = friction(U)
        np.testing.assert_allclose(friction(-U), -tau_f, atol=1e-15)
        assert (tau_f * U >= -1e-15).all()
        assert np.abs(tau_f).max() <= F + 1e-9
    # what a command that brakes the motion can create: at most F |speed| of power
    speed = np.linspace(-3, 3, 61)
    power = StaticFriction(F, 0.07, W)(U)[:, None] * speed[None, :]  # tau_f * rate
    assert power.min() >= -F * 3 - 1e-9
    driving = (
        U[:, None] * speed[None, :]
    ) >= 0  # command and motion agree: friction only dissipates
    assert (power[driving] >= -1e-15).all()


def test_the_symbolic_and_the_numeric_torques_agree_and_per_motor_values_work():
    friction = StaticFriction(F, 0.05, W)
    np.testing.assert_allclose(symbolic(friction, U), friction(U), atol=1e-15)
    per = StaticFriction([0.2, 0.1], [0.0, 0.05], [0.01, 0.02])
    u = np.array([[0.1, 0.1], [0.5, 0.5], [-0.15, 0.03]])
    got = per(u)
    for j, (f, fc, w) in enumerate([(0.2, 0.0, 0.01), (0.1, 0.05, 0.02)]):
        np.testing.assert_allclose(got[:, j], StaticFriction(f, fc, w)(u[:, j]), atol=1e-15)


def test_it_is_differentiable_everywhere_with_the_derivatives_of_a_finite_difference():
    for friction in (StaticFriction(F, 0.0, W), StaticFriction(F, F, W), StaticFriction()):
        x = ca.MX.sym("x")
        p = constants(friction.params)
        y = friction.torque(x, p)
        d1 = ca.Function("d1", [x], [ca.gradient(y, x)])
        d2 = ca.Function("d2", [x], [ca.hessian(y, x)[0]])
        h = 1e-6
        for u0 in (-0.5, -0.2, -0.1, 0.0, 1e-4, 0.05, 0.19, 0.2, 0.21, 0.4):
            fd = (friction(np.array(u0 + h)) - friction(np.array(u0 - h))) / (2 * h)
            assert np.isfinite(float(d2(u0)))
            np.testing.assert_allclose(float(d1(u0)), float(fd), rtol=1e-5, atol=1e-8)
            fd2 = float(d1(u0 + h) - d1(u0 - h)) / (2 * h)
            np.testing.assert_allclose(float(d2(u0)), fd2, rtol=1e-3, atol=1e-5)


def test_it_is_a_parameter_set_that_round_trips_and_belongs_to_the_efficiency():
    friction = StaticFriction(F, 0.05, 0.03)
    assert list(friction.params) == ["breakaway", "kinetic", "width"]
    assert all(p.scope == "design" and p.unit == "N*m" for p in friction.params.values())
    assert all(p.bounds == (0.0, np.inf) for p in friction.params.values())
    assert StaticFriction(F).params["width"].value == pytest.approx(0.01)  # the default smoothing
    assert StaticFriction(F).params["kinetic"].value == 0.0  # and no kinetic torque
    efficiency = Efficiency(0.12, friction=friction)
    assert list(efficiency.params) == [
        "c1",
        "friction.breakaway",
        "friction.kinetic",
        "friction.width",
    ]
    assert efficiency.degree == 1
    data = efficiency.to_dict()
    assert data["friction"] == {"type": "static", "breakaway": F, "kinetic": 0.05, "width": 0.03}
    again = Efficiency.from_dict(data)
    np.testing.assert_allclose(again(U), efficiency(U), atol=1e-15)
    assert get("friction", "static") is StaticFriction
    assert as_friction(None) is None and as_friction(friction) is friction
    assert as_friction(data["friction"]).to_dict() == data["friction"]
    with pytest.raises(TypeError, match="StaticFriction"):
        as_friction(0.2)
    efficiency.params["friction.breakaway"].value = 0.3  # the Params are live to tune
    assert efficiency.friction.params["breakaway"].value == pytest.approx(0.3)
    np.testing.assert_allclose(
        np.array(
            ca.evalf(efficiency.delivered(ca.DM(U[:5]), constants(efficiency.params)))
        ).ravel(),
        efficiency(U[:5]),
        atol=1e-15,
    )


def test_friction_comes_before_the_efficiency_polynomial_on_the_motor_torque():
    friction = StaticFriction(F, 0.0, W)
    efficiency = Efficiency(0.5, 2.0, friction=friction)
    u = np.array([0.1, 0.3, -0.25])
    net = u - friction(u)
    np.testing.assert_allclose(efficiency(u), 0.5 * net + 2.0 * net**2, atol=1e-15)


def delivered_torque(arm, u):
    """The generalized force on the arm at rest for the motor torques ``u``, from its dynamics."""
    dynamics = vmc.compile_dynamics(arm)
    n = dynamics.n_u
    space = arm.model.space
    q, v = np.zeros(space.nq), np.zeros(space.nv)
    zero = np.array(dynamics.forward(q, v, np.zeros(n), dynamics.live_values(), 0.0)).ravel()
    got = np.array(dynamics.forward(q, v, np.full(n, u), dynamics.live_values(), 0.0)).ravel()
    return got - zero


@pytest.mark.parametrize(
    "build",
    [
        lambda e: planar.add_dynamics(planar.arm("three-link", efficiency=e)),  # Underactuated
        lambda e: helyx.add_dynamics(helyx.arm("145-145-145", efficiency=e)),  # TendonTransmission
        lambda e: adapt.add_dynamics(adapt.finger(efficiency=e)),  # Direct
    ],
    ids=["underactuated", "tendons", "direct"],
)
def test_every_transmission_with_an_efficiency_carries_the_friction_in_its_dynamics(build):
    stuck = StaticFriction(F, 0.0, W)
    plain = delivered_torque(build(Efficiency(1.0)), 0.1)
    held = delivered_torque(build(Efficiency(1.0, friction=stuck)), 0.1)
    free = delivered_torque(build(Efficiency(1.0, friction=stuck)), 0.5)
    assert np.abs(plain).max() > 1e-3
    assert np.abs(held).max() < 0.05 * np.abs(plain).max()  # 0.1 < F: the actuator is stuck
    np.testing.assert_allclose(free, delivered_torque(build(Efficiency(1.0)), 0.5), atol=1e-3)


def spring_to_goal(friction, stiffness=1.0, goal=1.0, output=()):
    """A damped inertia on one joint, a virtual spring to ``goal`` and a damper: the system."""
    chain = vmc.models.SerialChain(
        ["revolute"], axes=[[0, 1, 0]], points=[[0, 0, 0]], sites={"b": (1, [0.5, 0, 0])}
    )
    efficiency = Efficiency(1.0, friction=friction)
    robot = vmc.Mechanism("robot", model=chain, actuation=vmc.models.Direct(efficiency))
    q = robot.joint(0)
    robot.add("mass", vmc.PointMass(robot.point("b"), 1.0))
    robot.add("friction", vmc.LinearDamper(q, 1.5))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(q - vmc.Ref("goal", 1, [goal]), stiffness))
    ctrl.add("damper", vmc.LinearDamper(q, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl)


def final_error(system, output=(), T=30.0):
    log = vmc.sim.rollout(system, [0.0], T, 0.02, max_step=0.02, output=output)
    return 1.0 - float(log["q"][-1, 0])


def test_a_spring_stops_short_by_the_breakaway_over_the_stiffness_and_compensation_closes_it():
    stiffness = 1.0
    assert abs(final_error(spring_to_goal(None, stiffness))) < 1e-3  # without friction it arrives
    kept = StaticFriction(0.3, 0.3, 0.005)  # friction stays: the band is F / K
    assert final_error(spring_to_goal(kept, stiffness)) == pytest.approx(0.3 / stiffness, rel=0.05)
    comp = StaticFrictionCompensation.of(kept)
    assert abs(final_error(spring_to_goal(kept, stiffness), [comp])) < 0.02  # Fc = F: no band left
    half = StaticFrictionCompensation.of(kept, 0.5)
    assert final_error(spring_to_goal(kept, stiffness), [half]) == pytest.approx(0.15, rel=0.1)
    gone = StaticFriction(0.3, 0.1, 0.005)  # a band of F - Fc once it is compensated for Fc
    both = final_error(spring_to_goal(gone, stiffness), [StaticFrictionCompensation.of(gone)])
    assert both == pytest.approx(0.2, rel=0.15)


def test_the_compensation_is_smooth_off_unless_asked_and_the_same_for_numbers_and_expressions():
    stage = StaticFrictionCompensation([0.1, 0.05], 0.01, 0.8)
    u = np.array([[0.0, 0.0], [0.3, -0.2], [-0.001, 0.001]])
    out = np.stack([stage(row, None) for row in u])
    sym = np.stack([np.array(ca.evalf(stage.symbolic(ca.DM(row)))).ravel() for row in u])
    np.testing.assert_allclose(out, sym, atol=1e-15)
    np.testing.assert_allclose(out[0], 0.0, atol=1e-15)  # nothing to compensate at u = 0
    np.testing.assert_allclose(
        out[1], u[1] + 0.8 * np.array([0.1, 0.05]) * np.array([1, -1]), atol=2e-3
    )
    assert get("output", "static_friction_compensation") is StaticFrictionCompensation
    default = StaticFrictionCompensation(0.1)
    assert (default.width, default.fraction) == (0.01, 1.0)  # all of it, smoothed over 0.01
    made = StaticFrictionCompensation.of(StaticFriction(F, 0.05, 0.03), 0.4)
    assert (made.kinetic, made.width, made.fraction) == (0.05, 0.03, 0.4)
    assert StaticFrictionCompensation.of(StaticFriction(F, 0.05, 0.03)).fraction == 1.0
    system = spring_to_goal(None)
    assert vmc.VMCController(vmc.compile(system)).output == []  # nothing is on by default
    ref = adapt.output_stages() if hasattr(adapt, "output_stages") else []
    assert not any(isinstance(s, StaticFrictionCompensation) for s in ref)  # nor in a template


def test_a_plan_with_the_actuator_friction_and_its_compensation_is_the_simulation_step_for_step():
    friction = StaticFriction(0.3, 0.1, 0.02)
    comp = [StaticFrictionCompensation.of(friction, 0.7)]
    for output in ([], comp):
        system = spring_to_goal(friction)
        problem = opt.Problem(system, output=output)
        problem.add(opt.Shooting([0.0], 3.0, 11, v0=[0.0], substeps=5))
        plan = problem.solve()
        log = vmc.sim.rollout(system, [0.0], 3.0, 0.06, v0=[0.0], max_step=0.06, output=output)
        assert plan.converged and plan.violation < 1e-8
        np.testing.assert_allclose(plan.q[:-1], log["q"][::5], atol=1e-8)
        np.testing.assert_allclose(plan.u[:-1], log["u"][::5], atol=1e-8)
    # the friction changed the plan: without it the same plan is another one
    other = opt.Problem(spring_to_goal(None))
    other.add(opt.Shooting([0.0], 3.0, 11, v0=[0.0], substeps=5))
    assert np.abs(other.solve().q - plan.q).max() > 0.02


def test_a_plan_can_use_the_stage_only_if_it_has_a_symbolic_form():
    with pytest.raises(ValueError, match="TorqueLimit has no symbolic form"):
        opt.Problem(spring_to_goal(None), output=[vmc.control.TorqueLimit(1.0)])


K_STEP, GOAL_STEP = "ctrl.spring.stiffness", "ctrl.spring.goal"
H, PERIOD = 0.02, 5  # a control step [s] and the control steps per plan step


def mass_with(friction):
    """A mass behind an actuator with ``friction``, a force-capped spring to a goal whose
    stiffness and reference are bounded Params, and a damper."""
    efficiency = Efficiency(1.0, friction=friction)
    robot = vmc.Mechanism(
        "robot", model=vmc.models.JointSpace(1, unit="m"), actuation=Direct(efficiency)
    )
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, 1.0))
    robot.add("damper", vmc.LinearDamper(x, 1.0))
    ref = vmc.Ref("goal", 1, vmc.Param("goal", [0.0], bounds=(-1.0, 2.0), scope="stage"))
    stiffness = vmc.Param("stiffness", 4.0, bounds=(0.5, 50.0), scope="stage")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.TanhSpring(x - ref, stiffness, 5.0))
    ctrl.add("damper", vmc.LinearDamper(x, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def mpc_for(system, x, output=()):
    problem = opt.Problem(system, solver="ipopt-exact", output=output)
    problem.add(opt.Shooting([0.0], 1.0, 11, steps=[K_STEP, GOAL_STEP], substeps=PERIOD))
    problem.add(opt.Effort(0.01))
    problem.add(opt.Cost(x - vmc.Ref("target", 1, [1.0]), 1.0, name="reach"))
    problem.parameter("reach.target")
    return opt.MPC(problem)


def closed_loop(plant_system, mpc, output=(), seconds=2.0):
    """The plant behind ``plant_system``, under its controller, the plan applied every
    ``PERIOD`` steps; the plan's state one plan step on, and the plant's, then."""
    plant = vmc.sim.ModelPlant(plant_system.robot, q0=[0.0], max_step=H)
    controller = vmc.VMCController(vmc.compile(plant_system), output=list(output))
    controller.reset(0.0, plant.read())
    predicted, converged = {}, []
    for i in range(round(seconds / H)):
        if i % PERIOD == 0:  # the plan is applied before the step it starts at
            mpc.step(controller, plant.q, plant.v)
            predicted[i + PERIOD] = float(mpc.result.q[1, 0])
            converged.append(bool(mpc.result.converged))
        command = controller.step(plant.t, plant.read())
        plant.write(command)
        plant.advance(H)
        if i + 1 in predicted:
            predicted[i + 1] = (predicted[i + 1], float(plant.q[0]))
    return plant, [v for v in predicted.values() if isinstance(v, tuple)], converged


def test_mpc_plans_with_the_actuator_friction_and_its_compensation_in_the_loop():
    friction = StaticFriction(0.3, 0.3, 0.02)
    for output in ([], [StaticFrictionCompensation.of(friction, 0.8)]):
        system, x = mass_with(friction)
        plant, pairs, converged = closed_loop(system, mpc_for(system, x, output), output)
        assert all(converged) and len(pairs) >= 15
        error = max(abs(plan - real) for plan, real in pairs)
        assert error < 1e-4  # the plan knows the friction: it predicts the plant
        assert abs(plant.q[0] - 1.0) < 0.1
    # a plan that leaves the friction out predicts the plant badly
    system, x = mass_with(friction)
    blind_system, blind_x = mass_with(None)
    blind = mpc_for(blind_system, blind_x)
    plant, pairs, _ = closed_loop(system, blind)
    assert max(abs(plan - real) for plan, real in pairs) > 1e-3


def test_the_finger_and_hand_have_the_labs_friction_only_when_asked_for():
    assert adapt.finger().actuation.efficiency.friction is None  # nothing by default
    assert adapt.hand().actuation.efficiency.friction is None
    finger, hand = adapt.finger_friction(), adapt.hand_friction(0.05)
    assert finger.params["breakaway"].value == pytest.approx(adapt.FRICTION[0])
    assert hand.params["breakaway"].value == pytest.approx(adapt.HAND_FRICTION[0])
    assert finger.params["kinetic"].value == 0.0 and hand.params["width"].value == 0.05
    robot = adapt.finger(efficiency=Efficiency(1.0, friction=finger))
    assert robot.actuation.efficiency.friction is finger
