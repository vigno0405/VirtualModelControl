"""A virtual mechanism with its own kinematics: a cart on a rail, driven by a state, a
reference or a function of time."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core.params import Binding, ParamSet
from virtualmodelcontrol.mechanisms import Context
from virtualmodelcontrol.models import SerialChain

WAYPOINTS = np.array(
    [[0.0, 0.0, 0.1], [0.1, 0.0, 0.25], [0.2, 0.0, 0.3], [0.3, 0.0, 0.25], [0.4, 0.0, 0.1]]
)
K, DAMPING, CART = 150.0, 3.0, 0.4  # [N/m], [N s/m], [kg]
Q, V = np.array([0.15, 0.22]), np.array([0.3, -0.2])


def rail():
    return SerialChain(
        [("rail", WAYPOINTS)], axes=[None], points=[[0, 0, 0]], sites={"cart": (1, WAYPOINTS[0])}
    )


def robot():
    """A mass that moves in x and z."""
    chain = SerialChain(
        ["prismatic", "prismatic"],
        axes=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        points=[[0, 0, 0]] * 2,
        sites={"tip": (2, [0.0, 0.0, 0.0])},
    )
    arm = vmc.Mechanism("robot", model=chain)
    arm.add("mass", vmc.PointMass(arm.point("tip"), 1.0))
    return arm


def tie_to(arm, ctrl, cart, damping=0.0):
    ctrl.add("tie", vmc.LinearSpring(arm.point("tip") - cart, K))
    if damping:
        ctrl.add("tie_damper", vmc.LinearDamper(arm.point("tip") - cart, damping))


def law(compiled, q=Q, v=V, z=(), t=0.0):
    u, zdot = compiled.law(q, v, np.asarray(z, dtype=float), compiled.live_values(), t)
    return np.array(u).ravel(), np.array(zdot).ravel()


def test_a_virtual_cart_on_a_rail_moves_as_a_mass_on_the_curve():
    arm = robot()
    ctrl = vmc.Mechanism("ctrl")
    s = ctrl.add_state("s")
    cart = vmc.FramePoint(rail(), "cart", q=s)
    ctrl.add("mass", vmc.Inertance(cart, CART))
    tie_to(arm, ctrl, cart)
    compiled = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
    s0, sdot = 0.37, 0.8
    u, zdot = law(compiled, z=[s0, sdot])
    # the same cart by hand, from the rail's own kinematics: m (J'J s'' + J'H s'^2) = J'f
    kin = vmc.Kinematics(rail())
    x, J = kin.position([s0], "cart"), kin.jacobian([s0], "cart")
    H = kin.hessian([s0], "cart")[:, 0, 0]
    tip = np.array([Q[0], 0.0, Q[1]])
    f = K * (tip - x)  # the spring pulls the cart to the tip
    accel = (J[:, 0] @ f / CART - J[:, 0] @ H * sdot**2) / (J[:, 0] @ J[:, 0])
    np.testing.assert_allclose(zdot, [sdot, accel], rtol=1e-10)
    np.testing.assert_allclose(u, -f[[0, 2]], rtol=1e-12)  # and the tip feels the opposite
    assert abs(J[:, 0] @ H) > 1e-3 and abs(accel) > 1.0  # the velocity term and the force count


def test_a_virtual_cart_on_a_rail_closes_the_energy_balance_of_the_run():
    arm = robot()
    arm.add("friction", vmc.LinearDamper(arm.point("tip"), 1.0))
    ctrl = vmc.Mechanism("ctrl")
    s = ctrl.add_state("s", initial=0.2)
    cart = vmc.FramePoint(rail(), "cart", q=s)
    ctrl.add("mass", vmc.Inertance(cart, CART))
    tie_to(arm, ctrl, cart, DAMPING)
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.1])
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1e-3), T=2.0, record="energy")
    b = vmc.sim.energy_balance(log)
    assert np.ptp(log.arrays()["energy/kinetic"]) > 0.01  # the cart and the tip moved
    np.testing.assert_allclose(b["margin"], b["energy"] + b["dissipated"] - b["injected"])
    assert b["margin"].min() > -1e-4 * (b["energy"].max() + 1e-9) and b["dissipated"][-1] > 0.01


def test_the_waypoints_of_a_virtual_rail_are_params_of_the_system():
    arm = robot()
    ctrl = vmc.Mechanism("ctrl")
    cart = vmc.FramePoint(rail(), "cart", q=vmc.Ref("s", 1, value=[0.3], unit=""))
    tie_to(arm, ctrl, cart)
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    name = "ctrl.tie.model.j1.waypoints"
    assert system.params[name].shape == (5, 3)
    folded = vmc.compile(system)
    live = vmc.compile(system, runtime=[name])
    assert name not in folded.live and name in live.live
    u_folded, u_live = law(folded)[0], law(live)[0]
    np.testing.assert_allclose(u_live, u_folded, rtol=1e-12)
    # a rail twice as big puts the cart elsewhere: the live Param is read at every step
    p = live.live_values()
    slot = live.live_slices()[name]
    p[slot] = 2.0 * p[slot]
    u, _ = live.law(Q, V, np.zeros(0), p, 0.0)
    assert np.abs(np.array(u).ravel() - u_folded).max() > 1.0


def test_a_cart_driven_by_a_reference_is_held_where_the_reference_says_and_it_can_move():
    arm = robot()
    ctrl = vmc.Mechanism("ctrl")
    cart = vmc.FramePoint(rail(), "cart", q=vmc.Ref("s", 1, value=[0.3], unit=""))
    tie_to(arm, ctrl, cart)
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    kin = vmc.Kinematics(rail())
    tip = np.array([Q[0], 0.0, Q[1]])
    meas = vmc.Signals(0.0, motor_position=Q, motor_velocity=V)
    for s in (0.3, 0.8):
        controller.set({"ctrl.tie.s": [s]})
        force = -K * (tip - kin.position([s], "cart"))
        torque = controller.step(0.0, meas)["motor_torque"]
        np.testing.assert_allclose(torque, force[[0, 2]], rtol=1e-12)


def test_a_goal_that_is_a_function_of_time_pulls_and_drags_with_its_own_velocity():
    arm = robot()
    ctrl = vmc.Mechanism("ctrl")
    omega = 2.0

    def circle(t):
        return ca.vertcat(0.1 * ca.cos(omega * t), 0.0, 0.2 + 0.05 * ca.sin(3 * t))

    goal = vmc.Custom(circle, [vmc.Time()], dim=3, unit="m")
    tie_to(arm, ctrl, goal, DAMPING)
    compiled = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
    for t in (0.0, 0.4, 1.3):
        place = np.array([0.1 * np.cos(omega * t), 0.0, 0.2 + 0.05 * np.sin(3 * t)])
        rate = np.array([-0.1 * omega * np.sin(omega * t), 0.0, 0.15 * np.cos(3 * t)])
        tip, tip_rate = np.array([Q[0], 0.0, Q[1]]), np.array([V[0], 0.0, V[1]])
        want = -K * (tip - place) - DAMPING * (tip_rate - rate)  # the damper sees the goal move
        np.testing.assert_allclose(law(compiled, t=t)[0], want[[0, 2]], rtol=1e-12)
        assert abs(rate).max() > 0.05


def test_time_needs_a_context_that_has_it_and_a_virtual_model_the_right_coordinates():
    ctx = Context(None, Binding(ParamSet(), []))
    with pytest.raises(ValueError, match="no time"):
        vmc.Time().value(ctx)
    assert vmc.Time().dim == 1 and vmc.Time().unit == "s"
    with pytest.raises(ValueError, match="1 coordinates, q has 2"):
        vmc.FramePoint(rail(), "cart", q=vmc.Ref("x", 2))


def test_a_joint_of_the_robot_follows_a_function_of_time_through_a_stiff_spring():
    omega, amplitude = 2.0, 0.3
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(1, unit="rad"))
    robot.add("inertia", vmc.Inertance(robot.joint(0), 1.0))

    def wanted(t):
        return amplitude * ca.sin(omega * t)

    goal = vmc.Custom(wanted, [vmc.Time()], dim=1, unit="rad")
    robot.add("servo", vmc.LinearSpring(robot.joint(0) - goal, 1e4))
    robot.add("servo_damper", vmc.LinearDamper(robot.joint(0) - goal, 100.0))
    plant = vmc.sim.ModelPlant(robot, max_step=1e-4)
    error = []
    for _ in range(60):
        plant.advance(0.05)
        if plant.t > 1.0:
            error.append(plant.q[0] - amplitude * np.sin(omega * plant.t))
    # the lag is the mass's, m w^2 / k = 0.04 %; a damper blind to the goal's speed would add
    # c w / k = 2 %
    assert np.abs(error).max() < 0.002 * amplitude
    assert abs(plant.v[0]) > 0.1  # it moves, and does not sit at the start
