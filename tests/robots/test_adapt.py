"""The finger against values recorded from the original finger kinematics."""

from pathlib import Path

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import adapt

DATA = np.load(Path(__file__).parents[1] / "data" / "finger.npz")


def test_positions_match_the_recorded_finger():
    model = adapt.finger().model
    for i, q in enumerate(DATA["q"]):
        for site in ("pip", "dip", "tip", "mcp_cog", "pip_cog", "dip_cog"):
            np.testing.assert_allclose(evaluate_frame(model, q, site)[1], DATA[site][i], atol=1e-15)


def test_joint_angles_coordinate_and_limits():
    robot = adapt.finger()
    angles = adapt.joint_angles(robot)
    params = vmc.ParamSet()
    params.merge(robot.params)
    q = ca.SX.sym("q", 2)
    ctx = vmc.mechanisms.Context(q, vmc.core.Binding(params))
    f = ca.Function("f", [q], [ctx.value(angles)])
    for i, x in enumerate(DATA["q"]):
        np.testing.assert_allclose(np.array(f(x)).ravel(), DATA["joints"][i], atol=1e-15)
    spring = adapt.joint_limit_spring(robot)
    assert spring.lower.value.tolist() == [0.0, 0.0, 0.0] and spring.upper.value[0] == np.pi / 2


def test_finger_controller_pushes_back_from_the_limits():
    robot, ctrl = adapt.finger(), vmc.Mechanism("ctrl")
    ctrl.add("limits", adapt.joint_limit_spring(robot))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    inside = controller.step(
        0.0, vmc.Signals(0.0, motor_position=[0.3, 0.3], motor_velocity=[0, 0])
    )
    below = controller.step(
        0.0, vmc.Signals(0.0, motor_position=[-0.3, -0.3], motor_velocity=[0, 0])
    )
    assert np.all(inside["motor_torque"] == 0.0)
    assert np.all(below["motor_torque"] > 0.0)  # flex back into the range
    assert constants(robot.params)["gravity"].shape == (3, 1)


def test_add_dynamics_gives_the_simulator_the_gravity_the_controller_compensates():
    q0, drift = np.array([0.6, 0.4]), []
    for efficiency in (1.0, adapt.MOTOR_EFFICIENCY):
        finger = adapt.add_dynamics(adapt.finger(efficiency=efficiency), damping=0.01)
        assert {"gravity", "damping"} <= set(finger.components)
        ctrl = vmc.Mechanism("ctrl")
        ctrl.add("gravity", vmc.GravityCompensation(finger))
        controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl)))
        plant = vmc.sim.ModelPlant(finger, q0=q0)
        vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 500), T=0.5)
        drift.append(np.abs(plant.q - q0).max())
    assert drift[0] < 1e-9  # efficiency 1 (the default): the compensated finger stays
    assert drift[1] > 1e-3  # the motors deliver η of the compensation: the finger sags


def test_the_finger_and_the_hand_receive_efficiency_times_the_command():
    assert adapt.finger().params["efficiency.c1"].value == 1.0  # the default efficiency
    assert adapt.hand().params["efficiency.c1"].value == 1.0
    cases = ((adapt.finger, adapt.MOTOR_EFFICIENCY), (adapt.hand, adapt.HAND_MOTOR_EFFICIENCY))
    for template, eta in cases:
        dyn = vmc.compile_dynamics(adapt.add_dynamics(template(efficiency=eta)))
        p, n = dyn.live_values(), len(eta)
        q, u = np.full(n, 0.2), np.linspace(-0.5, 0.5, n)
        r = [np.array(dyn.residual(q, 0 * q, 0 * q, x, p, 0.0)).ravel() for x in (0 * u, u)]
        np.testing.assert_allclose(r[0] - r[1], np.asarray(eta) * u, rtol=1e-12, atol=1e-15)
