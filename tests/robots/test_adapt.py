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
