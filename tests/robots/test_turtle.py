"""The turtle template against torques published by the original controller node."""

from pathlib import Path

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import turtle

DATA = np.load(Path(__file__).parents[1] / "data" / "turtle.npz")


def test_flywheel_controller_matches_the_recorded_node():
    p = {k[6:]: float(DATA[k]) for k in DATA.files if k.startswith("param_")}
    robot = turtle.robot()
    ctrl = turtle.controller(
        robot,
        stiffness=p["K"],
        damping=p["C"],
        inertia=p["J_v"],
        flywheel_damping=p["b_v"],
        speed=p["omega_bar"],
        phase=p["delta"],
        ramp_time=p["omega_ramp_time"],
        torque_bias=p["torque_bias"],
    )
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    signs = np.array(turtle.MOTOR_SIGNS)

    def measure(k):
        return vmc.Signals(
            DATA["t"][k],
            motor_position=signs * np.radians(DATA["position_deg"][k]),
            motor_velocity=signs * np.radians(DATA["velocity_deg"][k]),
        )

    first = measure(0)
    controller.reset(DATA["t"][0], first, z0=turtle.initial_state(first))
    for k in range(1, len(DATA["t"])):
        u = controller.step(DATA["t"][k], measure(k))["motor_torque"]
        np.testing.assert_allclose(signs * u, DATA["torque"][k], rtol=1e-9, atol=1e-12)


def test_unknown_parameter_is_reported():
    try:
        turtle.controller(turtle.robot(), stifness=2.0)
    except TypeError as e:
        assert "stifness" in str(e)
    else:
        raise AssertionError("a misspelt parameter must raise")
