"""The turtle template against torques published by the original controller node."""

from pathlib import Path

import numpy as np
import pytest

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


def hand_built(robot, depth, peak, steer, limit=None, stiffness=1.0, phase=0.0, **p):
    """The flywheel with PhaseSprings, as the crawling tutorial writes it."""
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, p.get("inertia", 0.1)))
    ctrl.add("drive", vmc.SpeedRegulator(phi, 0.1, p.get("speed", 0.1), 3.0))
    behind = vmc.Ref("phase", 1, value=phase, unit="rad")
    for name, i, phase, side in (("left", 0, phi, 1.0), ("right", 1, phi - behind, -1.0)):
        e = robot.joint(i) - phase
        spring = vmc.PhaseSpring(
            vmc.Stack(e, phase),
            stiffness,
            depth=depth,
            peak=peak,
            steer=steer,
            side=side,
            limit=limit,
        )
        ctrl.add(f"spring_{name}", spring)
        ctrl.add(f"damper_{name}", vmc.LinearDamper(e, 0.0001))
    ctrl.add("bias", vmc.ForceSource(vmc.Joint([0, 1], unit="rad"), [0.0, 0.0]))
    return ctrl


@pytest.mark.parametrize("limit", [None, 0.4])
def test_the_phase_options_give_the_springs_of_the_paper(limit):
    robot = turtle.robot()
    options = dict(depth=0.6, peak=0.7, steer=0.3, limit=limit, stiffness=2.5, phase=0.5)
    templated = turtle.controller(robot, **options)
    by_hand = hand_built(robot, **options)
    assert all(
        isinstance(templated.components[f"spring_{c}"], vmc.PhaseSpring) for c in turtle.CRANKS
    )
    torques = []
    for ctrl in (templated, by_hand):
        controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
        controller.reset(
            0.0, vmc.Signals(0.0, motor_position=np.zeros(2), motor_velocity=np.zeros(2))
        )
        rng = np.random.default_rng(3)
        out = []
        for k in range(40):
            meas = vmc.Signals(
                0.002 * k,
                motor_position=rng.uniform(-1, 1, 2),
                motor_velocity=rng.uniform(-2, 2, 2),
            )
            out.append(controller.step(0.002 * k, meas)["motor_torque"])
        torques.append(np.array(out))
    np.testing.assert_allclose(torques[0], torques[1], rtol=1e-9, atol=1e-12)


def test_without_a_phase_option_the_springs_stay_plain_and_a_wrong_name_lists_the_known():
    robot = turtle.robot()
    plain = turtle.controller(robot)
    assert all(isinstance(plain.components[f"spring_{c}"], vmc.LinearSpring) for c in turtle.CRANKS)
    asked = turtle.controller(robot, depth=0.0)  # any option, even one at its plain value
    assert isinstance(asked.components["spring_left"], vmc.PhaseSpring)
    with pytest.raises(TypeError, match="depth"):
        turtle.controller(robot, deph=0.5)
