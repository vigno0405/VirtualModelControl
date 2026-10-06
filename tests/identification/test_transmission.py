from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from virtualmodelcontrol.identification import fit_transmission
from virtualmodelcontrol.robots import adapt

DATA = np.load(Path(__file__).parents[1] / "data" / "transmission.npz")  # (motor, joint) in degrees
R = adapt.MOTOR_RADIUS
FINGERS = ("index", "middle", "ring", "pinky")


def pulley(name):  # joint per motor = r_pulley / r_motor
    return R * fit_transmission(*DATA[name].T)


def cable(name):  # joint per motor = r_motor / c
    return R / fit_transmission(*DATA[name].T)


def test_the_finger_constants_come_out_of_its_recorded_sweeps():
    assert pulley("finger_mcp") == pytest.approx(adapt.FINGER_PULLEY_RADIUS, abs=5.01e-6)
    assert cable("finger_pip") == pytest.approx(adapt.PIP_TRANSMISSION, abs=5.01e-6)


@pytest.mark.parametrize("finger", FINGERS)
def test_the_hand_constants_come_out_of_its_recorded_sweeps(finger):
    table = adapt.HAND_FINGER_TRANSMISSIONS[finger]
    assert pulley(f"hand_{finger}_MCP") == pytest.approx(table["r_pulley"], abs=5.01e-6)
    assert cable(f"hand_{finger}_PIP") == pytest.approx(table["c_param"], abs=5.01e-6)


@pytest.mark.parametrize("joint", ("MCP", "IP"))
def test_the_thumb_constants_come_out_of_its_recorded_sweeps(joint):
    table = adapt.HAND_THUMB_TRANSMISSION
    assert cable(f"hand_thumb_{joint}") == pytest.approx(table[f"{joint}_c"], abs=5.01e-6)


@pytest.mark.parametrize("name", sorted(DATA.files))
def test_the_lab_fit_by_minimizing_the_squared_error_agrees(name):
    motor, joint = np.radians(DATA[name]).T
    lab = minimize(lambda k: np.mean((k * joint - motor) ** 2), [1.0], method="L-BFGS-B").x[0]
    assert fit_transmission(motor, joint) == pytest.approx(1 / lab, rel=1e-5)


def test_the_noise_is_on_the_motor_reading():
    rng = np.random.default_rng(0)
    joint = np.linspace(0.1, 1.2, 4000)  # set exactly
    motor = 2.5 * joint + rng.normal(0, 0.5, joint.size)  # read with noise
    assert fit_transmission(motor, joint) == pytest.approx(1 / 2.5, rel=3e-2)


def test_the_ratio_keeps_its_sign_and_does_not_depend_on_the_unit():
    joint = np.array([0.0, 15.0, 30.0, 45.0])
    motor = -2.0 * joint
    assert fit_transmission(motor, joint) == pytest.approx(-0.5, rel=1e-12)
    assert fit_transmission(np.radians(motor), np.radians(joint)) == pytest.approx(-0.5, rel=1e-12)
