import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.core import constants
from virtualmodelcontrol.core.registry import get
from virtualmodelcontrol.models import Efficiency
from virtualmodelcontrol.models.efficiency import as_efficiency, coefficients


def test_a_polynomial_of_the_commanded_torque_motor_by_motor():
    shared = Efficiency(0.9, -0.5, 2.0)
    u = np.array([[0.1, -0.2, 0.3], [0.0, 0.05, -0.4]])
    np.testing.assert_allclose(shared(u), 0.9 * u - 0.5 * u**2 + 2.0 * u**3)
    per_motor = Efficiency((0.8, 0.4, 1.0), (0.0, 0.1, -0.2))
    np.testing.assert_allclose(per_motor(u), [0.8, 0.4, 1.0] * u + [0.0, 0.1, -0.2] * u**2)
    assert shared.degree == 3 and per_motor.degree == 2
    np.testing.assert_allclose(Efficiency(1.0)(u), u)  # the default delivers the command
    np.testing.assert_allclose(Efficiency(0.12)(u), 0.12 * u)  # a constant efficiency


def test_the_symbolic_and_numeric_torques_agree():
    efficiency = Efficiency((0.8, 0.4, 1.0), 0.3, (-1.0, 0.0, 2.0))
    u = np.array([0.2, -0.1, 0.05])
    symbolic = efficiency.delivered(ca.DM(u), constants(efficiency.params))
    np.testing.assert_allclose(np.array(ca.evalf(symbolic)).ravel(), efficiency(u))


def test_the_coefficients_are_params_to_tune():
    efficiency = Efficiency(0.9, -0.5)
    assert list(efficiency.params) == ["c1", "c2"]
    c1, c2 = efficiency.params["c1"], efficiency.params["c2"]
    assert c1.scope == c2.scope == "design"
    assert c1.unit == "" and c2.unit == "(N*m)^-1"
    assert c1.bounds == (0.0, 1.0) and c2.bounds == (-np.inf, np.inf)
    c1.value = 0.5
    np.testing.assert_allclose(efficiency(0.2), 0.5 * 0.2 - 0.5 * 0.04)
    with pytest.raises(ValueError, match="c1"):
        Efficiency()


def test_dict_round_trip_and_conversions():
    efficiency = Efficiency((0.8, 0.4), -0.1)
    data = efficiency.to_dict()
    assert data == {"type": "polynomial", "coefficients": [[0.8, 0.4], -0.1]}
    assert Efficiency.from_dict(data).to_dict() == data
    assert get("efficiency", "polynomial") is Efficiency
    assert as_efficiency(efficiency) is efficiency
    assert as_efficiency(data).to_dict() == data
    assert as_efficiency(0.12).to_dict() == Efficiency(0.12).to_dict()
    p = {"seg1.r": 1.0, "efficiency.c1": 0.5, "efficiency.c2": 0.1}
    assert coefficients(p) == {"c1": 0.5, "c2": 0.1}
