"""Components against forces recorded from the original lab implementation."""

from pathlib import Path

import numpy as np
import pytest

from helpers import component_function, evaluate
from virtualmodelcontrol.mechanisms import (
    GaussianSpring,
    Joint,
    LimitSpring,
    LinearDamper,
    LinearSpring,
    PolynomialSpring,
    Projection,
    SigmoidSpring,
    TanhDamper,
    TanhSpring,
)

DATA = np.load(Path(__file__).parents[1] / "data" / "components.npz")

X = Joint(slice(0, 3), unit="m")  # the point
Y = X - Joint(slice(3, 6), unit="m")  # deflection from the target (or obstacle)


def args(case):
    return {k.split(":")[1]: DATA[k] for k in DATA.files if k.startswith(f"{case}:")}


def cart(coord, a):
    return Projection(coord, a["n"].ravel())


SPRINGS = {
    "linear_scalar": lambda a: LinearSpring(Y, a["stiffness"]),
    "linear_axes": lambda a: LinearSpring(Y, a["stiffness"]),
    "linear_matrix": lambda a: LinearSpring(Y, a["stiffness"]),
    "tanh_scalar": lambda a: TanhSpring(Y, a["stiffness"], a["max_force"]),
    "tanh_axes": lambda a: TanhSpring(Y, a["stiffness"], a["max_force"]),
    "cart_linear_x": lambda a: LinearSpring(cart(Y, a), a["stiffness"]),
    "cart_linear_oblique": lambda a: LinearSpring(cart(Y, a), a["stiffness"]),
    "cart_tanh_z": lambda a: TanhSpring(cart(Y, a), a["stiffness"], a["max_force"]),
    "gaussian": lambda a: GaussianSpring(Y, a["strength"], a["sigma"]),
    "cart_gaussian_oblique": lambda a: GaussianSpring(cart(Y, a), a["strength"], a["sigma"]),
    "sigmoid_axes": lambda a: SigmoidSpring(Y, a["kmin"], a["kmax"], a["threshold"], a["alpha"]),
    "sigmoid_norm": lambda a: SigmoidSpring(
        Y, a["kmin"], a["kmax"], a["threshold"], a["alpha"], element_wise=False
    ),
    "polynomial_axes": lambda a: PolynomialSpring(Y, a["stiffness"], a["order"], a["dist_norm"]),
    "polynomial_norm": lambda a: PolynomialSpring(
        Y, a["stiffness"], a["order"], a["dist_norm"], element_wise=False
    ),
}

DAMPERS = {
    "damper_scalar": lambda a: LinearDamper(X, a["damping"]),
    "damper_axes": lambda a: LinearDamper(X, a["damping"]),
    "tanh_damper": lambda a: TanhDamper(X, a["damping"], a["max_force"]),
    "cart_damper_oblique": lambda a: LinearDamper(cart(X, a), a["damping"]),
    "cart_tanh_damper_y": lambda a: TanhDamper(cart(X, a), a["damping"], a["max_force"]),
}


@pytest.mark.parametrize("case", SPRINGS)
def test_springs_match_recorded_forces(case):
    fn = component_function(SPRINGS[case](args(case)), 6)
    Q = np.hstack([DATA["pos"], DATA["target"]])
    force_on_point = evaluate(fn, Q)[:, :3]
    np.testing.assert_allclose(force_on_point, DATA[case], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", DAMPERS)
def test_dampers_match_recorded_forces(case):
    fn = component_function(DAMPERS[case](args(case)), 3)
    force = evaluate(fn, DATA["pos"], DATA["vel"])
    np.testing.assert_allclose(force, DATA[case], rtol=1e-12, atol=1e-14)


def test_limit_spring_matches_recorded_torques():
    a = args("limit")
    spring = LimitSpring(Joint(0, unit="rad"), a["stiffness"], a["angle_lower"], a["angle_upper"])
    torque = evaluate(component_function(spring, 1), DATA["angle"][:, None])
    np.testing.assert_allclose(torque, DATA["limit"], rtol=1e-12, atol=1e-14)
