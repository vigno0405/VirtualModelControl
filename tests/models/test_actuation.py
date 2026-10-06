import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import Euclidean, Product, Quaternion, constants
from virtualmodelcontrol.models import (
    Direct,
    Efficiency,
    Passive,
    StackedActuation,
    TendonTransmission,
    from_dict,
)
from virtualmodelcontrol.robots import helyx, turtle

rng = np.random.default_rng(5)
TENDONS = TendonTransmission(list(helyx.TENDON_ANGLES), helyx.SPOOL_RADIUS)
P = constants(TENDONS.params)


def num(x):
    return np.array(ca.evalf(x)).squeeze()


def test_motor_angles_round_trip():
    for _ in range(10):
        theta = rng.uniform(-3, 3, 9)
        q = TENDONS.config_from_motors(ca.DM(theta), P)
        np.testing.assert_allclose(num(TENDONS.motor_angles(q, P)), theta, atol=1e-12)


def test_jacobian_is_the_derivative_of_motor_angles():
    q = ca.SX.sym("q", 9)
    J = ca.Function("J", [q], [ca.jacobian(TENDONS.motor_angles(q, P), q)])
    np.testing.assert_allclose(num(J(rng.normal(size=9))), num(TENDONS.jacobian(P)), atol=1e-12)


def test_pulling_shortens_tendons():
    # Elongating every segment lengthens all tendons: the motors must release (θ < 0).
    assert np.all(num(TENDONS.motor_angles(ca.DM([0, 0, 0.01] * 3), P)) < 0)
    # Bending segment 1 towards its first tendon (angle 0) shortens that tendon: θ > 0.
    theta = num(TENDONS.motor_angles(ca.DM([0.01, 0, 0] + [0] * 6), P))
    assert theta[0] > 0 and theta[0] == theta.max()
    lengths = num(TENDONS.tendon_lengths(ca.DM([0.01, 0, 0] + [0] * 6), P))
    np.testing.assert_allclose(lengths, -helyx.SPOOL_RADIUS * theta, atol=1e-15)


def test_allocation_realizes_the_generalized_force():
    tau = rng.normal(size=9)
    u = TENDONS.allocate(ca.DM(tau), None, P)
    np.testing.assert_allclose(num(ca.mtimes(TENDONS.jacobian(P).T, u)), tau, atol=1e-12)


def test_dict_round_trip_and_direct_drive():
    copy = from_dict(TENDONS.to_dict(), kind="actuation")
    assert copy.to_dict() == TENDONS.to_dict()
    direct = from_dict({"type": "direct"}, kind="actuation")
    assert isinstance(direct, Direct)
    x = ca.DM([1.0, 2.0])
    assert direct.allocate(x, None, {}) is x and direct.config_from_motors(x, {}) is x


def test_direct_drive_delivers_what_its_efficiency_gives():
    u = ca.DM([1.0, -2.0])
    assert float(Direct().params["efficiency.c1"].value) == 1.0  # the default
    cases = (
        (1.0, [1.0, -2.0]),
        (0.5, [0.5, -1.0]),
        ([0.8, 0.4], [0.8, -0.8]),
        (Efficiency(0.5, 0.1), [0.6, -0.6]),  # 0.5 u + 0.1 u²
    )
    for efficiency, delivered in cases:
        direct = Direct(efficiency)
        p = constants(direct.params)
        np.testing.assert_allclose(num(direct.generalized_force(u, None, p)), delivered)
        assert direct.allocate(u, None, p) is u  # commands are never divided by the efficiency
        assert from_dict(direct.to_dict(), kind="actuation").to_dict() == direct.to_dict()
    # dicts written by 0.2.0
    assert from_dict({"type": "direct"}, kind="actuation").to_dict() == Direct().to_dict()
    old = from_dict({"type": "direct", "efficiency": [0.8, 0.4]}, kind="actuation")
    assert old.to_dict() == Direct([0.8, 0.4]).to_dict()


def test_tendons_deliver_through_their_efficiency():
    u = rng.normal(size=9) * 0.05
    B = num(TENDONS.jacobian(P).T)
    np.testing.assert_allclose(num(TENDONS.generalized_force(ca.DM(u), None, P)), B @ u)
    lossy = TendonTransmission(
        list(helyx.TENDON_ANGLES), helyx.SPOOL_RADIUS, Efficiency(0.9, -1.0, 3.0)
    )
    delivered = 0.9 * u - u**2 + 3.0 * u**3
    p = constants(lossy.params)
    np.testing.assert_allclose(num(lossy.generalized_force(ca.DM(u), None, p)), B @ delivered)
    tau = rng.normal(size=9)
    np.testing.assert_allclose(
        num(lossy.allocate(ca.DM(tau), None, p)), num(TENDONS.allocate(ca.DM(tau), None, P))
    )
    assert from_dict(lossy.to_dict(), kind="actuation").to_dict() == lossy.to_dict()
    old = dict(TENDONS.to_dict(), efficiency=0.12)  # a dict written by 0.2.0
    assert from_dict(old, kind="actuation").to_dict()["efficiency"] == Efficiency(0.12).to_dict()


def test_a_passive_part_has_no_motors_and_no_force():
    passive = Passive(6)
    assert passive.motor_sizes(None) == (0, 0) and len(passive.params) == 0
    for empty in (
        passive.motor_angles(ca.DM.zeros(7), {}),
        passive.motor_rates(ca.DM.zeros(7), ca.DM.zeros(6), {}),
        passive.allocate(ca.DM.ones(6), ca.DM.zeros(7), {}),
    ):
        assert empty.shape == (0, 1)
    tau = passive.generalized_force(ca.DM.zeros(0, 1), ca.DM.zeros(7), {})
    assert tau.shape == (6, 1) and float(ca.norm_1(tau)) == 0.0
    for method, args in (
        ("config_from_motors", (None, {})),
        ("velocity_from_motors", (None, None, {})),
    ):
        with pytest.raises(NotImplementedError, match="passive"):
            getattr(passive, method)(*args)
    copy = from_dict(passive.to_dict(), kind="actuation")
    assert isinstance(copy, Passive) and copy.nv == 6 and copy.to_dict() == passive.to_dict()


def test_stacked_with_motors_only_the_driven_parts_are_in_the_motor_vector():
    free = Product(Euclidean(3), Quaternion())  # 7 coordinates, 6 velocities
    parts = [
        (free, Passive(6), "body"),
        (Euclidean(1), Direct(0.5), "a"),
        (Euclidean(1), Direct(), "b"),
    ]
    stack = StackedActuation(parts)
    space = Product(free, Euclidean(1), Euclidean(1))
    assert stack.motor_sizes(space) == (2, 2)
    p = constants(stack.params)
    q = ca.DM(np.arange(9.0))
    v = ca.DM(np.arange(8.0) * 10)
    np.testing.assert_array_equal(np.array(stack.motor_angles(q, p)).ravel(), [7.0, 8.0])
    np.testing.assert_array_equal(np.array(stack.motor_rates(q, v, p)).ravel(), [60.0, 70.0])
    tau = stack.generalized_force(ca.DM([2.0, 3.0]), q, p)
    np.testing.assert_allclose(np.array(tau).ravel(), [0, 0, 0, 0, 0, 0, 1.0, 3.0])
    np.testing.assert_allclose(
        np.array(stack.allocate(ca.DM(np.arange(8.0)), q, p)).ravel(), [6, 7]
    )


def test_a_controller_cannot_be_compiled_against_a_robot_with_a_passive_part():
    robot = turtle.crawler()
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(robot.joint(7), 1.0))
    with pytest.raises(NotImplementedError, match="passive"):
        vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
