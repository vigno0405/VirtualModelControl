import casadi as ca
import numpy as np

from virtualmodelcontrol.core import constants
from virtualmodelcontrol.models import Direct, Efficiency, TendonTransmission, from_dict
from virtualmodelcontrol.robots import helyx

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
