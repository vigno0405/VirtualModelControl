"""The static friction of the motors, from a slow rise of the command and from static torques."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.identification import fit_efficiency, fit_params
from virtualmodelcontrol.models import Direct, Efficiency, JointSpace, StaticFriction

MASS, K, D = np.array([0.5, 1.0]), np.array([40.0, 25.0]), np.array([2.0, 1.5])
BREAKAWAY, KINETIC, WIDTH = np.array([0.30, 0.18]), np.array([0.10, 0.05]), 0.002
DT = 1 / 330


def robot(friction=None):
    """Two masses with a spring and a damper each, driven by motors that may have friction."""
    efficiency = Efficiency(1.0, friction=friction)
    arm = vmc.Mechanism("robot", model=JointSpace(2, unit="m"), actuation=Direct(efficiency))
    x = arm.joint(slice(0, 2))
    arm.add("mass", vmc.Inertance(x, MASS))
    arm.add("spring", vmc.LinearSpring(x, K))
    arm.add("damper", vmc.LinearDamper(x, D))
    return arm


def ramps(top=1.0, rate=0.1):
    """Each motor in turn: the command rises slowly from rest to ``top`` and back down."""
    rise = np.arange(0.0, top, rate * DT)
    one = np.concatenate([rise, rise[::-1], np.zeros(int(0.5 / DT))])
    rest = np.zeros(int(0.5 / DT))
    first = np.stack(
        [np.concatenate([rest, one, np.zeros(len(one))]), np.zeros(len(one) * 2 + len(rest))]
    )
    second = first[::-1]
    return np.concatenate([first, second[:, len(rest) :]], axis=1).T


def run_of(friction):
    u = ramps()
    system = vmc.VirtualMechanismSystem(robot(friction), vmc.Mechanism("none"))
    log = vmc.sim.rollout(system, [0.0, 0.0], len(u) * DT, DT, u=u, max_step=DT)
    return {key: np.asarray(log[key]) for key in ("t", "q", "v", "u")}


def test_the_friction_params_are_fitted_with_the_stiffness_and_damping_to_ramps_of_the_command():
    log = run_of(StaticFriction(BREAKAWAY, KINETIC, WIDTH))
    guess = StaticFriction([0.2, 0.2], [0.05, 0.05], WIDTH)  # the first command that moved a motor
    arm = robot(guess)
    names = ["efficiency.friction.breakaway", "efficiency.friction.kinetic"]
    fit = fit_params(arm, [*names, "spring.stiffness", "damper.damping"], [log], smoothing=11)
    np.testing.assert_allclose(fit.values[names[0]], BREAKAWAY, rtol=0.02)
    np.testing.assert_allclose(fit.values[names[1]], KINETIC, rtol=0.05)
    np.testing.assert_allclose(fit.values["spring.stiffness"], K, rtol=0.01)
    np.testing.assert_allclose(fit.values["damper.damping"], D, rtol=0.1)
    assert fit.rms < 1e-3
    # a robot with the fitted friction repeats the run
    for name in names:
        arm.params[name].value = fit.values[name]
    again = run_of(arm.actuation.efficiency.friction)
    assert np.abs(again["q"] - log["q"]).max() < 0.02 * np.abs(log["q"]).max()


def test_a_robot_without_friction_is_fitted_to_no_dead_band():
    log = run_of(None)
    arm = robot(StaticFriction([0.2, 0.2], [0.05, 0.05], WIDTH))
    names = ["efficiency.friction.breakaway", "efficiency.friction.kinetic"]
    fit = fit_params(arm, names, [log], smoothing=11)
    assert fit.values[names[0]].max() < 0.02 and fit.values[names[1]].max() < 0.02


def static_points(friction, noise=0.0, seed=0, scale=(1.0, 1.0), coefficients=((0.8, 0.6),)):
    """Commanded and delivered torques of two motors (each over its own range of commands)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, 81)[:, None] * np.array(scale)[None, :]
    delivered = Efficiency(*coefficients, friction=friction)(u)
    return u, delivered + noise * rng.normal(size=u.shape)


def test_the_friction_of_the_efficiency_is_fitted_to_static_torques():
    truth = StaticFriction([0.25, 0.15], [0.05, 0.0], 0.02)
    u, tau = static_points(truth)
    fit = fit_efficiency(u, tau, friction=True, width=0.02)
    np.testing.assert_allclose(fit.params["c1"].value, [0.8, 0.6], rtol=0.02)
    np.testing.assert_allclose(fit.friction.params["breakaway"].value, [0.25, 0.15], atol=1e-3)
    np.testing.assert_allclose(fit.friction.params["kinetic"].value, [0.05, 0.0], atol=1e-3)
    assert fit.friction.params["width"].value == pytest.approx(
        0.02
    )  # as given: no point decides it
    np.testing.assert_allclose(fit(u), tau, atol=3e-3)
    noisy = fit_efficiency(*static_points(truth, noise=0.005), friction=True, width=0.02)
    np.testing.assert_allclose(noisy.friction.params["breakaway"].value, [0.25, 0.15], atol=0.04)


def test_a_large_breakaway_a_second_degree_and_a_friction_free_actuator_are_fitted_too():
    big = StaticFriction([0.6, 0.55], [0.1, 0.15], 0.01)
    fit = fit_efficiency(*static_points(big), friction=True)
    np.testing.assert_allclose(fit.friction.params["breakaway"].value, [0.6, 0.55], atol=0.02)
    np.testing.assert_allclose(fit.friction.params["kinetic"].value, [0.1, 0.15], atol=0.02)
    curved = static_points(StaticFriction(0.2, 0.05, 0.01), coefficients=((0.8, 0.6), (-0.3, 0.2)))
    fit = fit_efficiency(*curved, degree=2, friction=True)
    np.testing.assert_allclose(fit.params["c2"].value, [-0.3, 0.2], atol=0.02)
    np.testing.assert_allclose(fit.friction.params["breakaway"].value, 0.2, atol=0.02)
    none = fit_efficiency(*static_points(None), friction=True)  # no friction: no dead band
    assert none.friction.params["breakaway"].value.max() < 0.02
    assert none.friction.params["kinetic"].value.min() >= 0.0
    np.testing.assert_allclose(none.params["c1"].value, [0.8, 0.6], rtol=0.02)


def test_one_friction_for_all_the_motors_is_fitted_to_all_their_points():
    truth = StaticFriction(0.2, 0.05, 0.01)
    u, tau = static_points(truth, scale=(1.0, 0.7), coefficients=((0.8, 0.8),))
    shared = fit_efficiency(u, tau, friction=True, shared=True)
    assert shared.friction.params["breakaway"].value.size == 1  # one set for both motors
    np.testing.assert_allclose(shared.friction.params["breakaway"].value, 0.2, atol=0.01)
    np.testing.assert_allclose(shared.friction.params["kinetic"].value, 0.05, atol=0.01)
    np.testing.assert_allclose(shared.params["c1"].value, 0.8, rtol=0.02)


def test_without_the_friction_option_the_fit_is_as_it_was_and_a_weighted_sum_cannot_have_it():
    u, tau = static_points(None)
    plain = fit_efficiency(u, tau)
    assert plain.friction is None
    np.testing.assert_allclose(plain.params["c1"].value, [0.8, 0.6], rtol=1e-9)
    with pytest.raises(ValueError, match="weights"):
        fit_efficiency(u, tau.sum(axis=1), weights=np.ones_like(u), friction=True)
