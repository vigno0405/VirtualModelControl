"""The friction compensation, on the robot only, and the names that 2.0.0 removed."""

import warnings

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import FrictionCompensation
from virtualmodelcontrol.core.registry import get
from virtualmodelcontrol.identification import fit_efficiency, fit_stiffness_damping
from virtualmodelcontrol.models import Direct, Efficiency, JointSpace
from virtualmodelcontrol.robots import adapt

V = 0.03  # the speed [rad/s] past which the compensation fades


def meas(rate):
    rate = np.atleast_1d(np.asarray(rate, dtype=float))
    return vmc.Signals(0.0, motor_position=np.zeros_like(rate), motor_velocity=rate)


def test_it_adds_F_hat_at_rest_in_the_direction_of_the_command_with_a_sign_or_a_smooth_sign():
    F_hat, w = np.array([0.2, 0.1]), 0.02
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, rate = rng.uniform(-0.3, 0.3, 2), rng.uniform(-0.1, 0.1, 2)
        gate = np.exp(-((rate / V) ** 2))
        plain = FrictionCompensation(F_hat, V)(u, meas(rate))
        np.testing.assert_allclose(plain, u + F_hat * gate * np.sign(u), atol=1e-15)
        smooth = FrictionCompensation(F_hat, V, width=w)(u, meas(rate))
        np.testing.assert_allclose(
            smooth, u + F_hat * gate * u / np.sqrt(u * u + w * w), atol=1e-15
        )
    plain, smooth = FrictionCompensation(0.2, V), FrictionCompensation(0.2, V, width=w)
    assert plain.width == 0.0  # the lab's formula unless a width is given
    small = np.array([0.005])
    assert plain(small, meas(0.0))[0] == pytest.approx(0.205)  # the sign: all of F_hat
    assert smooth(small, meas(0.0))[0] == pytest.approx(0.005 + 0.2 * 0.005 / np.hypot(0.005, w))
    assert plain(np.zeros(1), meas(0.0))[0] == 0.0 == smooth(np.zeros(1), meas(0.0))[0]
    for rate in (0.1, -0.1):  # faded once the motor moves, either way
        assert abs(plain(small, meas(rate))[0] - 0.005) < 1e-5
    assert get("output", "friction_compensation") is FrictionCompensation


def test_a_plan_takes_no_friction_compensation():
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="rad"), actuation=Direct())
    q = robot.joint(0)
    robot.add("inertia", vmc.Inertance(q, 1.0))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(q - vmc.Ref("goal", 1, [1.0]), 1.0))
    system = vmc.VirtualMechanismSystem(robot, ctrl)
    with pytest.raises(ValueError, match="FrictionCompensation has no symbolic form"):
        opt.Problem(system, output=[FrictionCompensation(0.2, V, width=0.02)])


def test_the_compensation_of_the_finger_and_the_hand_is_off_unless_asked_for():
    assert adapt.COMPENSATION == (0.04, 0.03) and adapt.HAND_COMPENSATION == (0.1, 0.03)
    assert adapt.output_stage()[0] == FrictionCompensation(0.04, 0.03)
    assert adapt.hand_output_stage()[0] == FrictionCompensation(0.1, 0.03)
    assert "damping" not in adapt.add_dynamics(adapt.finger()).components


def test_the_static_friction_of_versions_before_2_0_0_is_gone():
    import virtualmodelcontrol.control as control
    import virtualmodelcontrol.models as models

    for module, name in (
        (vmc, "StaticFriction"),
        (models, "StaticFriction"),
        (control, "StaticFrictionCompensation"),
        (adapt, "finger_friction"),
        (adapt, "hand_friction"),
        (adapt, "FRICTION"),
        (adapt, "HAND_FRICTION"),
    ):
        assert not hasattr(module, name), name
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="rad"), actuation=Direct())
    robot.add("inertia", vmc.Inertance(robot.joint(0), 1.0))
    t = np.arange(400) * 0.01
    run = {"t": t, "q": np.sin(t)[:, None], "v": np.cos(t)[:, None], "u": np.sin(t)[:, None]}
    points = np.linspace(-1, 1, 41)[:, None]
    for call in (
        lambda: Efficiency(1.0, friction={"breakaway": 0.1}),
        lambda: fit_efficiency(points, points, friction=True),
        lambda: fit_stiffness_damping(robot, [run], smoothing=11, friction=True),
    ):
        with pytest.raises(TypeError, match="friction"):
            call()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and nothing warns
        FrictionCompensation(0.1, V, width=0.01)
        adapt.output_stage(), adapt.hand_output_stage()
        assert len(fit_stiffness_damping(robot, [run], smoothing=11)) == 2  # K and D only
