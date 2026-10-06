"""The model checks: the library's robots pass them, and models that break the contract do not."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt, helyx, turtle, ur5
from virtualmodelcontrol.testing import _derivatives, check_model


def test_the_library_s_robots_keep_the_contract():
    check_model(adapt.finger())
    check_model(ur5.arm())
    check_model(adapt.hand(), at=["thumb/tip", "ring/tip"], samples=2)
    check_model(helyx.arm("145-290-290"), at=["tip"], s=np.linspace(0.0, 1.0, 101), samples=2)
    check_model(turtle.robot())  # no sites to check


def test_the_worst_error_of_every_check_is_reported():
    worst = check_model(adapt.finger(), samples=2)
    assert set(worst) == {
        "finite",
        "rotation",
        "jacobian",
        "angular_jacobian",
        "hessian",
        "serialization",
        "energy",
    }
    assert worst["finite"] == 0 and worst["jacobian"] < 1e-8


class Pendulum:
    """A pendulum about z of length L: sound, and the base of the broken ones."""

    q_unit = "rad"

    def __init__(self, length=0.5):
        self.space = vmc.Euclidean(1)
        self.params = vmc.ParamSet([vmc.Param("L", length, unit="m", bounds=(0.0, np.inf))])
        self.sites = ("bob",)

    def frame(self, q, at, p):
        c, s = ca.cos(q[0]), ca.sin(q[0])
        R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
        return R, p["L"] * ca.vertcat(c, s, 0)


def failing(model, name, **kwargs):
    """The message of the check that fails, and that it names ``name``."""
    with pytest.raises(AssertionError, match="breaks the contract") as error:
        check_model(model, **kwargs)
    assert name in str(error.value), str(error.value)


def test_a_sound_model_passes():
    worst = check_model(Pendulum())
    assert worst["rotation"] < 1e-15 and worst["hessian"] < 1e-8


def test_a_pose_where_the_position_is_not_finite_is_caught():
    class Singular(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            return R, position + ca.vertcat(q[0] / ca.fabs(q[0]), 0, 0)  # 0 / 0 at the neutral pose

    failing(Singular(), "finite")


def test_a_pose_where_only_a_derivative_is_not_finite_is_caught():
    class Steep(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            return R, position + ca.vertcat(ca.sqrt(ca.fabs(q[0])), 0, 0)  # its slope is infinite

    failing(Steep(), "finite")


def test_a_kink_at_the_neutral_pose_is_caught_by_the_hessian():
    class Kinked(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            return R, position + ca.vertcat(ca.fabs(q[0]), 0, 0)  # no second derivative at zero

    failing(Kinked(), "hessian")


def test_a_rotation_that_is_not_orthonormal_is_caught():
    class Stretched(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            return 1.1 * R, position

    failing(Stretched(), "rotation")


def test_a_jump_along_the_body_is_caught():
    class Jumpy(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            along = 0.0 if isinstance(at, str) else at  # a site, or a point of the body
            return R, position + ca.vertcat(0, 0, ca.if_else(along < 0.5, 0.0, 1.0))

    check_model(Jumpy(), at=["bob"])  # with sites only there is no body to be continuous
    failing(Jumpy(), "continuity", at=[0.0], s=np.linspace(0.0, 1.0, 51))


def test_a_model_that_breaks_with_a_param_ten_times_larger_is_caught():
    class Fragile(Pendulum):
        def frame(self, q, at, p):
            R, position = super().frame(q, at, p)
            return R, position + ca.vertcat(1.0 / (p["L"] - 5.0), 0, 0)  # at L = 0.5, ten times it

    failing(Fragile(), "finite", at=["bob"])


@vmc.register("model", "forgetful_pendulum")
class Forgetful(Pendulum):
    """Serializes without its length, so the model read back is another one."""

    def to_dict(self):
        return {"type": "forgetful_pendulum"}

    @classmethod
    def from_dict(cls, data):
        return cls(1.0)


def test_a_model_that_does_not_survive_its_dict_is_caught():
    failing(Forgetful(0.5), "serialization")
    check_model(Forgetful(1.0))  # one that comes back as it was is fine


def test_a_robot_without_dampers_must_keep_its_energy():
    def pendulum(push):
        robot = vmc.Mechanism("pendulum", model=Pendulum())
        robot.add_param(vmc.Param("gravity", [0.0, -9.81, 0.0], unit="m/s^2"))
        robot.add("bob", vmc.PointMass(robot.point("bob"), 0.2))
        robot.add("weight", vmc.Gravity(robot))
        if push:  # a force that does work, whose drift does not fall with the step
            robot.add("push", vmc.ForceSource(robot.joint(0), 0.3))
        return robot

    assert check_model(pendulum(False))["energy"] == 0.0  # the drift falls with the step
    failing(pendulum(True), "energy")
    check_model(pendulum(True), energy=False)


class Skewed:
    """A kinematics whose Jacobians are 10 % off, to see that the checks would notice."""

    def __init__(self, which):
        self.real, self.which = vmc.Kinematics(Pendulum()), which

    def functions(self, at):
        def evaluate(q):
            p, R, J, Jw, H = (np.array(x) for x in self.real.functions(at)(q))
            return (
                p,
                R,
                J * (1.1 if self.which == "jacobian" else 1.0),
                Jw * (1.1 if self.which == "angular_jacobian" else 1.0),
                H * (1.1 if self.which == "hessian" else 1.0),
            )

        return evaluate

    def position(self, q, at):
        return self.real.position(q, at)

    def rotation(self, q, at):
        return self.real.rotation(q, at)

    def jacobian(self, q, at):
        return self.real.jacobian(q, at)


@pytest.mark.parametrize("which", ["jacobian", "angular_jacobian", "hessian"])
def test_a_derivative_that_disagrees_with_finite_differences_is_caught(which):
    errors = {}
    q = np.array([0.4])
    _derivatives(
        Skewed(which),
        vmc.Euclidean(1),
        "bob",
        q,
        lambda name, e, tol, where: errors.update({name: e}),
    )
    assert errors[which] > 1e-3
    assert all(errors[name] < 1e-4 for name in errors if name not in (which, "finite"))
