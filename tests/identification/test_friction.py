import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import ParamSet
from virtualmodelcontrol.identification import Steps, fit_stiffness_damping
from virtualmodelcontrol.models import JointSpace

MASS, K, D = np.array([0.5, 1.0]), np.array([40.0, 25.0]), np.array([2.0, 1.5])
F = np.array([0.2, 0.1])  # static friction of the two motors [N·m]


class Gears:
    """Motor angles θ = R q, one motor per joint, so τ = R u: a motor with a negative ratio turns
    against its joint."""

    def __init__(self, ratio):
        self.R, self.params = ca.diag(ca.DM(ratio)), ParamSet()
        self.inverse = ca.inv(self.R)

    def motor_sizes(self, space):
        return space.nq, space.nv

    def motor_angles(self, q, p):
        return ca.mtimes(self.R, q)

    def motor_rates(self, q, v, p):
        return ca.mtimes(self.R, v)

    def generalized_force(self, u, q, p):
        return ca.mtimes(self.R, u)

    def allocate(self, tau, q, p):
        return ca.mtimes(self.inverse, tau)

    def config_from_motors(self, theta, p):
        return ca.mtimes(self.inverse, theta)

    def velocity_from_motors(self, q, theta_dot, p):
        return ca.mtimes(self.inverse, theta_dot)


def robot(ratio, K=None, D=None, friction=None):
    """Two masses on gears, with the stiffness, damping and static friction of the motors."""
    robot = vmc.Mechanism("robot", model=JointSpace(2, unit="m"), actuation=Gears(ratio))
    x = robot.joint(slice(0, 2))
    robot.add("mass", vmc.Inertance(x, MASS))
    if K is not None:
        robot.add("spring", vmc.LinearSpring(x, K))
        robot.add("damper", vmc.LinearDamper(x, D))
    if friction is not None:  # on the motors' angles, steep enough to stick
        motors = vmc.Custom(lambda q: ca.mtimes(robot.actuation.R, q), [x], dim=2, unit="rad")
        robot.add("friction", vmc.TanhDamper(motors, 2e4, friction))
    return robot


def steps_on(ratio, friction):
    """Torque steps on the two masses, whose motors have the given friction."""
    plant = vmc.sim.ModelPlant(robot(ratio, K, D, friction))
    clock = vmc.sim.SimClock(1 / 330)
    vmc.sim.run(plant, Steps(0.0), clock, T=1.0)
    steps = Steps(0.0, [(0, 1.0), (1, 1.0), (0, 2.0), (1, 2.0)])
    return vmc.sim.run(plant, steps, clock, T=steps.duration)


def test_friction_biases_the_damping_unless_the_fit_has_a_column_for_it():
    ratio, log = [1.0, 1.0], steps_on([1.0, 1.0], F)
    known = robot(ratio)
    _, D_without = fit_stiffness_damping(known, [log], smoothing=11)
    assert abs(D_without / D - 1).min() > 0.3  # friction passes for damping
    K_fit, D_fit, F_fit = fit_stiffness_damping(known, [log], smoothing=11, friction=True)
    np.testing.assert_allclose(F_fit, F, rtol=0.1)
    np.testing.assert_allclose(D_fit, D, rtol=0.05)
    np.testing.assert_allclose(K_fit, K, rtol=0.05)


def test_a_robot_without_friction_gets_none():
    ratio = [1.0, 1.0]
    log = steps_on(ratio, None)
    K_fit, D_fit, F_fit = fit_stiffness_damping(robot(ratio), [log], smoothing=11, friction=True)
    assert F_fit.max() < 0.01
    np.testing.assert_allclose(K_fit, K, rtol=0.02)
    np.testing.assert_allclose(D_fit, D, rtol=0.02)


def test_the_friction_of_a_motor_follows_the_motor_not_the_joint():
    ratio = [-2.0, 3.0]  # the first motor turns against its joint
    log = steps_on(ratio, F)
    K_fit, D_fit, F_fit = fit_stiffness_damping(robot(ratio), [log], smoothing=11, friction=True)
    np.testing.assert_allclose(F_fit, F, rtol=0.15)
    np.testing.assert_allclose(K_fit, K, rtol=0.1)
    np.testing.assert_allclose(D_fit, D, rtol=0.1)


def test_without_friction_the_fit_returns_stiffness_and_damping_only():
    ratio = [1.0, 1.0]
    out = fit_stiffness_damping(robot(ratio), [steps_on(ratio, None)], smoothing=11)
    assert len(out) == 2
