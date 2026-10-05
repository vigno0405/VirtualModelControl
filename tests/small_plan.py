"""A mass on a line under a spring-damper controller: small enough to know the exact answers."""

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import JointSpace

MASS, FRICTION, STIFFNESS, DAMPING = 1.0, 1.0, 4.0, 1.0


def mass_spring(goal=1.0, stiffness=STIFFNESS):
    """The system, the mass's coordinate and the controller's goal.

    m x'' + (friction + damping) x' + stiffness (x - goal) = 0.
    """
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, MASS))
    robot.add("friction", vmc.LinearDamper(x, FRICTION))
    ref = vmc.Ref("goal", 1, [goal])
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(x - ref, stiffness))
    ctrl.add("damper", vmc.LinearDamper(x, DAMPING))
    return vmc.VirtualMechanismSystem(robot, ctrl), x, ref


def exact_position(t, goal=1.0, x0=0.0, stiffness=STIFFNESS):
    """x(t) of the underdamped mass-spring from rest at x0."""
    damping = FRICTION + DAMPING
    w0 = np.sqrt(stiffness / MASS)
    zeta = damping / (2.0 * np.sqrt(MASS * stiffness))
    wd = w0 * np.sqrt(1.0 - zeta**2)
    envelope = np.exp(-zeta * w0 * t)
    swing = np.cos(wd * t) + zeta / np.sqrt(1.0 - zeta**2) * np.sin(wd * t)
    return goal + (x0 - goal) * envelope * swing


def tanh_mass(stiffness=2.0, max_force=1.0, goal=1.0, bounds=(0.5, 50.0), scale=1.0):
    """The same mass under a saturating spring whose stiffness can be optimized."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, MASS))
    robot.add("friction", vmc.LinearDamper(x, FRICTION))
    ref = vmc.Ref("goal", 1, [goal])
    ctrl = vmc.Mechanism("ctrl")
    k = vmc.Param("stiffness", stiffness, bounds=bounds, scale=scale, scope="stage")
    ctrl.add("spring", vmc.TanhSpring(x - ref, k, max_force))
    ctrl.add("damper", vmc.LinearDamper(x, DAMPING))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def two_masses():
    """Two masses on lines, with a damping matrix (not symmetric) that can be optimized."""
    robot = vmc.Mechanism("robot", model=JointSpace(2, unit="m"))
    q = robot.joint(slice(0, 2))
    robot.add("mass", vmc.Inertance(q, MASS))
    ctrl = vmc.Mechanism("ctrl")
    goal = vmc.Ref("goal", 2, [1.0, 0.5])
    ctrl.add("spring", vmc.LinearSpring(q - goal, STIFFNESS))
    damping = vmc.Param("damping", [[2.0, 0.6], [-0.4, 1.5]], bounds=(-3.0, 8.0), scope="stage")
    ctrl.add("damper", vmc.LinearDamper(q, damping))
    return vmc.VirtualMechanismSystem(robot, ctrl), q
