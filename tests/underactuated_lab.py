"""The three underactuated robots and the lab's virtual models, as the library builds them."""

from pathlib import Path

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import StateController
from virtualmodelcontrol.control import underactuated as ua
from virtualmodelcontrol.robots import planar

DATA = Path(__file__).resolve().parent / "data" / "underactuated.npz"
ROBOTS = ("three", "five", "helyx")
COMBOS = [(b, c) for b in ("naive", "frozen") for c in (None, "passive", "tank")]


def lab():
    return np.load(DATA)


def build(prefix):
    """The robot with its dynamics; its plane is x-y for the rigid arms and x-z for the soft one."""
    if prefix == "helyx":
        return planar.add_continuum_dynamics(planar.continuum())
    return planar.add_dynamics(planar.arm({"three": "three-link", "five": "five-link"}[prefix]))


def to_3d(prefix, xy):
    """A planar lab vector (x, y) in the library's frame."""
    xy = np.asarray(xy, dtype=float)
    return np.array([xy[0], 0.0, xy[1]]) if prefix == "helyx" else np.array([xy[0], xy[1], 0.0])


def law(prefix, robot, gravity=False, matrix=False):
    """The lab's virtual spring and damper on the tip, compiled; ``matrix`` makes the stiffness a
    (3, 3) one, for the force tracking."""
    data = lab()
    K, D = float(data[f"{prefix}_stiffness"][0]), float(data[f"{prefix}_damping"][0])
    goal = to_3d(prefix, data[f"{prefix}_target"])
    tip = robot.point("tip")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - goal, K * np.eye(3) if matrix else K))
    ctrl.add("damp", vmc.LinearDamper(tip, D))
    if gravity:
        ctrl.add("gravity", vmc.GravityCompensation(robot))
    return vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))


def measurement(robot, q, v, t=0.0):
    """What a simulated plant reads: the motors and the state."""
    B = np.asarray(robot.actuation.params["B"].value)
    return vmc.Signals(t, motor_position=B.T @ q, motor_velocity=B.T @ v, q=q, v=v)


class LabRead:
    """The lab's moving rest: the passive part at its balance at every reading, and the torque the
    least-squares B⁺ τ of the wrench at that pose, without the pull-back through P = ∂π/∂θ. It is
    not the gradient of a potential; the library's ``rest="moving"`` adds the pull-back."""

    def __init__(self, robot):
        self.moving = ua.Moving(robot.actuation, vmc.compile_dynamics(robot))
        self.B = np.asarray(robot.actuation.params["B"].value)

    def __call__(self, meas):
        q = self.moving.pullback(meas)[0]
        return q, self.B @ np.linalg.solve(self.B.T @ self.B, meas["motor_velocity"])


def lab_controller(compiled, robot, base, correction=None, gravity=False):
    """The lab's controller: the library's, except that the frozen one with ``gravity`` follows the
    lab's moving rest."""
    if not (base == "frozen" and gravity):
        return ua.controller(compiled, base, correction)
    dynamics = vmc.compile_dynamics(robot)
    stages = (
        []
        if correction is None
        else [ua.Passivation(dynamics, 1.0 if correction == "tank" else None)]
    )
    return StateController(compiled, stages, read=LabRead(robot))
