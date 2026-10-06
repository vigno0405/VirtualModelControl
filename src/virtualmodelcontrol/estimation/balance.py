"""The static balance of a controlled robot, as a CasADi expression of the controller's inputs."""

from __future__ import annotations

from typing import Any

import casadi as ca

from ..core.params import constants
from ..dynamics import compile_dynamics


def balance(controller: Any, robot: Any = None) -> tuple[Any, Any, Any]:
    """The packed inputs x, the motor angles in it, and the motor torques the contact must take.

    That is what the controller's law delivers through the system's transmission, plus the forces
    the robot's own components hold at rest (``robot`` if given, else the system's). Velocities
    are left out; so are the output stages after the law.
    """
    compiled = controller.compiled
    system = compiled.system
    act, pa = system.actuation, constants(system.actuation.params)
    dynamics = compile_dynamics(system.robot if robot is None else robot, actuation=act)
    n_angles, n_rates = compiled.n_motors
    x = ca.SX.sym("x", compiled.fast.size1_in(0))
    theta = x[:n_angles]
    still = ca.vertcat(theta, ca.DM.zeros(n_rates), x[n_angles + n_rates :])
    u = compiled.fast(still)[: compiled.n_u]
    q = act.config_from_motors(theta, pa)
    zero = ca.DM.zeros(dynamics.robot.model.space.nv)
    off = ca.DM.zeros(dynamics.n_u)  # the robot's own forces: no command
    own = -dynamics.residual(q, zero, zero, off, dynamics.live_values(), x[-1])
    held = act.allocate(own + act.generalized_force(u, q, pa), q, pa)
    return x, theta, held
