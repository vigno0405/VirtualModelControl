"""Contact force from the controller's command and the robot's own model."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..dynamics import compile_dynamics
from ..models.kinematics import Kinematics, contact_map


class ContactForce:
    """The force the robot exerts at ``site`` when it rests, from what the controller commands.

    The robot's stiffness, its weight and the torques its motors deliver hold it at rest. What
    they do not balance is the contact; with a ``normal`` (any length) it is the force along it.
    Velocities are left out. ``robot`` is the model whose components do the holding: give one
    without the surroundings when the system's robot includes them. Evaluated with the Params'
    values at construction, for the controller's live Params as they are when called.
    """

    def __init__(
        self, controller: Any, site: Any, normal: ArrayLike | None = None, robot: Any = None
    ) -> None:
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
        residual = dynamics.residual(q, zero, zero, u, dynamics.live_values(), x[-1])
        held = act.allocate(-residual, q, pa)  # what the contact must take, as motor torques
        J = Kinematics(system.robot, coordinates="motors").functions(site)(theta)[2]
        self._force = ca.Function("contact_force", [x], [ca.mtimes(contact_map(J, normal), held)])

    def __call__(self, controller: Any) -> np.ndarray:
        """The force [N] at the controller's last step, (3,)."""
        return np.array(self._force(controller.inputs())).ravel()
