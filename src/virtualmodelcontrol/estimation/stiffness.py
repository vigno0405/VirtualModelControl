"""Task-space stiffness of a robot at rest in contact, from its controller and its model."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..dynamics import compile_dynamics
from ..models.kinematics import Kinematics, contact_map


class TaskStiffness:
    """The stiffness [N/m] that the robot, held at rest by its controller, shows at ``site``.

    The stiffness of the motors' static balance, with the changes of the Jacobians counted (the
    congruence transformation), is carried to the site through the contact map: along ``normal``
    if given, as a 3 by 3 matrix otherwise. ``f_ext`` is the contact force the site pushes with
    [N]; its default is the model's own, as ``ContactForce`` gives it. ``robot`` is the model that
    holds the arm, as for ``ContactForce``. Velocities are left out.
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
        f_ext = ca.SX.sym("f_ext", 3)
        theta = x[:n_angles]
        still = ca.vertcat(theta, ca.DM.zeros(n_rates), x[n_angles + n_rates :])
        u = compiled.fast(still)[: compiled.n_u]
        q = act.config_from_motors(theta, pa)
        zero = ca.DM.zeros(dynamics.robot.model.space.nv)
        residual = dynamics.residual(q, zero, zero, u, dynamics.live_values(), x[-1])
        held = act.allocate(-residual, q, pa)  # what the contact must take, as motor torques
        _, _, J, _, H = Kinematics(system.robot, coordinates="motors").functions(site)(theta)
        A = contact_map(J, normal)
        pushed = f_ext
        if normal is not None:
            n = ca.DM(np.asarray(normal, dtype=float).ravel() / np.linalg.norm(normal))
            pushed = ca.mtimes(n, ca.mtimes(n.T, f_ext))  # the part of the force along it
        size = n_angles
        geometric = sum(pushed[k] * H[k * size : (k + 1) * size, :] for k in range(3))
        K_motors = -ca.jacobian(held, theta) + geometric
        self.function = ca.Function("stiffness", [x, f_ext], [ca.mtimes([A, K_motors, A.T])])
        self._force = ca.Function("force", [x], [ca.mtimes(A, held)])

    def force(self, controller: Any) -> np.ndarray:
        """The contact force of the model [N] at the controller's last step, (3,)."""
        return np.array(self._force(controller.inputs())).ravel()

    def __call__(self, controller: Any, f_ext: ArrayLike | None = None) -> np.ndarray:
        """The stiffness at the controller's last step, (3, 3)."""
        f = self.force(controller) if f_ext is None else np.asarray(f_ext, dtype=float).ravel()
        return np.array(self.function(controller.inputs(), f))
