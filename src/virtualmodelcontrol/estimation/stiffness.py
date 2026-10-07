"""Task-space stiffness of a robot at rest in contact, from its controller and its model."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..control.state import require_motor_layout
from ..models.kinematics import Kinematics, contact_map, contact_projection
from .balance import balance


class TaskStiffness:
    """The stiffness [N/m] that the robot, held at rest by its controller, shows at ``site``.

    The stiffness of the motors' static balance, with the changes of the Jacobians counted, is
    carried to the site through the contact map: a 3 by 3 matrix, along ``normal`` if given.
    ``f_ext`` is the contact force [N] the geometric terms are taken at (default: the model's
    own). ``robot`` and the omissions are those of ``ContactForce``.
    """

    def __init__(
        self, controller: Any, site: Any, normal: ArrayLike | None = None, robot: Any = None
    ) -> None:
        require_motor_layout(controller, "TaskStiffness")
        x, theta, held = balance(controller, robot)
        robot_model = controller.compiled.system.robot
        _, _, J, _, H = Kinematics(robot_model, coordinates="motors").functions(site)(theta)
        self.terms = ca.Function("terms", [x], [held, -ca.jacobian(held, theta), J, H])
        self._normal = normal

    def motors(self, controller: Any, f_ext: ArrayLike | None = None) -> tuple[Any, Any]:
        """The contact map A (3, n) and the stiffness of the motors' balance (n, n) with the
        geometric terms of the site at the contact force ``f_ext``."""
        held, K, J, H = (np.array(m) for m in self.terms(controller.inputs()))
        A = contact_map(J, self._normal)
        f = A @ held.ravel() if f_ext is None else np.ravel(f_ext)
        pushed, n = contact_projection(self._normal) @ f, K.shape[0]
        return A, K + sum(pushed[k] * H[k * n : (k + 1) * n] for k in range(3))

    def __call__(self, controller: Any, f_ext: ArrayLike | None = None) -> np.ndarray:
        """The stiffness at the controller's last step, (3, 3)."""
        A, K = self.motors(controller, f_ext)
        return np.asarray(A @ K @ A.T)
