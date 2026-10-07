"""Contact force from the controller's command and the robot's own model."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..control.state import require_motor_layout
from ..models.kinematics import Kinematics, contact_map
from .balance import balance


class ContactForce:
    """The force [N] the robot exerts at ``site`` when it rests, from what its controller commands.

    What the controller's law delivers and the robot's own stiffness and weight (``robot``: the
    model of those, without the surroundings) do not balance goes through the contact; with a
    ``normal`` it is the force along it. Velocities and output stages are left out.
    """

    def __init__(
        self, controller: Any, site: Any, normal: ArrayLike | None = None, robot: Any = None
    ) -> None:
        require_motor_layout(controller, "ContactForce")
        x, theta, held = balance(controller, robot)
        robot_model = controller.compiled.system.robot
        J = Kinematics(robot_model, coordinates="motors").functions(site)(theta)[2]
        self._terms = ca.Function("terms", [x], [held, J])
        self._normal = normal

    def __call__(self, controller: Any) -> np.ndarray:
        """The force at the controller's last step, (3,)."""
        held, J = (np.array(m) for m in self._terms(controller.inputs()))
        return contact_map(J, self._normal) @ held.ravel()
