"""UR5 arm: six revolute joints from its Denavit–Hartenberg table, and the ADAPT hand on it."""

from __future__ import annotations

import numpy as np

from ..core.params import Param
from ..mechanisms import Mechanism
from ..models import Assembly, SerialChain
from . import adapt

DH_D = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
"""Link offsets d [m] (standard Denavit–Hartenberg, manufacturer values)."""
DH_A = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
"""Link lengths a [m]."""
DH_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
"""Link twists α [rad]."""

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRAVITY = (0.0, 0.0, -9.81)
"""Gravity in the arm's base frame [m/s²], base on a table."""


def model() -> SerialChain:
    """The arm's kinematics; site ``tool`` is the flange frame."""
    return SerialChain.from_dh(DH_D, DH_A, DH_ALPHA, tool="tool")


def arm(name: str = "ur5") -> Mechanism:
    """The UR5 alone; q holds its six joint angles [rad] (it is position-controlled)."""
    return Mechanism(name, model=model())


def with_hand(
    name: str = "ur5_hand", mounting_angle: float = adapt.HAND_MOUNTING_ANGLE
) -> Mechanism:
    """The UR5 with the ADAPT hand on its flange: q = (6 arm joints, 13 hand motors).

    Gravity stays in the arm's base frame, so the hand's gravity follows the arm's pose. The arm's
    joints are measured only: send the last 13 torques to the hand.
    """
    body = Assembly({
        "ur5": (model(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "hand": (adapt.hand_model(), (0.0, 0.0, 0.0), (0.0, 0.0, mounting_angle), "ur5/tool"),
    })  # fmt: skip
    robot = Mechanism(name, model=body)
    robot.add_param(
        Param("gravity", GRAVITY, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf))
    )
    return adapt.add_hand_masses(robot, prefix="hand/")
