"""ADAPT finger: three phalanges driven by two motors, the last two joints moving together."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..mechanisms import Custom, Joint, LimitSpring, Mechanism, PointMass
from ..models import LinearCoupling, SerialChain

LINK_LENGTHS = (0.040, 0.030, 0.0175)
"""Proximal, middle and distal phalanx [m], from the MCP joint to the fingertip."""

MOTOR_RADIUS = 0.005  # [m] motor pulley
FINGER_PULLEY_RADIUS = 0.00223  # [m] pulley on the MCP joint
PIP_TRANSMISSION = 0.00813  # [m] cable transmission constant of the PIP/DIP drive

COUPLING = np.array([
    [FINGER_PULLEY_RADIUS / MOTOR_RADIUS, 0.0],
    [0.0, MOTOR_RADIUS / PIP_TRANSMISSION],
    [0.0, MOTOR_RADIUS / PIP_TRANSMISSION],
])  # fmt: skip
"""Joint angles (MCP, PIP, DIP) = COUPLING · motor angles; the DIP joint mimics the PIP joint."""

LINK_MASSES = (0.0057, 0.0040, 0.025)
"""Masses of the three phalanges [kg]."""

LINK_COGS = (
    np.array([-0.000639, 0.020, 0.001276]),
    np.array([0.000925, 0.015, 0.001138]),
    np.array([-0.000716, 0.0105, 0.000955]),
)
"""Centres of gravity in each phalanx's frame [m]."""

GRAVITY = (0.0, 0.0, 9.81)
"""Gravity in the finger's base frame [m/s²], as mounted (the frame's z axis points down)."""

JOINT_LIMITS = {"MCP": (0.0, np.pi / 2), "PIP": (0.0, 1.22173), "DIP": (0.0, 1.22173)}
"""Joint ranges [rad]."""

LIMIT_STIFFNESS = 1.0  # [N·m/rad], stiffness of the joint-limit springs

MOTOR_EFFICIENCY = (0.8373, 0.3594)
"""Delivered over commanded torque of the two motors (MCP, PIP)."""

TORQUE_LIMIT = 0.8  # [N·m] per motor, for an optional clip at the hardware boundary


def finger(name: str = "finger", gravity: Any = None) -> Mechanism:
    """The finger as a robot mechanism; q holds the two motor angles [rad].

    Sites: ``pip``, ``dip``, ``tip`` and the phalanges' centres of gravity ``*_cog``.
    """
    a, b, c = LINK_LENGTHS
    x = [1.0, 0.0, 0.0]
    joints = ([0.0, 0.0, 0.0], [0.0, a, 0.0], [0.0, a + b, 0.0])
    sites = {
        "pip": (1, joints[1]),
        "dip": (2, joints[2]),
        "tip": (3, [0.0, a + b + c, 0.0]),
    }
    for i, link in enumerate(("mcp", "pip", "dip")):
        sites[f"{link}_cog"] = (i + 1, np.asarray(joints[i]) + LINK_COGS[i])
    model = LinearCoupling(SerialChain(["revolute"] * 3, [x, x, x], list(joints), sites), COUPLING)
    robot = Mechanism(name, model=model)
    robot.add_param(
        Param(
            "gravity",
            GRAVITY if gravity is None else gravity,
            unit="m/s^2",
            scope="design",
            bounds=(-np.inf, np.inf),
        )
    )
    for i, link in enumerate(("mcp", "pip", "dip")):
        robot.add(f"m_{link}", PointMass(robot.point(f"{link}_cog"), LINK_MASSES[i]))
    return robot


def joint_angles(robot: Mechanism) -> Custom:
    """Coordinate of the three joint angles (MCP, PIP, DIP) [rad] of a finger."""
    coupling = robot.model.coupling
    return Custom(
        lambda motors, coupling: coupling @ motors,
        [Joint(slice(0, 2), unit="rad")],
        dim=3,
        unit="rad",
        params={"coupling": coupling},
    )


def joint_limit_spring(robot: Mechanism, stiffness: Any = LIMIT_STIFFNESS) -> LimitSpring:
    """A spring that pushes each joint back inside its range (zero force inside)."""
    lower = [lim[0] for lim in JOINT_LIMITS.values()]
    upper = [lim[1] for lim in JOINT_LIMITS.values()]
    return LimitSpring(joint_angles(robot), stiffness, lower, upper)
