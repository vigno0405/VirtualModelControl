"""ADAPT finger and hand: tendon-driven phalanges, the last two joints of each finger coupled."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..mechanisms import Custom, Joint, LimitSpring, Mechanism, PointMass
from ..models import Assembly, LinearCoupling, SerialChain

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


# =============================================================================================
# ADAPT hand: thumb (4 joints) and four fingers (spread, MCP, PIP, DIP) driven by 13 motors.
# =============================================================================================

HAND_PALM_ORIGIN = (-0.017825999999999998, 0.0030209999999999994, -0.027998000000000002)
"""Palm origin in the hand base frame [m] (the wrist is held rigid at zero)."""

HAND_FINGER_BASES = {
    "thumb": (-0.001126, 0.011348, 0.056215),
    "index": (-0.001654, 0.040836, 0.093101),
    "middle": (0.02173, 0.039146, 0.092417),
    "ring": (0.032774, 0.039367, 0.090557),
    "pinky": (0.061943, 0.039534, 0.075302),
}
"""Base of each digit relative to the palm [m]."""

HAND_JOINTS = {
    "thumb": (
        ("CMC1", (-0.014257, 0.010829, -0.007467), (0.182125, -0.177737, -0.967078)),
        ("CMC2", (-0.008631, 0.00194, -0.008807), (0.0, 0.983527, -0.180761)),
        ("MCP", (-0.04412, -0.000957, -0.00886), (0.168863, 0.20364, -0.964373)),
        ("IP", (-0.031464, -0.001054, -0.005732), (0.168863, 0.20364, -0.964373)),
    ),
    "index": (
        ("Spread", (-0.006284, 0.000712, 0.005798), (0.140517, 0.943569, -0.299886)),
        ("MCP", (-0.001682, -0.00044, 0.009499), (-0.981214, 0.092278, -0.169421)),
        ("PIP", (-0.005287, 0.012722, 0.037552), (-0.981214, 0.092278, -0.169421)),
        ("DIP", (-0.003966, 0.009542, 0.028164), (-0.981214, 0.092278, -0.169421)),
    ),
    "middle": (
        ("Spread", (-0.005011, 0.000697, 0.006929), (0.036359, 0.936295, -0.349327)),
        ("MCP", (0.000289, -0.000149, 0.009651), (-0.998268, 0.050205, 0.030661)),
        ("PIP", (0.002081, 0.015643, 0.042143), (-0.998268, 0.050205, 0.030661)),
        ("DIP", (0.001387, 0.010428, 0.028095), (-0.998268, 0.050205, 0.030661)),
    ),
    "ring": (
        ("Spread", (0.006841, 0.000228, 0.005174), (-0.055584, 0.937022, -0.34482)),
        ("MCP", (0.002501, -0.000155, 0.009326), (-0.964998, 0.038236, 0.259457)),
        ("PIP", (0.010252, 0.013887, 0.036084), (-0.964998, 0.038236, 0.259457)),
        ("DIP", (0.007689, 0.010415, 0.027062), (-0.964998, 0.038236, 0.259457)),
    ),
    "pinky": (
        ("Spread", (-0.001073, -0.000153, 0.008511), (-0.211896, 0.947676, -0.238768)),
        ("MCP", (0.004931, -0.000503, 0.008288), (-0.859324, -0.064308, 0.507372)),
        ("PIP", (0.014895, 0.010006, 0.026496), (-0.859324, -0.064308, 0.507372)),
        ("DIP", (0.009309, 0.006254, 0.016559), (-0.859324, -0.064308, 0.507372)),
    ),
}
"""Joints of each digit: name, origin in the parent link frame [m], axis in the joint frame."""

HAND_TIP_OFFSETS = {
    "thumb": (0.0, 0.0, 0.0175),
    "index": (0.0, 0.0, 0.0175),
    "middle": (0.0, 0.0, 0.0175),
    "ring": (0.0, 0.0, 0.0175),
    "pinky": (0.0, 0.0, 0.0175),
}
"""Fingertip in the last joint frame [m]."""

HAND_FINGER_TRANSMISSIONS = {
    "index": {"r_motor": 0.005, "r_pulley": 0.00304, "c_param": 0.0109},
    "middle": {"r_motor": 0.005, "r_pulley": 0.00304, "c_param": 0.01127},
    "ring": {"r_motor": 0.005, "r_pulley": 0.00304, "c_param": 0.0109},
    "pinky": {"r_motor": 0.005, "r_pulley": 0.00304, "c_param": 0.00967},
}
"""Motor pulley, finger pulley and PIP cable constant of each finger [m]."""

HAND_THUMB_TRANSMISSION = {
    "r_motor": 0.005,
    "CMC1_pulley": 0.01346,
    "CMC2_pulley": 0.00883,
    "MCP_c": 0.00605,
    "IP_c": 0.00577,
}
"""Motor pulley, CMC pulleys and MCP/IP cable constants of the thumb [m]."""

HAND_SPREAD_RATIOS = {"index": 1.0, "middle": 0.0, "ring": -1.0, "pinky": -1.5}
HAND_SPREAD_CORRECTION = -0.03333333333333333
"""Spread angle of each finger = spread motor angle · correction · ratio."""

HAND_LINK_MASSES = {
    "wrist_joint": 0.0104,
    "palm": 0.0808,
    "thumb_base": 0.0035,
    "thumb_2dof_joint": 0.007,
    "thumb_proximal": 0.0068,
    "thumb_middle": 0.0043,
    "thumb_distal": 0.0035,
    "index_base": 0.0014,
    "index_2dof_joint": 0.0011,
    "index_proximal": 0.0057,
    "index_middle": 0.004,
    "index_distal": 0.0025,
    "middle_base": 0.0014,
    "middle_2dof_joint": 0.0011,
    "middle_proximal": 0.0066,
    "middle_middle": 0.004,
    "middle_distal": 0.0025,
    "ring_base": 0.0014,
    "ring_2dof_joint": 0.0011,
    "ring_proximal": 0.0057,
    "ring_middle": 0.004,
    "ring_distal": 0.0025,
    "pinky_base": 0.0014,
    "pinky_2dof_joint": 0.0011,
    "pinky_proximal": 0.0043,
    "pinky_middle": 0.0022,
    "pinky_distal": 0.0025,
}
"""Link masses [kg]."""

HAND_LINK_COGS = {
    "wrist_joint": (-2.553e-07, -0.015, -1.074e-05),
    "palm": (0.0193, 0.0162, 0.0492),
    "thumb_base": (-0.0059, 0.0034, -0.0113),
    "thumb_2dof_joint": (-0.0027, -0.0041, -0.0065),
    "thumb_proximal": (-0.021, -0.0001, -0.0038),
    "thumb_middle": (-0.0156, 0.0007, -0.0034),
    "thumb_distal": (-0.013, 0.0005, -0.0016),
    "index_base": (-0.0058, -0.0049, 1.46e-05),
    "index_2dof_joint": (-0.0021, -0.0022, 0.0035),
    "index_proximal": (-0.0018, 0.0075, 0.0185),
    "index_middle": (-0.0027, 0.0059, 0.0135),
    "index_distal": (-0.0005, 0.0041, 0.0096),
    "middle_base": (-0.0054, -0.0051, 0.0013),
    "middle_2dof_joint": (-0.0012, -0.0022, 0.0039),
    "middle_proximal": (0.0016, 0.009, 0.0205),
    "middle_middle": (-0.0001, 0.0063, 0.0136),
    "middle_distal": (0.0012, 0.0045, 0.0094),
    "ring_base": (0.0051, -0.0055, -0.0001),
    "ring_2dof_joint": (-0.0003, -0.0022, 0.0041),
    "ring_proximal": (0.0056, 0.0081, 0.0174),
    "ring_middle": (0.0028, 0.0063, 0.0133),
    "ring_distal": (0.0033, 0.0045, 0.0089),
    "pinky_base": (-0.0037, -0.0057, 0.0033),
    "pinky_2dof_joint": (0.0009, -0.0024, 0.0038),
    "pinky_proximal": (0.0079, 0.0061, 0.0125),
    "pinky_middle": (0.003, 0.0037, 0.0089),
    "pinky_distal": (0.0052, 0.0042, 0.0081),
}
"""Centre of gravity of each link in its frame [m]."""

HAND_JOINT_LIMITS = {
    "thumb_CMC1": (0.0, 1.5),
    "thumb_CMC2": (-0.5, 0.5),
    "thumb_MCP": (0.0, 1.5),
    "thumb_IP": (0.0, 1.2),
    "index_spread": (-0.3, 0.0),
    "index_MCP": (0.0, 1.5),
    "index_PIP": (0.0, 1.2),
    "index_DIP": (0.0, 1.2),
    "middle_spread": (0.0, 0.0),
    "middle_MCP": (0.0, 1.5),
    "middle_PIP": (0.0, 1.2),
    "middle_DIP": (0.0, 1.2),
    "ring_spread": (0.0, 0.3),
    "ring_MCP": (0.0, 1.5),
    "ring_PIP": (0.0, 1.2),
    "ring_DIP": (0.0, 1.2),
    "pinky_spread": (0.0, 0.3),
    "pinky_MCP": (0.0, 1.5),
    "pinky_PIP": (0.0, 1.2),
    "pinky_DIP": (0.0, 1.2),
}
"""Joint ranges [rad]."""

HAND_MOTOR_EFFICIENCY = (
    1.0,
    1.0,
    0.8373,
    0.3594,
    1.0,
    0.8373,
    0.3594,
    0.8373,
    0.3594,
    0.8373,
    0.3594,
    0.8373,
    0.3594,
)
HAND_LIMIT_STIFFNESS = 0.6  # [N·m/rad]
HAND_TORQUE_LIMIT = 0.8  # [N·m]
HAND_FRICTION = (0.1, 0.03)  # Stribeck: max torque [N·m], velocity [rad/s]
HAND_MASS_ON_FLANGE = 1.61  # [kg], for the arm's payload
HAND_COG_ON_FLANGE = (-0.01, 0.004, 0.072)  # [m]
HAND_MOUNTING_ANGLE = 0.0  # [rad], about the flange z axis
HAND_WRIST = {
    "yaw_origin": (-0.002826, 0.018021, -0.027987),
    "yaw_axis": (0.0, 1.0, 0.000749),
    "pitch_origin": (-0.015, -0.015, -1.1e-05),
    "pitch_axis": (-1.0, 0.0, 0.0),
    "spur_ratio": 0.8,
    "bevel_ratio": 0.85714286,
}

HAND_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
HAND_MOTORS = (
    "thumb_CMC1", "thumb_CMC2", "thumb_MCP", "thumb_IP", "spread",
    "index_MCP", "index_PIP", "middle_MCP", "middle_PIP",
    "ring_MCP", "ring_PIP", "pinky_MCP", "pinky_PIP",
)  # fmt: skip
"""Motor order of the hand (the wrist motors 13 and 14 are position-controlled, not modelled)."""

HAND_GRAVITY = (0.0, 0.0, -9.81)
"""Gravity in the hand base frame [m/s²] when the hand is upright; on an arm, keep it live."""

_LINK_JOINTS = {
    "thumb": {"base": "CMC1", "2dof_joint": "CMC1", "proximal": "CMC2", "middle": "MCP", "distal": "IP"},
    "finger": {"base": "MCP", "2dof_joint": "MCP", "proximal": "MCP", "middle": "PIP", "distal": "DIP"},
}  # fmt: skip
"""Joint frame each link's mass is attached to."""


def hand_coupling() -> np.ndarray:
    """Joint angles (20: thumb CMC1, CMC2, MCP, IP; each finger spread, MCP, PIP, DIP) per motor."""
    C = np.zeros((20, 13))
    t = HAND_THUMB_TRANSMISSION
    C[0, 0] = t["r_motor"] / t["CMC1_pulley"]
    C[1, 1] = t["r_motor"] / t["CMC2_pulley"]
    C[2, 2] = -t["r_motor"] / t["MCP_c"]
    C[3, 3] = -t["r_motor"] / t["IP_c"]
    for k, finger in enumerate(("index", "middle", "ring", "pinky")):
        row, mcp, pip = 4 + 4 * k, 5 + 2 * k, 6 + 2 * k
        tr = HAND_FINGER_TRANSMISSIONS[finger]
        C[row, 4] = HAND_SPREAD_CORRECTION * HAND_SPREAD_RATIOS[finger]
        C[row + 1, mcp] = tr["r_pulley"] / tr["r_motor"]
        C[row + 2, pip] = -tr["r_motor"] / tr["c_param"]
        C[row + 3, pip] = -tr["r_motor"] / tr["c_param"]
    return C


def _digit_chain(digit: str) -> SerialChain:
    """One digit as a chain in the hand base frame; sites: the tip and each link's mass."""
    joints = HAND_JOINTS[digit]
    base = np.asarray(HAND_PALM_ORIGIN) + np.asarray(HAND_FINGER_BASES[digit])
    points, axes = [], []
    for _name, origin, axis in joints:
        base = base + np.asarray(origin)
        points.append(base.copy())
        axes.append(list(axis))
    sites: dict[str, tuple[int, Any]] = {
        "tip": (4, points[-1] + np.asarray(HAND_TIP_OFFSETS[digit]))
    }
    names = [j[0] for j in joints]
    mapping = _LINK_JOINTS["thumb" if digit == "thumb" else "finger"]
    for link, joint in mapping.items():
        k = names.index(joint)
        sites[f"{link}_cog"] = (k + 1, points[k] + np.asarray(HAND_LINK_COGS[f"{digit}_{link}"]))
    return SerialChain(["revolute"] * 4, axes, [list(p) for p in points], sites)


def hand(name: str = "hand", gravity: Any = None) -> Mechanism:
    """The hand as a robot: q holds the 13 motor angles [rad] in ``HAND_MOTORS`` order.

    Sites: ``"<digit>/tip"`` and ``"<digit>/<link>_cog"``. The wrist is rigid; on an arm whose
    orientation changes, compile with ``runtime=["*.gravity"]`` and set the gravity each step.
    """
    body = Assembly({d: (_digit_chain(d), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) for d in HAND_DIGITS})
    robot = Mechanism(name, model=LinearCoupling(body, hand_coupling()))
    robot.add_param(
        Param(
            "gravity",
            HAND_GRAVITY if gravity is None else gravity,
            unit="m/s^2",
            scope="design",
            bounds=(-np.inf, np.inf),
        )
    )
    for digit in HAND_DIGITS:
        mapping = _LINK_JOINTS["thumb" if digit == "thumb" else "finger"]
        for link in mapping:
            point = robot.point(f"{digit}/{link}_cog")
            robot.add(f"m_{digit}_{link}", PointMass(point, HAND_LINK_MASSES[f"{digit}_{link}"]))
    return robot


def hand_joint_angles(robot: Mechanism) -> Custom:
    """Coordinate of the hand's 20 joint angles [rad] (see ``hand_coupling`` for the order)."""
    return Custom(
        lambda motors, coupling: coupling @ motors,
        [Joint(slice(0, 13), unit="rad")],
        dim=20,
        unit="rad",
        params={"coupling": robot.model.coupling},
    )


def hand_joint_limit_spring(robot: Mechanism, stiffness: Any = HAND_LIMIT_STIFFNESS) -> LimitSpring:
    """Springs that push each of the 20 joints back inside its range."""
    names = [f"{d}_{j[0]}" for d in HAND_DIGITS for j in HAND_JOINTS[d]]
    limits = [
        HAND_JOINT_LIMITS[n if not n.endswith("_Spread") else n[:-7] + "_spread"] for n in names
    ]
    return LimitSpring(
        hand_joint_angles(robot), stiffness, [lo for lo, _ in limits], [hi for _, hi in limits]
    )
