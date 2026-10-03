"""Bimanual Helyx: two 290/145/145 mm soft arms side by side on one frame, pointing up."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..control.output import Pretension, TorqueLimit, TorqueOffset
from ..core.params import Param
from ..mechanisms import Gravity, LinearDamper, LinearSpring, Mechanism, PointMass
from ..models import Assembly
from . import helyx

GEOMETRY = "290-145-145"
ARMS = ("right", "left")
"""Order of the arms in q and in the motor vector: right first, then left."""

BASE_POSITIONS = {"right": (0.125, 0.0, 0.0), "left": (-0.125, 0.0, 0.0)}
"""Arm bases on the frame [m]; both arms keep the frame's orientation."""

GRAVITY = (0.0, 0.0, -9.81)
"""Gravity in the frame [m/s²]: the arms point up."""

SEGMENT_MASS = 0.030
"""Lumped mass [kg] per 145 mm of arm, at each section's centre (60, 30, 30 g)."""

EFFICIENCY = 0.12
"""Delivered over commanded motor torque, calibrated against a load cell."""

STIFFNESS = {
    "right": [
        234.0878082533943, 199.5160032646154, 410.00652675113776,
        391.7368779384143, 475.48588391722103, 831.8400830004166,
        445.3460323321939, 517.897763947965, 938.3750233876958,
    ],
    "left": [
        248.00753057771666, 270.89848713023935, 451.2551978336581,
        631.3985192570362, 619.4268893527735, 1205.3960742665633,
        779.3445118931818, 801.3675528661909, 1478.1926805912778,
    ],
}  # fmt: skip
"""Diagonal stiffness in Δ of each arm [N/m], physical (referred to the delivered torque)."""

DAMPING = {
    "right": [
        125.02998503799478, 104.47366885419642, 205.44115492186452,
        139.25665709952858, 148.06713140688805, 284.3437615659688,
        135.32320950338075, 141.0787208817908, 280.4637921085779,
    ],
    "left": [
        110.79813679873712, 109.05602570818645, 169.78235387942692,
        135.8665659409267, 136.7603243830659, 262.78637230057535,
        129.65653312195928, 128.6226181777195, 237.87572608992838,
    ],
}  # fmt: skip
"""Diagonal damping in Δ of each arm [N·s/m], physical."""

ENCODER_SIGN = 1.0
"""Sign of the motor encoders against the library convention (θ > 0 pulls a tendon)."""

CONTROL_RATE = 500.0  # [Hz]

TORQUE_OFFSET = 0.03  # [N·m] on every motor
PRETENSION_WEIGHTS = 0.03  # [N·m/rad] per motor, only on tendons released past the threshold
PRETENSION_THRESHOLD = np.radians(30.0)  # [rad]
TORQUE_LIMIT = 0.5  # [N·m]


def output_stage() -> list[Any]:
    """The arms' output stage: a torque offset, a soft stop on slack tendons, a torque clip."""
    return [
        TorqueOffset(TORQUE_OFFSET),
        Pretension(np.full(18, PRETENSION_WEIGHTS), PRETENSION_THRESHOLD),
        TorqueLimit(TORQUE_LIMIT),
    ]


def arms(name: str = "arms", gravity: Any = None) -> Mechanism:
    """Both arms as one robot: q = [Δ_right, Δ_left] (18), motors right then left.

    Points: ``robot.point("right", s=1.0)`` is the right tip. The tendons carry ``EFFICIENCY``.
    """
    body = Assembly(
        {arm: (helyx.model(GEOMETRY), BASE_POSITIONS[arm], (0.0, 0.0, 0.0)) for arm in ARMS}
    )
    actuation = body.stacked_actuation({arm: helyx.tendons(EFFICIENCY) for arm in ARMS})
    robot = Mechanism(name, model=body, actuation=actuation)
    g = GRAVITY if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    L0 = np.array(helyx.GEOMETRIES[GEOMETRY]["L0"])
    b = np.concatenate([[0.0], np.cumsum(L0)]) / L0.sum()
    for arm in ARMS:
        for i, length in enumerate(L0):
            mass = SEGMENT_MASS * length / helyx.REFERENCE_LENGTH
            robot.add(f"{arm}_m{i + 1}", PointMass(robot.point(arm, s=(b[i] + b[i + 1]) / 2), mass))
    return robot


def add_dynamics(robot: Mechanism, stiffness: Any = None, damping: Any = None) -> Mechanism:
    """Give both arms their identified stiffness and damping in Δ, and gravity, for simulation."""
    for i, arm in enumerate(ARMS):
        delta = robot.joint(slice(9 * i, 9 * i + 9))
        K = (stiffness or STIFFNESS)[arm]
        D = (damping or DAMPING)[arm]
        robot.add(
            f"{arm}_stiffness",
            LinearSpring(delta, Param("stiffness", K, unit="N/m", scope="design")),
        )
        robot.add(
            f"{arm}_damping", LinearDamper(delta, Param("damping", D, unit="N*s/m", scope="design"))
        )
    robot.add("gravity", Gravity(robot))
    return robot
