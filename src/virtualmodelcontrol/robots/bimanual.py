"""Bimanual Helyx: two 290/145/145 mm soft arms side by side on one frame, pointing up."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..control.output import Pretension, TorqueLimit, TorqueOffset
from ..core.params import Param
from ..hardware import HardwareProfile, Motor
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
"""Delivered over commanded motor torque, calibrated against a load cell: for estimates of
external forces. Not the default (1), because the stiffness and damping are referred to the
commanded torque."""

STIFFNESS = {
    "right": np.array([
        234.0878082533943, 199.5160032646154, 410.00652675113776,
        391.7368779384143, 475.48588391722103, 831.8400830004166,
        445.3460323321939, 517.897763947965, 938.3750233876958,
    ]) / 0.12,
    "left": np.array([
        248.00753057771666, 270.89848713023935, 451.2551978336581,
        631.3985192570362, 619.4268893527735, 1205.3960742665633,
        779.3445118931818, 801.3675528661909, 1478.1926805912778,
    ]) / 0.12,
}  # fmt: skip
"""Diagonal stiffness in Δ of each arm [N/m], referred to the commanded torque (identified values
referred to the delivered torque, divided by the efficiency 0.12 they were identified with)."""

DAMPING = {
    "right": np.array([
        125.02998503799478, 104.47366885419642, 205.44115492186452,
        139.25665709952858, 148.06713140688805, 284.3437615659688,
        135.32320950338075, 141.0787208817908, 280.4637921085779,
    ]) / 0.12,
    "left": np.array([
        110.79813679873712, 109.05602570818645, 169.78235387942692,
        135.8665659409267, 136.7603243830659, 262.78637230057535,
        129.65653312195928, 128.6226181777195, 237.87572608992838,
    ]) / 0.12,
}  # fmt: skip
"""Diagonal damping in Δ of each arm [N·s/m], referred like the stiffness."""

ENCODER_SIGN = 1.0
"""Sign of the motor encoders against the library convention (θ > 0 pulls a tendon)."""

CONTROL_RATE = 500.0  # [Hz]

MOTOR_IDS = (1, 3, 2, 5, 9, 7, 4, 8, 6, 10, 12, 11, 14, 18, 16, 13, 17, 15)
"""Bus IDs of the 18 XL330-M288 motors in the motor order: the right arm's nine, then the left's."""

HOME = {
    1: 2024, 2: 2687, 3: 798, 4: 2248, 5: 2179, 6: 911, 7: 2828, 8: 2935, 9: 828,
    10: 2789, 11: 2999, 12: 3593, 13: 2373, 14: 2068, 15: 2454, 16: 1395, 17: 2207, 18: 897,
}  # fmt: skip
"""Home pose of each motor [absolute ticks] by ID, recorded on the robot."""

BAUDRATE = 3_000_000  # [bit/s]
KT = 0.00115  # [N·m per unit of goal current], the constant the efficiency was calibrated with

TORQUE_OFFSET = 0.03  # [N·m] on every motor
PRETENSION_WEIGHTS = 0.03  # [N·m/rad] per motor, only on tendons released past the threshold
PRETENSION_THRESHOLD = np.radians(30.0)  # [rad]
TORQUE_LIMIT = 0.5  # [N·m]


def hardware() -> HardwareProfile:
    """The two arms' motors and their bus."""
    motors = tuple(Motor(i, "XL330-M288", ENCODER_SIGN, kt=KT, home=HOME[i]) for i in MOTOR_IDS)
    return HardwareProfile(motors, baudrate=BAUDRATE, rate=CONTROL_RATE)


def output_stage(motors: int = 18) -> list[Any]:
    """The arms' output stage: a torque offset, a soft stop on slack tendons, a torque clip."""
    return [
        TorqueOffset(TORQUE_OFFSET),
        Pretension(np.full(motors, PRETENSION_WEIGHTS), PRETENSION_THRESHOLD),
        TorqueLimit(TORQUE_LIMIT),
    ]


def _by_arm(value: Any, default: Any) -> dict[str, Any]:
    """One value for both arms, or a dict by arm; arms left out keep the default."""
    if isinstance(value, dict):
        return {arm: value.get(arm, default) for arm in ARMS}
    return {arm: default if value is None else value for arm in ARMS}


def arms(
    name: str = "arms",
    gravity: Any = None,
    *,
    lengths: Any = None,
    base_positions: Any = None,
    efficiency: Any = 1.0,
    segment_mass: float = SEGMENT_MASS,
    section_radius: Any = helyx.SECTION_RADIUS,
    spool_radius: Any = helyx.SPOOL_RADIUS,
    tendon_angles: Any = None,
) -> Mechanism:
    """Both arms as one robot: q = [Δ_right, Δ_left] (18), motors right then left.

    Points: ``robot.point("right", s=1.0)`` is the right tip. ``lengths`` [m] (one tuple for
    both arms, or a dict by arm), ``base_positions`` [m] (a dict by arm; arms left out keep
    their defaults), ``efficiency`` (1 by default),
    ``segment_mass`` [kg per 145 mm], the radii [m] and ``tendon_angles`` [rad] (one array for
    both arms, or a dict by arm) override the defaults.
    """
    per_arm = _by_arm(lengths, helyx.GEOMETRIES[GEOMETRY]["L0"])
    angles = _by_arm(tendon_angles, helyx.TENDON_ANGLES)
    bases = {**BASE_POSITIONS, **(base_positions or {})}
    model = {arm: helyx.model(lengths=per_arm[arm], section_radius=section_radius) for arm in ARMS}
    body = Assembly({arm: (model[arm], bases[arm], (0.0, 0.0, 0.0)) for arm in ARMS})
    for arm in ARMS:
        if len(angles[arm]) != len(per_arm[arm]):
            raise ValueError(f"the {arm} arm needs one row of tendon_angles per segment")
    drive = {
        arm: helyx.tendons(efficiency, angles=angles[arm], spool_radius=spool_radius)
        for arm in ARMS
    }
    robot = Mechanism(name, model=body, actuation=body.stacked_actuation(drive))
    g = GRAVITY if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    for arm in ARMS:
        b = model[arm].breakpoints()
        for i, length in enumerate(per_arm[arm]):
            mass = segment_mass * length / helyx.REFERENCE_LENGTH
            robot.add(f"{arm}_m{i + 1}", PointMass(robot.point(arm, s=(b[i] + b[i + 1]) / 2), mass))
    return robot


def add_dynamics(robot: Mechanism, stiffness: Any = None, damping: Any = None) -> Mechanism:
    """Give both arms their identified stiffness and damping in Δ, and gravity, for simulation.

    ``stiffness`` [N/m] and ``damping`` [N·s/m] are dicts by arm (9 values each); an arm left out
    keeps its identified values.
    """
    stiffness = {**STIFFNESS, **(stiffness or {})}
    damping = {**DAMPING, **(damping or {})}
    offset = 0
    for arm in ARMS:
        n = robot.model.parts[arm].space.nq
        delta = robot.joint(slice(offset, offset + n))
        offset += n
        K, D = stiffness[arm], damping[arm]
        robot.add(
            f"{arm}_stiffness",
            LinearSpring(delta, Param("stiffness", K, unit="N/m", scope="design")),
        )
        robot.add(
            f"{arm}_damping", LinearDamper(delta, Param("damping", D, unit="N*s/m", scope="design"))
        )
    robot.add("gravity", Gravity(robot))
    return robot
