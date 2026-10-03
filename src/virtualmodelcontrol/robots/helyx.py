"""Helyx soft arm: three tendon-driven PCC segments, in the lab's three geometries."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..mechanisms import Gravity, LinearDamper, LinearSpring, Mechanism, PointMass
from ..models import PCC, TendonTransmission

GEOMETRIES: dict[str, dict[str, Any]] = {
    "145-145-145": {"L0": (0.145, 0.145, 0.145), "gravity": (0.0, -9.81, 0.0)},
    "145-290-290": {"L0": (0.145, 0.290, 0.290), "gravity": (0.0, 0.0, 9.81)},
    "290-145-145": {"L0": (0.290, 0.145, 0.145), "gravity": (0.0, 0.0, -9.81)},
}
"""Segment rest lengths base to tip [m], and gravity in the base frame as mounted [m/s²]:
side-mounted, hanging, and pointing up."""

ENCODER_SIGN: dict[str, float] = {"145-145-145": -1.0, "145-290-290": -1.0, "290-145-145": 1.0}
"""Sign of each arm's motor encoders against the library convention (θ > 0 pulls a tendon)."""

SECTION_RADIUS = 0.030  # [m]
SPOOL_RADIUS = 0.003  # [m]
TENDON_ANGLES = np.radians([[0.0, 120.0, -120.0], [60.0, 180.0, -60.0], [150.0, 270.0, 30.0]])
"""Tendon angles around the section at each segment's base [rad]."""
SEGMENT_MASS = 0.040  # [kg] per REFERENCE_LENGTH of segment, lumped at the segment's midpoint
REFERENCE_LENGTH = 0.145  # [m]


def arm(geometry: str = "145-290-290", name: str = "arm", gravity: Any = None) -> Mechanism:
    """The arm as a robot mechanism: PCC model, tendons, lumped masses and a gravity Param.

    ``gravity`` [m/s², base frame] overrides the geometry's mounting.
    """
    spec = GEOMETRIES[geometry]
    model = PCC(spec["L0"], SECTION_RADIUS)
    robot = Mechanism(
        name, model=model, actuation=TendonTransmission(list(TENDON_ANGLES), SPOOL_RADIUS)
    )
    g = spec["gravity"] if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    b = model.breakpoints()
    for i, length in enumerate(spec["L0"]):
        mass = SEGMENT_MASS * (length / REFERENCE_LENGTH)
        robot.add(f"m{i + 1}", PointMass(robot.point(s=(b[i] + b[i + 1]) / 2), mass))
    return robot


SIM_STIFFNESS = np.array([
    233.9329569634883, 199.3719645292524, 410.00362883032545,
    391.6936839908778, 475.436633573179, 831.8383997691384,
    445.34472425862594, 517.8961860162584, 938.3740743393556,
]) / 0.12  # fmt: skip
"""Diagonal stiffness in Δ of the simulated 145-290-290 arm [N/m], referred to commanded torque
(identified values referred to delivered torque, divided by the efficiency 0.12)."""

SIM_DAMPING = np.array([
    125.13438533964865, 104.5563494024772, 205.43182586195965,
    139.28436710395044, 148.1129490938869, 284.33900221754124,
    135.3243281016147, 141.08179561149416, 280.464403371687,
]) / 0.12  # fmt: skip
"""Diagonal damping in Δ of the simulated 145-290-290 arm [N·s/m], referred like the stiffness."""


def add_dynamics(robot: Mechanism, stiffness: Any = None, damping: Any = None) -> Mechanism:
    """Give the arm its physical stiffness and damping in Δ and gravity, for simulation.

    ``stiffness`` [N/m] and ``damping`` [N·s/m] are per-axis (9 values); the defaults are the
    simulated arm's.
    """
    delta = robot.joint(slice(0, 9))
    K = SIM_STIFFNESS if stiffness is None else stiffness
    D = SIM_DAMPING if damping is None else damping
    robot.add("stiffness", LinearSpring(delta, Param("stiffness", K, unit="N/m", scope="design")))
    robot.add("damping", LinearDamper(delta, Param("damping", D, unit="N*s/m", scope="design")))
    robot.add("gravity", Gravity(robot))
    return robot
