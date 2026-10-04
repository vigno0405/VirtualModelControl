"""Helyx soft arm: three tendon-driven PCC segments, in the lab's three geometries."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..control.output import Pretension
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


PRETENSION_WEIGHTS = 0.010  # [N·m/rad] per motor, on the real arm only


def output_stage(motors: int = 9) -> list[Any]:
    """The real arm's output stage: a linear pretension W ∘ θ on every motor."""
    return [Pretension(np.full(motors, PRETENSION_WEIGHTS))]


EFFICIENCY = 0.12
"""Delivered over commanded motor torque of the tendon transmission, calibrated against a load
cell."""


def model(
    geometry: str = "145-290-290", *, lengths: Any = None, section_radius: Any = SECTION_RADIUS
) -> PCC:
    """The PCC model of an arm; ``lengths`` [m], base to tip, replace the geometry's."""
    return PCC(GEOMETRIES[geometry]["L0"] if lengths is None else lengths, section_radius)


def tendons(
    efficiency: Any = EFFICIENCY, *, angles: Any = TENDON_ANGLES, spool_radius: Any = SPOOL_RADIUS
) -> TendonTransmission:
    """The tendon transmission of one arm: ``angles`` [rad] holds a row of tendons per segment."""
    return TendonTransmission(list(np.asarray(angles, dtype=float)), spool_radius, efficiency)


def arm(
    geometry: str = "145-290-290",
    name: str = "arm",
    gravity: Any = None,
    *,
    lengths: Any = None,
    section_radius: Any = SECTION_RADIUS,
    spool_radius: Any = SPOOL_RADIUS,
    tendon_angles: Any = TENDON_ANGLES,
    masses: Any = None,
    efficiency: Any = EFFICIENCY,
) -> Mechanism:
    """The arm as a robot mechanism: PCC model, tendons, lumped masses and a gravity Param.

    ``geometry`` gives the defaults; ``lengths`` [m], ``section_radius`` [m], ``spool_radius``
    [m], ``tendon_angles`` [rad], ``masses`` [kg, one per segment], ``gravity`` [m/s², base
    frame] and the transmission's ``efficiency`` override them, for any number of segments.
    """
    spec = GEOMETRIES[geometry]
    L0 = spec["L0"] if lengths is None else lengths
    if len(tendon_angles) != len(L0):
        raise ValueError(
            f"{len(L0)} segments need {len(L0)} rows of tendon_angles, got {len(tendon_angles)}"
        )
    if masses is None:
        masses = [SEGMENT_MASS * (length / REFERENCE_LENGTH) for length in L0]
    robot = Mechanism(
        name,
        model=model(lengths=L0, section_radius=section_radius),
        actuation=tendons(efficiency, angles=tendon_angles, spool_radius=spool_radius),
    )
    g = spec["gravity"] if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    b = robot.model.breakpoints()
    for i, mass in enumerate(masses):
        robot.add(f"m{i + 1}", PointMass(robot.point(s=(b[i] + b[i + 1]) / 2), mass))
    return robot


SIM_STIFFNESS = np.array([
    233.9329569634883, 199.3719645292524, 410.00362883032545,
    391.6936839908778, 475.436633573179, 831.8383997691384,
    445.34472425862594, 517.8961860162584, 938.3740743393556,
])  # fmt: skip
"""Diagonal stiffness in Δ of the simulated 145-290-290 arm [N/m]: the identified, physical
values (the transmission's ``EFFICIENCY`` scales the commanded torque, not these)."""

SIM_DAMPING = np.array([
    125.13438533964865, 104.5563494024772, 205.43182586195965,
    139.28436710395044, 148.1129490938869, 284.33900221754124,
    135.3243281016147, 141.08179561149416, 280.464403371687,
])  # fmt: skip
"""Diagonal damping in Δ of the simulated 145-290-290 arm [N·s/m], physical like the stiffness."""


def add_dynamics(robot: Mechanism, stiffness: Any = None, damping: Any = None) -> Mechanism:
    """Give the arm its physical stiffness and damping in Δ and gravity, for simulation.

    ``stiffness`` [N/m] and ``damping`` [N·s/m] are per-axis (3 values per segment); the
    defaults are the simulated three-segment arm's, so other arms must give their own.
    """
    n = robot.model.space.nq
    if n != SIM_STIFFNESS.size and (stiffness is None or damping is None):
        raise ValueError(f"an arm with {n // 3} segments needs its own stiffness and damping")
    delta = robot.joint(slice(0, n))
    K = SIM_STIFFNESS if stiffness is None else stiffness
    D = SIM_DAMPING if damping is None else damping
    robot.add("stiffness", LinearSpring(delta, Param("stiffness", K, unit="N/m", scope="design")))
    robot.add("damping", LinearDamper(delta, Param("damping", D, unit="N*s/m", scope="design")))
    robot.add("gravity", Gravity(robot))
    return robot
