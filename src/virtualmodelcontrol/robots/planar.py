"""Planar arms with fewer motors than coordinates.

A three-link arm with one passive joint, a five-link arm with three, and a three-section
continuum arm driven by two tendons.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..core.registry import register
from ..mechanisms import (
    FrameRotation,
    Gravity,
    LinearDamper,
    LinearSpring,
    Mechanism,
    PointMass,
    Ref,
    RotationalInertia,
)
from ..models import PCC, LinearCoupling, SerialChain, Underactuated

PRESETS: dict[str, dict[str, Any]] = {
    "three-link": {"lengths": (0.30,) * 3, "masses": (0.50,) * 3, "actuated": (0, 2)},
    "five-link": {"lengths": (0.20,) * 5, "masses": (0.30,) * 5, "actuated": (0, 4)},
}
"""Link lengths [m] and masses [kg], base to tip, and the joints with a motor (0-based). The
others are passive: joint 2 of the three-link arm; joints 2, 3 and 4 of the five-link arm."""

GRAVITY = (0.0, -9.81, 0.0)
"""Gravity in the arm's base frame [m/s²]: the arm moves in the x-y plane, y up."""

PASSIVE_STIFFNESS = 6.0  # [N·m/rad] torsional spring of each passive joint
PASSIVE_DAMPING = 0.40  # [N·m·s/rad] torsional damper of each passive joint


def model(lengths: Any) -> SerialChain:
    """The kinematics: revolute joints about z, links along x at q = 0 (q = relative angles).

    Sites: ``tip`` and the centre of each link, ``link1`` … ``linkN``.
    """
    length = np.asarray(lengths, dtype=float)
    x = np.concatenate([[0.0], np.cumsum(length)])  # [m] where each joint sits
    sites: dict[str, tuple[Any, ...]] = {"tip": (len(length), (x[-1], 0.0, 0.0))}
    for i in range(len(length)):
        sites[f"link{i + 1}"] = (i + 1, (x[i] + length[i] / 2, 0.0, 0.0))
    points = [(xi, 0.0, 0.0) for xi in x[:-1]]
    return SerialChain(["revolute"] * len(length), [(0.0, 0.0, 1.0)] * len(length), points, sites)


@register("robot", "planar.arm")
def arm(
    preset: str = "three-link",
    name: str = "arm",
    gravity: Any = None,
    *,
    lengths: Any = None,
    masses: Any = None,
    actuated: Any = None,
    rest: Any = None,
    efficiency: Any = 1.0,
) -> Mechanism:
    """The arm as a robot mechanism: q holds the joint angles [rad], the motors drive the
    ``actuated`` joints only, and every link is a uniform rod (a mass and the inertia m l²/12 at
    its centre).

    ``preset`` gives the defaults; ``lengths`` [m], ``masses`` [kg], ``actuated`` joints,
    ``gravity`` [m/s², base frame], ``rest`` (the passive joints' rest angles [rad], also the
    frozen controller's q̄: one value for every passive joint, or one per joint of the arm) and
    the motors' ``efficiency`` override them, for any number of links.
    """
    spec = PRESETS[preset]
    length = np.asarray(spec["lengths"] if lengths is None else lengths, dtype=float)
    mass = np.asarray(spec["masses"] if masses is None else masses, dtype=float)
    motors = list(spec["actuated"] if actuated is None else actuated)
    if length.shape != mass.shape:
        raise ValueError(f"{length.size} link lengths but {mass.size} masses")
    n = length.size
    rest_angles = np.broadcast_to(np.asarray(0.0 if rest is None else rest, dtype=float), (n,))
    robot = Mechanism(
        name,
        model=model(length),
        actuation=Underactuated.joints(n, motors, rest_angles, efficiency),
    )
    g = GRAVITY if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    for i in range(n):
        link = f"link{i + 1}"
        robot.add(f"m{i + 1}", PointMass(robot.point(link), mass[i]))
        rod = mass[i] * length[i] ** 2 / 12.0  # [kg·m²] about the centre, along the link's axis
        robot.add(f"I{i + 1}", RotationalInertia(FrameRotation(robot.model, link), (0.0, rod, rod)))
    return robot


def passive_joints(robot: Mechanism) -> list[int]:
    """The joints no motor drives (the rows of the input matrix B that are zero)."""
    B = np.asarray(robot.actuation.params["B"].value)
    return [int(i) for i in np.flatnonzero(~B.any(axis=1))]


@register("dynamics", "planar.add_dynamics")
def add_dynamics(
    robot: Mechanism, stiffness: Any = PASSIVE_STIFFNESS, damping: Any = PASSIVE_DAMPING
) -> Mechanism:
    """Give the arm its gravity and, on each passive joint, a torsional spring and damper.

    ``stiffness`` [N·m/rad] and ``damping`` [N·m·s/rad] are one value for every passive joint,
    or one per passive joint. The spring's rest angle is the arm's ``rest``.
    """
    passive = passive_joints(robot)
    joints = robot.joint(passive)
    rest = np.asarray(robot.actuation.params["q_rest"].value)[passive]
    reference = Ref("rest", len(passive), value=rest, unit="rad", scope="design")
    robot.add("spring", LinearSpring(joints - reference, stiffness))
    robot.add("damper", LinearDamper(joints, damping))
    robot.add("gravity", Gravity(robot))
    return robot


CONTINUUM = {
    "lengths": (0.25, 0.25, 0.25),  # [m]
    "masses": (0.25, 0.20, 0.15),  # [kg], at the section tips
    "section_radius": 0.02,  # [m]
    "offsets": (0.02, 0.02),  # [m] of each tendon from the axis, in the bending plane
    "depths": (3, 1),  # sections each tendon runs through, from the base
    "spool_radius": 0.01,  # [m]
    "stiffness": 6.0,  # [N·m/rad] bending stiffness of each section
    "damping": 0.40,  # [N·m·s/rad] bending damping of each section
}
"""The continuum arm's defaults: two tendons, one through all three sections and one through the
first only, so the bending of sections 2 against 3 has no motor."""

CONTINUUM_GRAVITY = (0.0, 0.0, -9.81)
"""Gravity in the continuum arm's base frame [m/s²]: it points up the z axis."""


def tendon_matrix(offsets: Any, depths: Any, spool_radius: Any, sections: int) -> np.ndarray:
    """The input matrix B of tendons on bending angles: tendon i, at ``offsets[i]`` [m] from the
    axis, wound on a spool of radius ``spool_radius`` [m], runs through the first ``depths[i]``
    sections; B[j, i] = −offset / spool_radius there and 0 elsewhere."""
    B = np.zeros((sections, len(offsets)))
    for i, (offset, depth) in enumerate(zip(offsets, depths, strict=True)):
        B[: int(depth), i] = -offset / spool_radius
    return B


@register("robot", "planar.continuum")
def continuum(
    name: str = "arm",
    gravity: Any = None,
    *,
    lengths: Any = None,
    masses: Any = None,
    section_radius: Any = None,
    offsets: Any = None,
    depths: Any = None,
    spool_radius: Any = None,
    rest: Any = None,
    efficiency: Any = 1.0,
) -> Mechanism:
    """A planar continuum arm of constant-curvature sections, driven by tendons.

    q holds the bending angle of each section [rad]; the arm bends in the x-z plane (x opposite to
    the sign of the soft arm's Δx), and a section of length L bent by φ is an arc of that
    length. The tendons (``offsets``, ``depths``, ``spool_radius``; see ``tendon_matrix``) are
    fewer than the sections. Each section's tip carries a mass and a rod's inertia m L² / 12.
    The defaults are ``CONTINUUM``.
    """
    spec = CONTINUUM
    length = np.asarray(spec["lengths"] if lengths is None else lengths, dtype=float)
    mass = np.asarray(spec["masses"] if masses is None else masses, dtype=float)
    radius = np.broadcast_to(
        np.asarray(spec["section_radius"] if section_radius is None else section_radius),
        length.shape,
    )
    n = length.size
    B = tendon_matrix(
        spec["offsets"] if offsets is None else offsets,
        spec["depths"] if depths is None else depths,
        spec["spool_radius"] if spool_radius is None else spool_radius,
        n,
    )
    coupling = np.zeros((3 * n, n))
    coupling[3 * np.arange(n), np.arange(n)] = -radius  # Δx = −d φ: an arc of bend φ
    rest_angles = np.broadcast_to(np.asarray(0.0 if rest is None else rest, dtype=float), (n,))
    body = LinearCoupling(PCC(length, radius), coupling)
    robot = Mechanism(name, model=body, actuation=Underactuated(B, rest_angles, efficiency))
    g = CONTINUUM_GRAVITY if gravity is None else gravity
    robot.add_param(Param("gravity", g, unit="m/s^2", scope="design", bounds=(-np.inf, np.inf)))
    sites = [f"seg{i + 1}" for i in range(n - 1)] + ["tip"]
    for i, site in enumerate(sites):
        robot.add(f"m{i + 1}", PointMass(robot.point(site), mass[i]))
        rod = mass[i] * length[i] ** 2 / 12.0  # [kg·m²], the section's own rotation
        robot.add(f"I{i + 1}", RotationalInertia(FrameRotation(body, site), (0.0, rod, 0.0)))
    return robot


@register("dynamics", "planar.add_continuum_dynamics")
def add_continuum_dynamics(
    robot: Mechanism, stiffness: Any = CONTINUUM["stiffness"], damping: Any = CONTINUUM["damping"]
) -> Mechanism:
    """Give the continuum arm its gravity and, on every section, a bending spring and damper.

    ``stiffness`` [N·m/rad] and ``damping`` [N·m·s/rad] are one value for every section, or one
    per section; the spring's rest is the arm's ``rest``.
    """
    bends = robot.joint(slice(0, robot.model.space.nq))
    rest = np.asarray(robot.actuation.params["q_rest"].value)
    reference = Ref("rest", rest.size, value=rest, unit="rad", scope="design")
    robot.add("spring", LinearSpring(bends - reference, stiffness))
    robot.add("damper", LinearDamper(bends, damping))
    robot.add("gravity", Gravity(robot))
    return robot
