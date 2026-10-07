"""Crawling turtle: two crank motors coordinated by a virtual flywheel, and a VSA motor.

``robot`` and ``controller`` are the lab's; ``crawler`` is a simple body for the simulator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import Param
from ..core.registry import register
from ..core.signals import Signals
from ..hardware import HardwareProfile, Motor
from ..mechanisms import (
    ContactDamper,
    ContactFriction,
    ContactSpring,
    ForceSource,
    FrameRotation,
    Gravity,
    Inertance,
    Joint,
    LinearDamper,
    LinearSpring,
    Mechanism,
    PhaseSpring,
    PlaneDistance,
    PointMass,
    Ref,
    RotationalInertia,
    SpeedRegulator,
    Stack,
)
from ..models import Assembly, Direct, JointSpace, Passive, SerialChain

CRANKS = ("left", "right")
"""Order of the two crank motors in q and in the motor vector."""

MOTOR_SIGNS = (1.0, -1.0)
"""Sign of each crank motor against the gait convention (the right motor is mounted mirrored)."""

CONTROL_RATE = 450.0  # [Hz]

MOTOR_IDS = (1, 2)
"""Bus IDs of the two crank motors, left then right."""

VSA_ID = 0
"""Bus ID of the VSA motor, held at its start position."""

VSA_RANGE = (0.0, np.pi / 2)
"""Range of the VSA motor's angle [rad] (position-controlled, not part of the crank law)."""

VSA_RATE = np.radians(20.0)  # [rad/s], how fast the teleoperation moves the VSA angle

DEFAULTS = {
    "stiffness": 1.0,  # K [N·m/rad], crank to flywheel
    "damping": 0.0001,  # C [N·m·s/rad]
    "inertia": 0.1,  # J_v [kg·m²], virtual flywheel
    "flywheel_damping": 0.1,  # b_v [N·m·s/rad], speed regulator gain
    "speed": 0.1,  # ω̄ [rad/s], free-running flywheel speed
    "phase": 0.0,  # δ [rad], right crank behind the left one
    "ramp_time": 3.0,  # [s] to ramp the speed from zero
    "torque_bias": 0.0,  # [N·m] on both cranks
}
"""Default controller parameters (starting values, not tuned)."""

PHASE_SPRINGS = ("depth", "peak", "steer", "limit")
"""Optional controller parameters (depth of the stiffness swing, phase of its peak, steering,
largest stretch): give any and the cranks' springs follow the phase, as a ``PhaseSpring``."""


@register("hardware", "turtle.hardware")
def hardware() -> HardwareProfile:
    """The cranks' motors and the held VSA motor."""
    cranks = tuple(Motor(i, sign=sign) for i, sign in zip(MOTOR_IDS, MOTOR_SIGNS, strict=True))
    motors = (*cranks, Motor(VSA_ID, hold=True))
    return HardwareProfile(motors, rate=CONTROL_RATE)


@register("robot", "turtle.robot")
def robot(name: str = "turtle") -> Mechanism:
    """The two cranks as a robot in the gait convention: q = (left, right) [rad]."""
    return Mechanism(name, model=JointSpace(2))


@register("controller", "turtle.controller")
def controller(robot: Mechanism, name: str = "ctrl", **params: Any) -> Mechanism:
    """The virtual flywheel controller; each crank follows the flywheel phase through a spring.

    u_i = −K e_i − C ė_i + bias with e_left = q_left − φ, e_right = q_right − (φ − δ); the
    flywheel J_v φ̈ = Σ (K e_i + C ė_i) + b_v (ω_cmd − φ̇). Keyword arguments override ``DEFAULTS``;
    with any of ``PHASE_SPRINGS`` the springs follow the phase (K_i of the paper's potential).
    """
    p = {**DEFAULTS, **params}
    unknown = set(params) - set(DEFAULTS) - set(PHASE_SPRINGS)
    if unknown:
        known = sorted({*DEFAULTS, *PHASE_SPRINGS})
        raise TypeError(f"unknown turtle parameters {sorted(unknown)}; known: {known}")
    phased = {k: params[k] for k in PHASE_SPRINGS if params.get(k) is not None}
    ctrl = Mechanism(name)
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", Inertance(phi, p["inertia"]))
    phase = Ref("phase", 1, value=p["phase"], unit="rad")
    phases = {"left": (phi, 1.0), "right": (phi - phase, -1.0)}
    errors = {"left": robot.joint(0) - phi, "right": robot.joint(1) - (phi - phase)}
    for crank, e in errors.items():
        if phased:
            at, side = phases[crank]
            spring: Any = PhaseSpring(Stack(e, at), p["stiffness"], side=side, **phased)
        else:
            spring = LinearSpring(e, p["stiffness"])
        ctrl.add(f"spring_{crank}", spring)
        ctrl.add(f"damper_{crank}", LinearDamper(e, p["damping"]))
    ctrl.add("drive", SpeedRegulator(phi, p["flywheel_damping"], p["speed"], p["ramp_time"]))
    ctrl.add("bias", ForceSource(Joint([0, 1], unit="rad"), [p["torque_bias"]] * 2))
    return ctrl


@register("initial_state", "turtle.initial_state")
def initial_state(meas: Signals) -> np.ndarray:
    """Flywheel state at start: the left crank's angle, at rest."""
    return np.array([meas["motor_position"][0], 0.0])


CRAWLER = {
    "mass": 1.0,  # [kg] the body, with the cranks
    "inertia": (3.6e-3, 7.8e-3, 1.08e-2),  # [kg·m²] about its center: x forward, y left, z up
    "belly": (0.12, 0.07, 0.03),  # [m] its four underside corners: ±x, ±y, and z below the center
    "axle": (0.0, 0.11, 0.0),  # [m] the left crank's axis in the body (the right one is mirrored)
    "crank_radius": 0.05,  # [m] from the axis to the foot, which hangs at the bottom at q = 0
    "crank_inertia": 2e-3,  # [kg·m²] of each crank about its axis
    "crank_damping": 0.02,  # [N·m·s/rad] on each crank
    "gravity": (0.0, 0.0, -9.81),  # [m/s²]
    "stiffness": 5e3,  # [N/m] of the ground under each foot and corner
    "ground_damping": 100.0,  # [N·s/m] of the ground
    "friction": 0.8,  # of the feet on the ground
    "belly_friction": 0.03,  # of the underside corners on the ground
    "slip_speed": 1e-3,  # [m/s] below which the friction falls linearly
    "smoothing": 0.0,  # [m] width over which the ground's edge is rounded (0: sharp)
    "efficiency": 1.0,  # torque a crank delivers per commanded one
}
"""Constants of ``crawler`` (SI). A simple crawler, not the lab's turtle: placeholders."""


def crawler(name: str = "crawler", **params: Any) -> Mechanism:
    """A simple crawler for the simulator, not the lab's turtle: its constants are placeholders.

    A floating body on the ground z = 0 with gravity, and two cranks turning about its lateral
    axis, each with a foot on the ground. ``q`` = (x, y, z, w, x, y, z of the body's quaternion,
    left crank, right crank); the motors are the two cranks, as for ``robot``, in the same gait
    convention. Keyword arguments override ``CRAWLER``.
    """
    p = {**CRAWLER, **params}
    unknown = set(params) - set(CRAWLER)
    if unknown:
        raise TypeError(f"unknown crawler parameters {sorted(unknown)}; known: {sorted(CRAWLER)}")
    a, b, c = p["belly"]
    corners = {"fl": (a, b, -c), "fr": (a, -b, -c), "bl": (-a, b, -c), "br": (-a, -b, -c)}
    sites = {"centre": (1, (0.0, 0.0, 0.0)), **{f"belly_{k}": (1, v) for k, v in corners.items()}}
    body = SerialChain(["floating"], axes=[None], points=[(0.0, 0.0, 0.0)], sites=sites)
    axle = np.array(p["axle"], dtype=float)
    mirrored = axle * [1.0, -1.0, 1.0]
    parts: dict[str, tuple[Any, ...]] = {"body": (body, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}
    for side, at in (("left", axle), ("right", mirrored)):
        crank = SerialChain(
            ["revolute"],
            axes=[(0.0, 1.0, 0.0)],
            points=[(0.0, 0.0, 0.0)],
            sites={"axis": (1, (0.0, 0.0, 0.0)), "foot": (1, (0.0, 0.0, -p["crank_radius"]))},
        )
        parts[side] = (crank, at, (0.0, 0.0, 0.0), "body/centre")
    assembly = Assembly(parts)
    motors = {"body": Passive(body.space.nv), "left": Direct(p["efficiency"])}
    motors["right"] = Direct(p["efficiency"])  # its own copy: each crank has its own efficiency
    robot = Mechanism(name, model=assembly, actuation=assembly.stacked_actuation(motors))
    robot.add_param(Param("gravity", p["gravity"], unit="m/s^2", scope="design"))
    robot.add("body_mass", PointMass(robot.point("body/centre"), p["mass"]))
    robot.add(
        "body_inertia",
        RotationalInertia(FrameRotation(assembly, "body/centre"), p["inertia"]),
    )
    crank_inertia = p["crank_inertia"] * np.array([0.5, 1.0, 0.5])  # a disc about its axis
    for i, side in enumerate(CRANKS):
        robot.add(
            f"{side}_inertia",
            RotationalInertia(FrameRotation(assembly, f"{side}/axis"), crank_inertia),
        )
        robot.add(
            f"{side}_damper", LinearDamper(robot.joint(body.space.nq + i), p["crank_damping"])
        )
    robot.add("gravity", Gravity(robot))

    normal = Param("normal", (0.0, 0.0, 1.0), scope="episode")
    origin = Param("origin", (0.0, 0.0, 0.0), unit="m", scope="episode")
    stiffness = Param("stiffness", p["stiffness"], unit="N/m", scope="stage")
    damping = Param("damping", p["ground_damping"], unit="N*s/m", scope="stage")
    speed = Param("slip_speed", p["slip_speed"], unit="m/s", scope="episode")
    edge = Param("smoothing", p["smoothing"], unit="m", scope="episode")
    friction = {
        "feet": Param("friction", p["friction"], scope="stage"),
        "belly": Param("belly_friction", p["belly_friction"], scope="stage"),
    }
    contacts = {f"{side}_foot": (robot.point(f"{side}/foot"), "feet") for side in CRANKS}
    contacts |= {f"belly_{k}": (robot.point(f"body/belly_{k}"), "belly") for k in corners}
    for key, (point, rubs) in contacts.items():
        ground = PlaneDistance(point, normal, origin)
        robot.add(key, ContactSpring(ground, stiffness, edge))
        robot.add(f"{key}_damper", ContactDamper(ground, damping, edge))
        robot.add(
            f"{key}_friction", ContactFriction(ground, stiffness, friction[rubs], speed, edge)
        )
    return robot
