"""Crawling turtle: two crank motors coordinated by a virtual flywheel, and a VSA motor."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.registry import register
from ..core.signals import Signals
from ..hardware import HardwareProfile, Motor
from ..mechanisms import (
    ForceSource,
    Inertance,
    Joint,
    LinearDamper,
    LinearSpring,
    Mechanism,
    Ref,
    SpeedRegulator,
)
from ..models import JointSpace

CRANKS = ("left", "right")
"""Order of the two crank motors in q and in the motor vector."""

MOTOR_SIGNS = (1.0, -1.0)
"""Sign of each crank motor against the gait convention (the right motor is mounted mirrored)."""

CONTROL_RATE = 450.0  # [Hz]

MOTOR_IDS = (1, 2)
"""Bus IDs of the two crank motors (XC330-T288), left then right."""

VSA_ID = 0
"""Bus ID of the VSA motor, held at its start position."""

BAUDRATE = 4_000_000  # [bit/s]

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


@register("hardware", "turtle.hardware")
def hardware() -> HardwareProfile:
    """The cranks' motors and the held VSA motor, and their bus."""
    cranks = tuple(Motor(i, sign=sign) for i, sign in zip(MOTOR_IDS, MOTOR_SIGNS, strict=True))
    motors = (*cranks, Motor(VSA_ID, mode="hold"))
    return HardwareProfile(motors, baudrate=BAUDRATE, rate=CONTROL_RATE)


@register("robot", "turtle.robot")
def robot(name: str = "turtle") -> Mechanism:
    """The two cranks as a robot in the gait convention: q = (left, right) [rad]."""
    return Mechanism(name, model=JointSpace(2))


@register("controller", "turtle.controller")
def controller(robot: Mechanism, name: str = "ctrl", **params: Any) -> Mechanism:
    """The virtual flywheel controller; each crank follows the flywheel phase through a spring.

    u_i = −K e_i − C ė_i + bias with e_left = q_left − φ, e_right = q_right − (φ − δ); the
    flywheel J_v φ̈ = Σ (K e_i + C ė_i) + b_v (ω_cmd − φ̇). Keyword arguments override ``DEFAULTS``.
    """
    p = {**DEFAULTS, **params}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown turtle parameters {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    ctrl = Mechanism(name)
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", Inertance(phi, p["inertia"]))
    phase = Ref("phase", 1, value=p["phase"], unit="rad")
    errors = {"left": robot.joint(0) - phi, "right": robot.joint(1) - (phi - phase)}
    for crank, e in errors.items():
        ctrl.add(f"spring_{crank}", LinearSpring(e, p["stiffness"]))
        ctrl.add(f"damper_{crank}", LinearDamper(e, p["damping"]))
    ctrl.add("drive", SpeedRegulator(phi, p["flywheel_damping"], p["speed"], p["ramp_time"]))
    ctrl.add("bias", ForceSource(Joint([0, 1], unit="rad"), [p["torque_bias"]] * 2))
    return ctrl


@register("initial_state", "turtle.initial_state")
def initial_state(meas: Signals) -> np.ndarray:
    """Flywheel state at start: the left crank's angle, at rest."""
    return np.array([meas["motor_position"][0], 0.0])
