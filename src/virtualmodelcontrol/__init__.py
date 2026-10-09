"""Virtual Model Control: robots and controllers as mechanisms of springs, dampers, inertances."""

import importlib
from typing import TYPE_CHECKING, Any

from . import adaptation, estimation, identification, math, optimization, sim
from .compiler import Compiled, compile
from .control import VMCController
from .core import SO2, Euclidean, Param, ParamSet, Product, Quaternion, Signals, register
from .dynamics import Dynamics, compile_dynamics
from .mechanisms import (
    BoxDistance,
    CapsuleDistance,
    ConstrainedGaussianSpring,
    ConstrainedLinearDamper,
    ConstrainedLinearSpring,
    ConstrainedTanhDamper,
    ConstrainedTanhSpring,
    ContactDamper,
    ContactFriction,
    ContactSpring,
    Custom,
    CylinderDistance,
    DiodeDamper,
    ForceSource,
    FramePoint,
    FrameRotation,
    FromFrame,
    Gated,
    GaussianSpring,
    Gravity,
    GravityCompensation,
    Inertance,
    InFrame,
    Joint,
    LimitSpring,
    LinearDamper,
    LinearSpring,
    Mechanism,
    Norm,
    OrientationError,
    PhaseSpring,
    PlaneDistance,
    PointMass,
    PolynomialSpring,
    Projection,
    Ref,
    RotationalInertia,
    SigmoidSpring,
    SpeedRegulator,
    SphereDistance,
    Stack,
    Sum,
    TanhDamper,
    TanhSpring,
    Time,
)
from .models import Efficiency, Kinematics, StaticFriction
from .system import VirtualMechanismSystem

try:
    from ._version import __version__
except ImportError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from . import config, hardware, testing, viz


def __getattr__(name: str) -> Any:
    # ``vmc.viz`` (matplotlib), ``vmc.hardware``, ``vmc.config`` and ``vmc.testing`` load
    # only when first used.
    if name in ("viz", "hardware", "config", "testing"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SO2",
    "BoxDistance",
    "CapsuleDistance",
    "Compiled",
    "ConstrainedGaussianSpring",
    "ConstrainedLinearDamper",
    "ConstrainedLinearSpring",
    "ConstrainedTanhDamper",
    "ConstrainedTanhSpring",
    "ContactDamper",
    "ContactFriction",
    "ContactSpring",
    "Custom",
    "CylinderDistance",
    "DiodeDamper",
    "Dynamics",
    "Efficiency",
    "Euclidean",
    "ForceSource",
    "FramePoint",
    "FrameRotation",
    "FromFrame",
    "Gated",
    "GaussianSpring",
    "Gravity",
    "GravityCompensation",
    "InFrame",
    "Inertance",
    "Joint",
    "Kinematics",
    "LimitSpring",
    "LinearDamper",
    "LinearSpring",
    "Mechanism",
    "Norm",
    "OrientationError",
    "Param",
    "ParamSet",
    "PhaseSpring",
    "PlaneDistance",
    "PointMass",
    "PolynomialSpring",
    "Product",
    "Projection",
    "Quaternion",
    "Ref",
    "RotationalInertia",
    "SigmoidSpring",
    "Signals",
    "SpeedRegulator",
    "SphereDistance",
    "Stack",
    "StaticFriction",
    "Sum",
    "TanhDamper",
    "TanhSpring",
    "Time",
    "VMCController",
    "VirtualMechanismSystem",
    "__version__",
    "adaptation",
    "compile",
    "compile_dynamics",
    "config",
    "estimation",
    "hardware",
    "identification",
    "math",
    "optimization",
    "register",
    "sim",
    "testing",
    "viz",
]
