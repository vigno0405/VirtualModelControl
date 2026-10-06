"""Virtual Model Control: robots and controllers as mechanisms of springs, dampers, inertances."""

import importlib
from typing import TYPE_CHECKING, Any

from . import identification, optimization, sim
from .compiler import Compiled, compile
from .control import VMCController
from .core import SO2, Euclidean, Param, ParamSet, Product, Signals, register
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
    ContactSpring,
    Custom,
    CylinderDistance,
    ForceSource,
    FramePoint,
    Gated,
    GaussianSpring,
    Gravity,
    GravityCompensation,
    Inertance,
    Joint,
    LimitSpring,
    LinearDamper,
    LinearSpring,
    Mechanism,
    Norm,
    PlaneDistance,
    PointMass,
    PolynomialSpring,
    Projection,
    Ref,
    SigmoidSpring,
    SpeedRegulator,
    SphereDistance,
    Stack,
    TanhDamper,
    TanhSpring,
)
from .models import Efficiency, Kinematics
from .system import VirtualMechanismSystem

try:
    from ._version import __version__
except ImportError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from . import config, hardware, viz


def __getattr__(name: str) -> Any:
    # ``vmc.viz`` (matplotlib), ``vmc.hardware`` and ``vmc.config`` load
    # only when first used.
    if name in ("viz", "hardware", "config"):
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
    "ContactSpring",
    "Custom",
    "CylinderDistance",
    "Dynamics",
    "Efficiency",
    "Euclidean",
    "ForceSource",
    "FramePoint",
    "Gated",
    "GaussianSpring",
    "Gravity",
    "GravityCompensation",
    "Inertance",
    "Joint",
    "Kinematics",
    "LimitSpring",
    "LinearDamper",
    "LinearSpring",
    "Mechanism",
    "Norm",
    "Param",
    "ParamSet",
    "PlaneDistance",
    "PointMass",
    "PolynomialSpring",
    "Product",
    "Projection",
    "Ref",
    "SigmoidSpring",
    "Signals",
    "SpeedRegulator",
    "SphereDistance",
    "Stack",
    "TanhDamper",
    "TanhSpring",
    "VMCController",
    "VirtualMechanismSystem",
    "__version__",
    "compile",
    "compile_dynamics",
    "config",
    "hardware",
    "identification",
    "optimization",
    "register",
    "sim",
    "viz",
]
