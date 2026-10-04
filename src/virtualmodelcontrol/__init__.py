"""Virtual Model Control: robots and controllers as mechanisms of springs, dampers, inertances."""

import importlib
from typing import TYPE_CHECKING, Any

from . import sim
from .compiler import Compiled, compile
from .control import VMCController
from .core import SO2, Euclidean, Param, ParamSet, Product, Signals, register
from .dynamics import Dynamics, compile_dynamics
from .mechanisms import (
    ContactDamper,
    ContactSpring,
    Custom,
    ForceSource,
    FramePoint,
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
from .models import Kinematics
from .system import VirtualMechanismSystem

try:
    from ._version import __version__
except ImportError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from . import viz


def __getattr__(name: str) -> Any:
    # ``vmc.viz`` loads matplotlib only when first used.
    if name == "viz":
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SO2",
    "Compiled",
    "ContactDamper",
    "ContactSpring",
    "Custom",
    "Dynamics",
    "Euclidean",
    "ForceSource",
    "FramePoint",
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
    "register",
    "sim",
    "viz",
]
