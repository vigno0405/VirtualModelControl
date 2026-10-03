"""Virtual Model Control: robots and controllers as mechanisms of springs, dampers, inertances."""

from . import sim
from .compiler import Compiled, compile
from .control import VMCController
from .core import SO2, Euclidean, Param, ParamSet, Product, Signals, register
from .dynamics import Dynamics, compile_dynamics
from .mechanisms import (
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
    PointMass,
    PolynomialSpring,
    Projection,
    Ref,
    SigmoidSpring,
    SpeedRegulator,
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

__all__ = [
    "SO2",
    "Compiled",
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
    "PointMass",
    "PolynomialSpring",
    "Product",
    "Projection",
    "Ref",
    "SigmoidSpring",
    "Signals",
    "SpeedRegulator",
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
]
