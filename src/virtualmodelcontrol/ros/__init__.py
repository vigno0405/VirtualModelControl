"""ROS 2, any distribution: the robot over the lab driver's topics, a digital twin and live
parameters. Uses rclpy and std_msgs, imported only when a ROS class is first used.

Deprecated: the library is communication-agnostic, so this package leaves in 0.4.0; using its
names warns.
"""

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .joint_io import TOPICS, JointIO
    from .node import control
    from .params import LiveParams
    from .plant import RosPlant
    from .twin import serve

_LAZY = {
    "TOPICS": ".joint_io",
    "JointIO": ".joint_io",
    "RosPlant": ".plant",
    "LiveParams": ".params",
    "serve": ".twin",
    "control": ".node",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        warnings.warn(
            f"virtualmodelcontrol.ros.{name} is deprecated and leaves in 0.4.0: the library is "
            "communication-agnostic, so the robot's driver is used outside it",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TOPICS", "JointIO", "LiveParams", "RosPlant", "control", "serve"]
