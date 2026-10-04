"""ROS 2, any distribution: the robot over the lab driver's topics, a digital twin and live
parameters. Uses rclpy and std_msgs, imported only when a ROS class is first used."""

import importlib
from typing import TYPE_CHECKING, Any

from .joint_io import TOPICS, JointIO

if TYPE_CHECKING:
    from .params import LiveParams
    from .plant import RosPlant
    from .twin import serve

_LAZY = {"RosPlant": ".plant", "LiveParams": ".params", "serve": ".twin"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TOPICS", "JointIO", "LiveParams", "RosPlant", "serve"]
