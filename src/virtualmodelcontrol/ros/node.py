"""The controller node: a controller on the robot behind the driver's topics, in real time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from std_msgs.msg import String

from ..control.blending import SwapController
from ..hardware.profile import HardwareProfile
from ..sim.run import RunLog, WallClock, run
from .joint_io import TOPICS
from .params import LiveParams
from .plant import RosPlant


def control(
    controller: Any,
    profile: HardwareProfile,
    *,
    duration: float | None = None,
    stale: float | None = 0.05,
    swaps: Mapping[str, Any] | None = None,
    swap_time: float = 1.0,
    name: str = "vmc_controller",
) -> RunLog:
    """Run ``controller`` at the profile's rate on the robot behind the driver's topics.

    Its live Params are ROS parameters of the node ``name`` (``ros2 param set`` changes them
    between steps). With ``swaps``, controllers built in advance by name, a ``String`` naming
    one on the swap topic blends to it over ``swap_time`` [s] (and the parameters are not
    exposed). The run lasts ``duration`` [s] or until Ctrl-C; readings older than ``stale`` [s]
    send zero torque, and the robot gets zero torque at the end. Returns the log.
    """
    plant = RosPlant(profile, name=name)
    try:
        plant.wait()
        if swaps:
            live: Any = SwapController(controller)

            def on_swap(msg: String) -> None:
                if msg.data in swaps:
                    live.swap(swaps[msg.data], swap_time)
                else:
                    plant.node.get_logger().warning(f"no controller named {msg.data!r} to swap to")

            plant.node.create_subscription(String, TOPICS["swap"], on_swap, 10)
        else:
            live = LiveParams(plant.node, controller)
        return run(plant, live, WallClock(1.0 / profile.rate, stale=stale), T=duration)
    finally:
        plant.close()
