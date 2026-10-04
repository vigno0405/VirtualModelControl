"""The controller node: a controller on the robot behind the driver's topics, in real time."""

from __future__ import annotations

from typing import Any

from ..hardware.profile import HardwareProfile
from ..sim.run import RunLog, WallClock, run
from .params import LiveParams
from .plant import RosPlant


def control(
    controller: Any,
    profile: HardwareProfile,
    *,
    duration: float | None = None,
    stale: float | None = 0.05,
    name: str = "vmc_controller",
) -> RunLog:
    """Run ``controller`` at the profile's rate on the robot behind the driver's topics.

    Its live Params are ROS parameters of the node ``name`` (``ros2 param set`` changes them
    between steps). The run lasts ``duration`` [s] or until Ctrl-C; readings older than
    ``stale`` [s] send zero torque, and the robot gets zero torque at the end. Returns the log
    (``log.info`` holds the loop's statistics).
    """
    plant = RosPlant(profile, name=name)
    try:
        plant.wait()
        live = LiveParams(plant.node, controller)
        return run(plant, live, WallClock(1.0 / profile.rate, stale=stale), T=duration)
    finally:
        plant.close()
