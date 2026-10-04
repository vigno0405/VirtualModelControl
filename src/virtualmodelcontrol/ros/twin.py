"""A digital twin: a simulated robot on the driver's topics, for controllers written for the real
robot."""

from __future__ import annotations

import threading
import time
from typing import Any

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, Float64MultiArray

from ..core.signals import Signals
from ..hardware.profile import HardwareProfile
from .joint_io import TOPICS, JointIO
from .plant import LATCHED


def serve(
    plant: Any,
    profile: HardwareProfile,
    *,
    rate: float = 1000.0,
    lockstep: bool = False,
    duration: float | None = None,
    stop: threading.Event | None = None,
    name: str = "vmc_twin",
) -> None:
    """Publish the simulated ``plant`` as the driver would, and apply the torques it receives.

    In real time the twin advances by 1 / ``rate`` [s] at that rate, holding the last torque.
    With ``lockstep`` it repeats its start state until the first command, then advances one step
    per command and publishes the state that follows, so runs over ROS are repeatable. It
    publishes ``robot_is_simulated`` and serves until ``duration`` [s], ``stop`` or Ctrl-C.
    """
    io, dt = JointIO(profile), 1.0 / rate
    own_context = not rclpy.ok()
    if own_context:
        rclpy.init()
    node = rclpy.create_node(name)
    positions = node.create_publisher(Float64MultiArray, TOPICS["positions"], 10)
    velocities = node.create_publisher(Float64MultiArray, TOPICS["velocities"], 10)
    node.create_publisher(Bool, TOPICS["simulated"], LATCHED).publish(Bool(data=True))
    start = plant.read()["motor_position"].copy()  # the driver publishes from its start position
    commanded = False

    def publish() -> None:
        meas = plant.read()
        velocities.publish(Float64MultiArray(data=io.angle_message(meas["motor_velocity"])))
        positions.publish(Float64MultiArray(data=io.angle_message(meas["motor_position"] - start)))

    def on_torque(msg: Float64MultiArray) -> None:
        nonlocal commanded
        plant.write(Signals(plant.t, motor_torque=io.motor_torques(msg.data)))
        commanded = True
        if lockstep:
            plant.advance(dt)
            publish()

    def tick() -> None:
        if not lockstep:
            plant.advance(dt)
            publish()
        elif not commanded:
            publish()

    node.create_subscription(Float64MultiArray, TOPICS["torque"], on_torque, 10)
    node.create_timer(0.05 if lockstep else dt, tick)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    end = None if duration is None else time.monotonic() + duration
    try:
        while (stop is None or not stop.is_set()) and (end is None or time.monotonic() < end):
            executor.spin_once(timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if own_context and rclpy.ok():
            rclpy.shutdown()
