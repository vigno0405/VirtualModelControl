"""A robot over ROS 2: the driver's joint topics in, torques out, zero torque on close."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float64MultiArray

from ..core.signals import Signals
from ..hardware.profile import HardwareProfile
from .joint_io import TOPICS, JointIO

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RosPlant:
    """The robot behind the driver's topics, as a plant.

    ``read`` returns the newest angles and rates, stamped with their arrival on the plant's clock
    ``t``, so a ``WallClock`` with ``stale`` refuses old ones (before the first message they are
    NaN, which the guard refuses). ``write`` publishes torques; ``close`` publishes zero torque.
    With ``lockstep`` the plant runs in step with a digital twin (``serve(..., lockstep=True)``):
    ``advance`` waits for the state that follows each command, so ``vmc.sim.run`` with a
    ``SimClock`` gives the same run as on the simulator itself. ``simulated`` tells whether a twin
    is publishing.
    """

    def __init__(
        self,
        profile: HardwareProfile,
        *,
        name: str = "vmc_plant",
        lockstep: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.io, self.lockstep, self.timeout = JointIO(profile), lockstep, timeout
        self.simulated = False
        self.u: np.ndarray = np.zeros(len(profile.ids))
        self._own_context = not rclpy.ok()
        if self._own_context:
            rclpy.init()
        self.node = rclpy.create_node(name)
        self._messages: dict[str, Any] = {"positions": None, "velocities": None}
        self._counts = {"positions": 0, "velocities": 0}
        self._stamp = -np.inf
        self._sent: dict[str, int] | None = None  # message counts when the last command left
        self._arrived = threading.Condition()
        self._t0, self._steps, self._dt = time.monotonic(), 0, 0.0
        self._publisher = self.node.create_publisher(Float64MultiArray, TOPICS["torque"], 10)
        for topic in ("positions", "velocities"):
            self.node.create_subscription(
                Float64MultiArray, TOPICS[topic], lambda msg, k=topic: self._on_state(k, msg), 10
            )
        self.node.create_subscription(Bool, TOPICS["simulated"], self._on_simulated, LATCHED)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    @property
    def t(self) -> float:
        """The plant's time [s]: wall time since it started, or steps times dt in lockstep."""
        return self._steps * self._dt if self.lockstep else time.monotonic() - self._t0

    def _on_state(self, topic: str, msg: Float64MultiArray) -> None:
        with self._arrived:
            self._messages[topic] = np.array(msg.data, dtype=float)
            self._counts[topic] += 1
            if topic == "positions":
                self._stamp = self.t
            self._arrived.notify_all()

    def _on_simulated(self, msg: Bool) -> None:
        self.simulated = bool(msg.data)

    def wait(self, after: dict[str, int] | None = None) -> None:
        """Block until angles and rates newer than the counts ``after`` (default: any) arrived."""
        after = after or {"positions": 0, "velocities": 0}
        with self._arrived:
            done = self._arrived.wait_for(
                lambda: all(self._counts[k] > n for k, n in after.items()), self.timeout
            )
        if not done:
            raise TimeoutError(f"no joint state on {TOPICS['positions']} within {self.timeout} s")

    def read(self) -> Signals:
        """The newest motor angles [rad] and rates [rad/s]."""
        with self._arrived:
            positions, velocities = self._messages["positions"], self._messages["velocities"]
            stamp = self._stamp
        if positions is None or velocities is None:
            nan = np.full(self.u.size, np.nan)
            return Signals(stamp, motor_position=nan, motor_velocity=nan)
        return self.io.measurement(stamp, positions, velocities)

    def write(self, cmd: Signals) -> None:
        """Publish ``cmd["motor_torque"]`` [N·m]."""
        self.u = np.array(cmd["motor_torque"], dtype=float)
        with self._arrived:
            self._sent = dict(self._counts)
        self._publisher.publish(Float64MultiArray(data=self.io.torque_message(self.u)))

    def advance(self, dt: float) -> None:
        """In lockstep: wait for the state that follows the last command; time moves by dt."""
        if self.lockstep:
            self.wait(self._sent)
            self._steps, self._dt = self._steps + 1, dt

    def close(self) -> None:
        """Publish zero torque, stop the node."""
        self._publisher.publish(Float64MultiArray(data=[0.0] * self.u.size))
        self._executor.shutdown()
        self.node.destroy_node()
        if self._own_context and rclpy.ok():
            rclpy.shutdown()
