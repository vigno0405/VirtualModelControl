"""The robot over ROS 2 against a digital twin, on a private domain."""

import threading

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

import virtualmodelcontrol as vmc  # noqa: E402
from virtualmodelcontrol.robots import adapt  # noqa: E402
from virtualmodelcontrol.ros import LiveParams, RosPlant, serve  # noqa: E402

DT = 1 / 500


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def finger_controller():
    robot, ctrl = adapt.add_dynamics(adapt.finger(), damping=0.01), vmc.Mechanism("ctrl")
    ctrl.add("tip", vmc.LinearSpring(robot.point("tip") - [0.0, 0.04, 0.06], 100.0))
    ctrl.add("damp", vmc.LinearDamper(robot.point("tip"), 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(robot))
    return robot, vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


def start_twin(robot, **options):
    stop = threading.Event()
    plant = vmc.sim.ModelPlant(robot)
    thread = threading.Thread(
        target=serve, args=(plant, adapt.finger_hardware()), kwargs={"stop": stop, **options}
    )
    thread.start()
    return stop, thread


def test_a_lockstep_run_over_ros_repeats_the_simulators(ros):
    robot, controller = finger_controller()
    here = vmc.sim.run(vmc.sim.ModelPlant(robot), controller, vmc.sim.SimClock(DT), T=0.2)

    robot, controller = finger_controller()
    stop, thread = start_twin(robot, rate=1 / DT, lockstep=True)
    plant = RosPlant(adapt.finger_hardware(), lockstep=True)
    try:
        plant.wait()
        remote = vmc.sim.run(plant, controller, vmc.sim.SimClock(DT), T=0.2)
        assert plant.simulated
    finally:
        plant.close()
        stop.set()
        thread.join()
    a, b = here.arrays(), remote.arrays()
    np.testing.assert_allclose(b["motor_position"], a["motor_position"], atol=1e-9)
    np.testing.assert_allclose(b["motor_torque"], a["motor_torque"], atol=1e-9)


def test_live_parameters_reach_the_controller_between_steps(ros):
    robot, controller = finger_controller()
    stop, thread = start_twin(robot, rate=1 / DT, lockstep=True)
    plant = RosPlant(adapt.finger_hardware(), lockstep=True)
    try:
        live = LiveParams(plant.node, controller)
        plant.wait()
        vmc.sim.run(plant, live, vmc.sim.SimClock(DT), T=0.01)
        param = rclpy.parameter.Parameter("ctrl.tip.stiffness", value=250.0)
        assert plant.node.set_parameters([param])[0].successful
        before = controller.params.copy()
        vmc.sim.run(plant, live, vmc.sim.SimClock(DT), T=0.01)
    finally:
        plant.close()
        stop.set()
        thread.join()
    where = controller.compiled.live_slices()["ctrl.tip.stiffness"]
    assert before[where] == 100.0 and controller.params[where] == 250.0


def test_close_sends_zero_torque(ros):
    from std_msgs.msg import Float64MultiArray

    received = []
    listener = rclpy.create_node("listener")
    listener.create_subscription(
        Float64MultiArray, "/goal_torque", lambda m: received.append(list(m.data)), 10
    )
    plant = RosPlant(adapt.finger_hardware())
    plant.write(vmc.Signals(0.0, motor_torque=[0.1, 0.2]))
    plant.close()
    for _ in range(50):
        rclpy.spin_once(listener, timeout_sec=0.05)
        if len(received) >= 2:
            break
    listener.destroy_node()
    assert received[-1] == [0.0, 0.0] and received[0] == [0.1, 0.2]
