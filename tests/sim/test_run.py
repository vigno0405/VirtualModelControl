"""The run loop: closed-loop simulation of the soft arm, the guard and the recorder."""

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import helyx, turtle


def test_closed_loop_soft_arm_reaches_towards_the_goal():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    goal = np.array([0.08, 0.0, 0.68])
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 300.0))
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    plant = vmc.sim.ModelPlant(arm)

    start = np.linalg.norm(evaluate_frame(arm.model, plant.q, 1.0)[1] - goal)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=2.0)
    tip = evaluate_frame(arm.model, plant.q, 1.0)[1]

    rows = log.arrays()
    assert rows["motor_torque"].shape == (660, 9) and rows["t"].shape == (660, 1)
    assert np.abs(plant.v).max() < 1e-4  # at rest
    assert np.linalg.norm(tip - goal) < 0.5 * start  # pulled towards the goal
    assert tip[0] > 0.01  # and bent towards +x


class BrokenSensor:
    """A plant whose velocity reading is NaN."""

    t = 0.0
    u = np.zeros(2)

    def read(self):
        return vmc.Signals(self.t, motor_position=[0.0, 0.0], motor_velocity=[np.nan, 0.0])

    def write(self, cmd):
        self.u = cmd["motor_torque"]

    def advance(self, dt):
        self.t += dt


class Never:
    def reset(self, t, meas):
        pass

    def step(self, t, meas):  # pragma: no cover - must not be called
        raise AssertionError("the guard should have stopped this step")


def test_guard_sends_zero_torque_on_bad_measurements():
    guard = vmc.sim.Guard()
    log = vmc.sim.run(BrokenSensor(), Never(), vmc.sim.SimClock(0.01), T=0.05, guard=guard)
    assert guard.trips == 5
    assert np.all(log.arrays()["motor_torque"] == 0.0)


def test_virtual_states_start_from_z0_and_are_logged():
    robot = turtle.robot()
    robot.add("inertia", vmc.Inertance(robot.joint([0, 1]), 1e-3))
    robot.add("friction", vmc.LinearDamper(robot.joint([0, 1]), 1e-2))
    ctrl = turtle.controller(robot, speed=2.0, ramp_time=0.5, stiffness=0.5, damping=0.01)
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    plant = vmc.sim.ModelPlant(robot, q0=[0.3, 0.0])
    z0 = turtle.initial_state(plant.read())
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=0.01), T=3.0, z0=z0)

    z = log.arrays()["z"]
    assert z.shape == (300, 2)  # flywheel angle and speed after every step
    np.testing.assert_allclose(z[0], [0.3, 0.0])  # started at the left crank, at rest
    # The crank friction loads the flywheel: b_v (ω̄ − ω) = 2 c ω at steady state.
    steady = 0.1 * 2.0 / (0.1 + 2 * 1e-2)
    assert 0.9 * steady < z[-1, 1] < steady  # spinning up towards it


def test_controllers_without_virtual_states_log_no_z():
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(dt=1 / 330), T=0.05)
    assert "z" not in log.arrays()


def test_the_log_copies_each_row():
    log = vmc.sim.RunLog()
    state = np.zeros(2)
    for _ in range(3):
        state += 1.0  # updated in place, as a controller's virtual state is
        log.append(z=state)
    np.testing.assert_array_equal(log.arrays()["z"], [[1, 1], [2, 2], [3, 3]])
