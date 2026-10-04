"""The run loop on real time: steps on schedule, measured time, stale readings, overruns."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.hardware import DynamixelPlant, FakeBus
from virtualmodelcontrol.robots import adapt


class FakeTime:
    """A clock that moves only when the loop sleeps or a step takes time."""

    def __init__(self):
        self.t = 100.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Plant:
    """Two motors at rest; ``lag`` [s] makes every reading that old."""

    def __init__(self, time, lag=0.0):
        self.time, self.lag, self.sent = time, lag, []

    @property
    def t(self):
        return self.time.t

    def read(self):
        return vmc.Signals(
            self.t - self.lag, motor_position=np.zeros(2), motor_velocity=np.zeros(2)
        )

    def write(self, cmd):
        self.sent.append(np.array(cmd["motor_torque"]))


class Controller:
    """Constant torque; each step takes ``cost`` [s] of the fake clock."""

    def __init__(self, time, cost=0.0):
        self.time, self.cost, self.times = time, cost, []

    def reset(self, t, meas):
        pass

    def step(self, t, meas):
        self.times.append(t)
        self.time.t += self.cost
        return vmc.Signals(t, motor_torque=np.ones(2))


def test_steps_run_on_schedule_with_the_measured_time():
    time = FakeTime()
    clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
    controller = Controller(time, cost=0.002)
    log = vmc.sim.run(Plant(time), controller, clock, T=0.1)
    np.testing.assert_allclose(controller.times, np.arange(10) * 0.01, atol=1e-12)
    assert log.info["steps"] == 10 and log.info["overruns"] == 0
    assert log.info["rate"] == pytest.approx(100.0)
    np.testing.assert_allclose(log.arrays()["dt"][1:].ravel(), 0.01)


def test_late_steps_are_counted_and_warned_without_a_burst():
    time = FakeTime()
    clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
    controller = Controller(time, cost=0.015)
    with pytest.warns(RuntimeWarning, match="10 of 10 steps took longer than 10.0 ms"):
        log = vmc.sim.run(Plant(time), controller, clock, T=0.1)
    assert log.info["overruns"] == 10
    np.testing.assert_allclose(np.diff(controller.times), 0.015)  # never squeezed to catch up


def test_stale_readings_send_zero_torque():
    time = FakeTime()
    clock = vmc.sim.WallClock(dt=0.01, stale=0.05, now=time.now, sleep=time.sleep)
    plant, controller = Plant(time, lag=0.2), Controller(time)
    log = vmc.sim.run(plant, controller, clock, T=0.05)
    assert log.info["stale"] == 5 and controller.times == []
    assert all(np.all(u == 0.0) for u in plant.sent)


def test_a_real_time_run_on_the_finger_through_a_fake_bus():
    profile = adapt.finger_hardware()
    robot, ctrl = adapt.finger(), vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)), 1.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    bus = FakeBus(
        dict.fromkeys(profile.ids, "XC330-T288"), dict(zip(profile.ids, (2000, 3000), strict=True))
    )
    with DynamixelPlant(profile, bus) as plant:
        log = vmc.sim.run(plant, controller, vmc.sim.WallClock(dt=0.005, stale=0.05), T=0.1)
    assert log.info["steps"] == 20 and log.info["stale"] == 0
    assert log.info["rate"] == pytest.approx(200.0, rel=0.5)  # paced, not flat out


def test_ctrl_c_ends_a_real_time_run_with_the_log_so_far():
    time = FakeTime()

    class Interrupted(Controller):
        def step(self, t, meas):
            if len(self.times) == 5:
                raise KeyboardInterrupt
            return super().step(t, meas)

    clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
    log = vmc.sim.run(Plant(time), Interrupted(time), clock, T=None)
    rows = log.arrays()
    assert log.info["steps"] == 5
    assert len(rows["t"]) == len(rows["motor_torque"]) == len(rows["motor_position"]) == 5


def test_a_simulated_run_needs_its_duration():
    with pytest.raises(ValueError, match="duration"):
        vmc.sim.run(None, None, vmc.sim.SimClock(0.01), T=None)
