"""The run loop on real time: steps on schedule, measured time, stale readings, overruns."""

import time

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt


class FakeTime:
    """A clock that moves only when the loop sleeps or a step takes time."""

    def __init__(self):
        self.t = 100.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Computer:
    """The computer's own clock, as the time of a plant."""

    @property
    def t(self):
        return time.monotonic()


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


def test_a_real_time_run_on_the_computers_clock_is_paced():
    robot, ctrl = adapt.finger(), vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)), 1.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    clock = vmc.sim.WallClock(dt=0.005, stale=0.05)
    log = vmc.sim.run(Plant(Computer()), controller, clock, T=0.1)
    assert log.info["steps"] == 20 and log.info["stale"] == 0
    assert 20.0 < log.info["rate"] < 300.0  # paced at 200 Hz, as far as a busy computer lets it


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


def test_a_window_keeps_the_last_seconds_of_a_run_that_goes_on():
    for steps in range(300, 313):  # the run ends at every point of the trimming cycle
        time = FakeTime()
        clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
        log = vmc.sim.run(Plant(time), Controller(time), clock, T=steps * 0.01, window=0.5)
        rows = log.arrays()
        t = rows["t"].ravel()
        assert log.info["steps"] == steps  # every step ran
        assert 50 <= len(t) <= 62  # the last 0.5 s, and at most a quarter more
        assert t[-1] == pytest.approx((steps - 1) * 0.01)  # the newest steps are the kept ones
        np.testing.assert_allclose(np.diff(t), 0.01, atol=1e-9)  # no gap: the oldest rows went
        assert all(len(v) == len(t) for v in rows.values())  # every signal was trimmed together
    time = FakeTime()
    clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
    full = vmc.sim.run(Plant(time), Controller(time), clock, T=3.0)
    assert len(full.arrays()["t"]) == 300  # without a window nothing is dropped


def test_ctrl_c_in_the_middle_of_a_step_of_a_windowed_run_keeps_only_whole_steps(monkeypatch):
    original = vmc.sim.runlog.RunLog.step
    calls = []

    def interrupted(self, **values):
        calls.append(1)
        if len(calls) == 121:  # the signal comes after the first signal of the step is logged
            first, *_ = values
            self.rows.setdefault(first, []).append(np.array([0.0]))
            raise KeyboardInterrupt
        original(self, **values)

    monkeypatch.setattr(vmc.sim.runlog.RunLog, "step", interrupted)
    time = FakeTime()
    clock = vmc.sim.WallClock(dt=0.01, now=time.now, sleep=time.sleep)
    log = vmc.sim.run(Plant(time), Controller(time), clock, T=None, window=0.5)
    rows = log.arrays()
    assert log.info["steps"] == 120 and 50 <= len(rows["t"]) <= 62
    assert len({len(v) for v in rows.values()}) == 1 and rows["t"].ravel()[-1] == pytest.approx(
        1.19
    )


def test_a_window_is_for_a_run_on_real_time():
    with pytest.raises(ValueError, match="real-time"):
        vmc.sim.run(None, None, vmc.sim.SimClock(0.01), T=1.0, window=0.5)
