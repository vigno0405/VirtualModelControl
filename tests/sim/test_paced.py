"""A simulation paced to the computer's clock: waits, no bursts, runs until stopped."""

import time

import numpy as np
import pytest

import virtualmodelcontrol as vmc


class FakeTime:
    """A clock that moves only when the loop sleeps or a step takes time."""

    def __init__(self):
        self.t, self.sleeps = 100.0, []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class Plant:
    """Two motors at rest, simulated time only."""

    def __init__(self):
        self.t = 0.0

    def read(self):
        return vmc.Signals(self.t, motor_position=np.zeros(2), motor_velocity=np.zeros(2))

    def write(self, cmd):
        pass

    def advance(self, dt):
        self.t += dt


class Controller:
    """Zero torque; the step number ``stop`` raises Ctrl-C, and each step takes ``cost`` [s]."""

    def __init__(self, clock, cost=0.0, stop=None):
        self.clock, self.cost, self.stop, self.steps = clock, cost, stop, 0

    def reset(self, t, meas):
        pass

    def step(self, t, meas):
        if self.steps == self.stop:
            raise KeyboardInterrupt
        self.steps += 1
        self.clock.t += self.cost
        return vmc.Signals(t, motor_torque=np.zeros(2))


def paced(time_, speed, **kwargs):
    return vmc.sim.SimClock(0.01, speed=speed, now=time_.now, sleep=time_.sleep, **kwargs)


def test_a_paced_simulation_follows_the_clock_at_its_speed():
    fake, plant = FakeTime(), Plant()
    log = vmc.sim.run(plant, Controller(fake), paced(fake, speed=2.0), T=0.1)
    assert len(log.arrays()["t"]) == 10 and plant.t == pytest.approx(0.1)
    assert fake.t - 100.0 == pytest.approx(0.05)  # twice as fast as the clock
    assert fake.sleeps == pytest.approx([0.005] * 10)


def test_a_late_step_starts_again_from_now_instead_of_catching_up():
    fake = FakeTime()
    controller = Controller(fake, cost=0.025)  # slower than the 10 ms steps
    vmc.sim.run(Plant(), controller, paced(fake, speed=1.0), T=0.1)
    assert fake.sleeps == []  # never waits, never squeezes steps together
    assert fake.t - 100.0 == pytest.approx(10 * 0.025)

    fake = FakeTime()

    class Spike(Controller):
        def step(self, t, meas):
            self.cost = 0.2 if self.steps == 2 else 0.0  # one slow step
            return super().step(t, meas)

    vmc.sim.run(Plant(), Spike(fake), paced(fake, speed=1.0), T=0.1)
    assert fake.sleeps == pytest.approx([0.01] * 9)  # every other step waits one period, no burst
    assert fake.t - 100.0 == pytest.approx(0.2 + 9 * 0.01)


def test_a_paced_simulation_runs_until_stopped_and_keeps_the_log():
    fake, plant = FakeTime(), Plant()
    log = vmc.sim.run(plant, Controller(fake, stop=5), paced(fake, speed=1.0), T=None)
    rows = log.arrays()
    assert len(rows["t"]) == len(rows["motor_torque"]) == len(rows["motor_position"]) == 5
    assert plant.t == pytest.approx(0.05)


def test_unpaced_simulations_are_unchanged():
    clock = vmc.sim.SimClock(0.01)
    with pytest.raises(ValueError, match="duration"):
        vmc.sim.run(Plant(), Controller(FakeTime()), clock, T=None)
    with pytest.raises(KeyboardInterrupt):  # Ctrl-C still aborts a plain simulation
        vmc.sim.run(Plant(), Controller(FakeTime(), stop=3), clock, T=0.1)


def test_the_real_clock_is_waited_for():
    start = time.monotonic()
    vmc.sim.run(Plant(), Controller(FakeTime()), vmc.sim.SimClock(0.005, speed=1.0), T=0.05)
    assert time.monotonic() - start >= 0.045  # at least the time the steps stand for
