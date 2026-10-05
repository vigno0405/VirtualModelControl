---
file_format: mystnb
kernelspec:
  name: python3
---

# Real-time runs

In this tutorial we run a controller as it runs on a robot: at a fixed rate on the computer's
clock, with every measurement checked. We read what the loop records about its own timing, and
see what it does when a step is late, a reading is old or missing, or the operator presses
Ctrl-C. Then comes a checklist for the first run on a real robot.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The loop

`vmc.sim.run` is the loop of simulations and robots alike. With a `WallClock` it runs on the
computer's clock: at every step it reads the plant, checks the reading, asks the controller for
the torques, writes them, and waits for the next step.

```python
clock = vmc.sim.WallClock(dt=1 / 330, stale=0.05)  # [s]
log = vmc.sim.run(robot, controller, clock, T=10.0)  # robot: your plant
```

The controller gets the time it really is, not a nominal $k\,\Delta t$: the virtual states
integrate over the measured step. `T=None` runs until Ctrl-C. A plant is anything with `read()`,
which returns the measurements with the time of the reading, and `write(cmd)`, which sends the
motor torques; `close()` releases it. `run` does not close the plant: do it yourself, with
`try` and `finally` or with `with` if your plant is a context manager.

On this page the plant is the simulated soft arm, and the clock a fake one that moves when the
loop waits or the controller takes time, so that nothing waits for real and every run repeats:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
tip = arm.point(s=1.0)
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - [0.08, 0.0, 0.40], 300.0))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
dt = 1 / 330  # [s]


class Robot:
    """The simulated arm on the clock's time, with faults to switch on."""

    def __init__(self, lag=0.0, blind=(1.0, 0.0)):
        self.sim = vmc.sim.ModelPlant(arm)
        self.lag, self.blind = lag, blind

    t = property(lambda self: self.sim.t)  # the clock's time

    def read(self):
        meas = self.sim.read()
        meas.t -= self.lag  # an old reading
        if self.blind[0] <= self.t < self.blind[1]:  # no velocity reading
            meas.set("motor_velocity", np.full(9, np.nan))
        return meas

    def write(self, cmd):
        self.sim.write(cmd)


class Busy:
    """The controller, some of whose steps take long (a busy computer)."""

    def __init__(self, robot, slow=()):
        self.robot, self.slow, self.steps = robot, slow, 0
        self.controller = vmc.VMCController(law)

    def reset(self, t, meas=None, z0=None):
        self.controller.reset(t, meas, z0)

    def step(self, t, meas):
        self.steps += 1
        if self.steps in self.slow:
            self.robot.sim.advance(0.02)  # [s]
        return self.controller.step(t, meas)


def run(robot, controller, T=0.5, stale=0.05):
    clock = vmc.sim.WallClock(
        dt, stale, now=lambda: robot.t, sleep=robot.sim.advance
    )
    return vmc.sim.run(robot, controller, clock, T)
```

## A healthy run

The log has the usual signals and `dt`, the measured length of each step. `log.info` sums up
the run:

```{code-cell} python
robot = Robot()
log = run(robot, Busy(robot))
log.info
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

assert log.info["steps"] == 165 and log.info["overruns"] == 0
glue("rate", log.info["rate"], display=False)
```

It ran at {glue:text}`rate:.0f` steps per second, as asked. `overruns` counts the steps that took
longer than the period, `stale` the readings the loop refused as too old (below), and
`guard_trips` the readings it refused as missing or not numbers.

## A late step

Some steps take too long, here three of them, by 20 ms. The loop does not hurry the next steps
to catch up: that would give the controller a burst of steps shorter than the period, and its
virtual states a time step it never planned for. It starts again from the moment the late step
ended, counts it, and warns once at the end:

```{code-cell} python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    robot = Robot()
    late = run(robot, Busy(robot, slow=(40, 80, 120)))
str(caught[0].message), late.info["overruns"]
```

```{code-cell} python
fig, ax = plt.subplots()
for name, result in (("healthy", log), ("three late steps", late)):
    ax.plot(1000 * result.arrays()["dt"][1:], label=name)
ax.set_xlabel("step")
ax.set_ylabel("step length [ms]")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
steps = late.arrays()["dt"].ravel()[1:]
assert steps.min() > 0.99 * dt, "a step was squeezed: rewrite"
glue("late", 1000 * float(steps.max()), display=False)
glue("nominal", 1000 * dt, display=False)
```

The late steps last {glue:text}`late:.0f` ms against {glue:text}`nominal:.1f` ms, and no step is
shorter than the period.

## An old reading

Over a network or a bus, a reading can arrive late. With `stale=0.05` the loop refuses a reading
more than 50 ms older than the plant's clock and sends zero torque instead, as it does for a
missing one. Here every reading is 80 ms old:

```{code-cell} python
robot = Robot(lag=0.08)
old = run(robot, Busy(robot), T=0.2)
old.info["stale"], bool(old.arrays()["motor_torque"].any())
```

## A missing reading

A reading that is missing or not a finite number trips the guard: the loop sends zero torque
and counts the trip. The velocities are NaN between 0.2 s and 0.3 s:

```{code-cell} python
robot = Robot(blind=(0.2, 0.3))
blind = run(robot, Busy(robot))
rows = blind.arrays()
silent = (rows["t"] > 0.2) & (rows["t"] < 0.3)
blind.info["guard_trips"], bool(rows["motor_torque"][silent.ravel()].any())
```

```{code-cell} python
:tags: [remove-cell]
assert old.info["stale"] == len(old.arrays()["t"])
assert not old.arrays()["motor_torque"].any() and blind.info["guard_trips"] > 20
```

## Ctrl-C

Ctrl-C ends a real-time run with the log so far, and the same does `controls.stop()` of an
[interactive session](interactive.md). Here the controller raises it at its 60th step:

```{code-cell} python
class Interrupted(Busy):
    def step(self, t, meas):
        if self.steps == 60:
            raise KeyboardInterrupt
        return super().step(t, meas)


robot = Robot()
len(run(robot, Interrupted(robot), T=None).arrays()["t"])
```

The torque of the last step stays on the motors until something changes it. That is why the
robot's node needs a watchdog and a zero torque on exit (below).

## Before the first run on a robot

Nothing touches a real robot without its owner's go-ahead. Before the first run, check:

1. **The same controller in simulation first.** Same file, same rate, the robot's own
   dynamics; look at the torques and the energies ([Run logs](run-logs.md)).
2. **Signs and units.** Move the robot by hand with the torque off and watch
   `motor_position` and `motor_velocity` change the way the model says; the hardware profile
   holds each motor's sign, and everything inside the library is in SI units.
3. **A run that writes nothing.** Run the loop on a plant whose `write` does nothing, to see
   that the readings are fresh and the rate holds (`log.info`).
4. **Low gains, then more.** Start with a small stiffness, a damping that holds, and a goal
   close to where the robot is. Change gains live with `controller.set`, never by editing the
   file in the middle of a run.
5. **Limits.** Joint-limit springs, an output stage with a `TorqueLimit` for the first runs,
   and a goal inside the workspace. No torque limit is applied unless you ask for one.
6. **Stale and missing data.** Run with `stale` set to a few periods, and check the
   `stale` and `guard_trips` counts at the end of every run.
7. **A way out.** The robot's node stops the motors when the loop stops writing (a watchdog)
   and on exit, and someone is within reach of the power switch.
8. **The first step.** The controller starts from the robot's current state. Read the torques
   of the first steps in the log: they should be small, before you raise the gains.
