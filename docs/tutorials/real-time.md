---
file_format: mystnb
kernelspec:
  name: python3
---

# Real-time runs

A controller runs the same way on a robot and in a simulation: `vmc.sim.run` reads the plant,
asks the controller for torques, writes them, and waits for the next step. This page has two
parts: how to connect a robot and run on it, and how to test a controller in simulation first.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## Part 1: On a robot

### The mechanism needs only kinematics

A real robot needs no dynamics in the library: the robot itself is the plant. The robot's
mechanism holds its kinematics, its motors and the masses that gravity compensation uses, and
the hardware profile holds the motors' order, signs and units:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")  # kinematics, tendons, masses
profile = helyx.hardware("145-290-290")  # motor order, signs, units
tip = arm.point(s=1.0)

ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - [0.05, 0.0, 0.70], 100.0))
ctrl.add("damp", vmc.LinearDamper(tip, 2.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))
```

### The plant is your driver, seen by the library

The library does not talk to motors. A plant is the small object between it and your driver (a
ROS node's last messages, a serial driver, anything) with three members: `t`, the time now [s];
`read()`, which returns `motor_position` [rad] and `motor_velocity` [rad/s] with the time of
the reading; and `write(cmd)`, which sends `cmd["motor_torque"]` [N·m]. The profile does the
unit conversions:

```{code-cell} python
class Plant:
    def __init__(self, driver):
        self.driver = driver

    t = property(lambda self: self.driver.time())

    def read(self):
        deg, deg_s = self.driver.read()  # from the start pose
        return vmc.Signals(
            self.t,
            motor_position=profile.angles_from_degrees(deg),
            motor_velocity=profile.angles_from_degrees(deg_s),
        )

    def write(self, cmd):
        self.driver.write(profile.bus_torques(cmd["motor_torque"]))

    def close(self):
        self.driver.write(np.zeros(9))  # no torque when we stop
```

`driver` is yours. So that this page runs without a robot, a simulated arm stands in for it and
answers like the motors:

```{code-cell} python
class Driver:
    """Stands in for your driver: a simulated arm, in degrees."""

    def __init__(self):
        arm = helyx.add_dynamics(helyx.arm("145-290-290"))
        self.sim = vmc.sim.ModelPlant(arm)

    def time(self):  # [s]
        return self.sim.t

    def read(self):
        meas = self.sim.read()
        return (profile.degrees(meas["motor_position"]),
                profile.degrees(meas["motor_velocity"]))

    def write(self, torques):  # this profile's conversion undoes itself
        tau = profile.bus_torques(torques)
        self.sim.write(vmc.Signals(self.time(), motor_torque=tau))
```

### Run

`WallClock` runs the loop on the computer's clock. On the robot it is just:

```python
clock = vmc.sim.WallClock(dt=1 / 330, stale=0.05)  # [s]
log = vmc.sim.run(Plant(driver), controller, clock, T=10.0)
```

`T=None` runs until Ctrl-C.

Here its `now` and `sleep` are the stand-in's, so that nothing waits for real:

```{code-cell} python
driver = Driver()
clock = vmc.sim.WallClock(1 / 330, 0.05, now=driver.time,
                          sleep=driver.sim.advance)
log = vmc.sim.run(Plant(driver), controller, clock, T=1.0)
log.info
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

assert log.info["steps"] == 330 and log.info["overruns"] == 0
glue("rate", log.info["rate"], display=False)
```

At every step the loop reads the plant, checks the reading, asks the controller for the torques,
writes them and waits for the next step. The controller gets the time it really is, not a
nominal $k\,\Delta t$, so its virtual states integrate over the measured step. The run took
{glue:text}`rate:.0f` steps per second, as asked. `log` holds the signals and `dt`, the measured
length of each step; `log.info` counts the `overruns` (steps longer than the period), the
`stale` readings the loop refused as too old and the `guard_trips` (readings missing or not
numbers). A refused reading means zero torque for that step.

`run` does not close the plant: do it yourself, with `try` and `finally`.

### Your own loop

If your robot already has a node that loops, skip `run` and call the controller in it:

```python
controller.reset(plant.t, plant.read())
while running:
    meas = plant.read()
    plant.write(controller.step(plant.t, meas))
```

### Before the first run

Nothing touches a real robot without its owner's go-ahead. Check, in order:

1. **The same controller in simulation first** (Part 2): same file, same rate; look at the
   torques and the energies ([Run logs](run-logs.md)).
2. **Signs and units.** Move the robot by hand with the torque off and watch `motor_position`
   and `motor_velocity` change as the model says.
3. **A run that writes nothing.** Make `write` do nothing and look at `log.info`: the readings
   should be fresh and the rate should hold.
4. **Low gains, then more.** A small stiffness, a damping that holds, a goal close to where the
   robot is. Change gains live with `controller.set`, never by editing the file mid-run.
5. **Limits.** Joint-limit springs, an output stage with a `TorqueLimit` for the first runs, a
   goal inside the workspace. No torque limit is applied unless you ask for one.
6. **Stale and missing data.** Set `stale` to a few periods, and read `stale` and `guard_trips`
   at the end of every run.
7. **A way out.** Your node stops the motors when the loop stops writing (a watchdog) and on
   exit, and someone is within reach of the power switch. The last torque stays on the motors
   until something changes it.
8. **The first step.** The controller starts from the robot's current state; its first torques
   should be small, before you raise the gains.

## Part 2: In simulation

### The same controller on a simulated arm

For a simulation the arm also needs its own stiffness, damping and gravity, which
`helyx.add_dynamics` adds. `ModelPlant` is the plant that integrates them, and `SimClock` steps
simulated time:

```{code-cell} python
sim_arm = helyx.add_dynamics(helyx.arm("145-290-290"))
plant = vmc.sim.ModelPlant(sim_arm)
sim = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=3.0)
sim.arrays()["motor_torque"].shape
```

Nothing else changes: the same `controller` and the same `run`. The simulation is the place to
choose gains and to read the energies ([Parameters](parameters.md), [Energy](energy.md)).

### The real-time loop under faults

The loop also has to survive a robot that is late or silent. To see how, we run it with a fake
clock, so that every run repeats, on a simulated arm with faults to switch on:

```{code-cell} python
dt = 1 / 330  # [s]


class Faulty:
    """The simulated arm on the clock's time, with faults to switch on."""

    def __init__(self, lag=0.0, blind=(1.0, 0.0)):
        self.sim = vmc.sim.ModelPlant(sim_arm)
        self.lag, self.blind = lag, blind

    t = property(lambda self: self.sim.t)

    def read(self):
        meas = self.sim.read()
        meas.t -= self.lag  # an old reading
        if self.blind[0] <= self.t < self.blind[1]:  # no velocity reading
            meas.set("motor_velocity", np.full(9, np.nan))
        return meas

    def write(self, cmd):
        self.sim.write(cmd)


class Slow:
    """The controller, some of whose steps take long (a busy computer)."""

    def __init__(self, plant, slow=()):
        self.plant, self.slow, self.steps = plant, slow, 0
        self.controller = vmc.VMCController(vmc.compile(system))

    def reset(self, t, meas=None, z0=None):
        self.controller.reset(t, meas, z0)

    def step(self, t, meas):
        self.steps += 1
        if self.steps in self.slow:
            self.plant.sim.advance(0.02)  # [s]
        return self.controller.step(t, meas)


def go(plant, ctrl, T=0.5, stale=0.05):
    clock = vmc.sim.WallClock(dt, stale, now=lambda: plant.t,
                              sleep=plant.sim.advance)
    return vmc.sim.run(plant, ctrl, clock, T)
```

**A late step.** Three steps take 20 ms too long. The loop does not hurry the next steps to catch
up: that would give the controller a burst of steps shorter than the period. It starts again
from the moment the late step ended, counts it, and warns once at the end:

```{code-cell} python
import warnings

plant = Faulty()
ok = go(plant, Slow(plant))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plant = Faulty()
    late = go(plant, Slow(plant, slow=(40, 80, 120)))
str(caught[0].message), late.info["overruns"]
```

```{code-cell} python
fig, ax = plt.subplots()
for name, result in (("healthy", ok), ("three late steps", late)):
    ax.plot(1000 * result.arrays()["dt"][1:], label=name)
ax.set_xlabel("step")
ax.set_ylabel("step length [ms]")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          fontsize=18);
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

**An old reading.** Over a network or a bus, a reading can arrive late. With `stale=0.05` the
loop refuses a reading more than 50 ms older than the plant's clock and sends zero torque. Here
every reading is 80 ms old:

```{code-cell} python
plant = Faulty(lag=0.08)
old = go(plant, Slow(plant), T=0.2)
old.info["stale"], bool(old.arrays()["motor_torque"].any())
```

**A missing reading.** A reading that is missing or not a finite number trips the guard: zero
torque, and the trip is counted. Here the velocities are NaN between 0.2 s and 0.3 s:

```{code-cell} python
plant = Faulty(blind=(0.2, 0.3))
blind = go(plant, Slow(plant))
rows = blind.arrays()
silent = (rows["t"] > 0.2) & (rows["t"] < 0.3)
blind.info["guard_trips"], bool(rows["motor_torque"][silent.ravel()].any())
```

```{code-cell} python
:tags: [remove-cell]
assert old.info["stale"] == len(old.arrays()["t"])
assert not old.arrays()["motor_torque"].any() and blind.info["guard_trips"] > 20
```

**Ctrl-C.** It ends a real-time run with the log so far. Here the controller raises it at its
60th step:

```{code-cell} python
class Stop(Slow):
    def step(self, t, meas):
        if self.steps == 60:
            raise KeyboardInterrupt
        return super().step(t, meas)


plant = Faulty()
len(go(plant, Stop(plant), T=None).arrays()["t"])
```

The torque of the last step stays on the motors until something changes it, which is why a
robot's node needs a watchdog and a zero torque on exit.
