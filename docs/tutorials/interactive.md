---
file_format: mystnb
kernelspec:
  name: python3
---

# Interactive control

In this tutorial we change a running controller by hand: move its goal, change a gain, swap
the controller. Every change passes through a mailbox and is applied just before the next
control step. It gives the controller energy, which we count. And it is recorded as the
schedule of a configuration, so that a session can be repeated exactly.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## A session

A session simulates the robot in real time, in a worker thread, and opens a window on it. The
window draws the robot, with a cross for each goal to drag, a slider for each gain, buttons to
swap the controller and to record, and keys (below). We take the experiment of
[Experiments in files](configurations.md) without its schedule, since nobody else should move
the goal:

```python
import virtualmodelcontrol as vmc

spec = vmc.config.files.read("reach.yaml")
del spec["experiment"]["schedule"]
experiment = vmc.config.load(spec)

session = vmc.interactive.Session.from_experiment(
    experiment,
    goals=["ctrl.reach.goal"],
    sliders={"ctrl.reach.stiffness": (0.0, 600.0)},
)
log = session.run()  # the window opens; closing it ends the run
session.recorder.save(experiment, "my-session.yaml")
```

`run` returns the log of the session, a `RunLog` like any other
([Run logs](run-logs.md)). The simulation runs at the speed of the clock: `speed` simulated
seconds per second, 1 by default.

The documentation cannot hold a mouse, so a script plays the person. It presses the arrow
keys, moves the slider and clicks the second controller, through the events of the window.
The session runs as fast as the computer allows, with a speed of 1000:

```{code-cell} python
import time
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc

spec = vmc.config.files.read("reach.yaml")
del spec["experiment"]["schedule"]
experiment = vmc.config.load(spec)

session = vmc.interactive.Session.from_experiment(
    experiment,
    goals=["ctrl.reach.goal"],
    sliders={"ctrl.reach.stiffness": (0.0, 600.0)},
    speed=1000.0,
)
session.recorder.min_dt = 0.0  # keep every change
session.controls.toggle_recording()  # the Record button
session.start()
plt.close(session.window.fig)  # shown below, when there is something to see
```

```{code-cell} python
dt = 1 / 330  # [s]


def wait(steps):  # let the simulation take some steps
    while session.interactive.last is None:
        time.sleep(0.001)
    end = session.interactive.last[0] + steps * dt
    while session.interactive.last[0] < end:
        time.sleep(0.001)


window = session.window
window.keyboard.step = 0.04  # [m] per key press
wait(40)
window.keyboard.press("right")
window.keyboard.press("right")
wait(60)
window.keyboard.press("up")
wait(60)
window.sliders["ctrl.reach.stiffness"].set_val(450.0)  # [N/m]
wait(60)
window.radio.set_active(1)  # swap to the gentle controller
wait(120)
log = session.stop()
window.refresh()
rows = log.arrays()
len(rows["t"])
```

```{code-cell} python
window.fig
```

The goal went 8 cm to the right, 4 cm up, the stiffness went from 300 to 450 N/m, and the
gentle controller, which limits the force of its spring, took over. The window shows the
arm at the end of the run.

## What a change is

A change is not made on the controller from the window's thread. The window, the keys and the
joystick leave it in a mailbox, `session.controls`, which any thread may use:

```python
controls.set("ctrl.reach.goal", [0.05, 0.0, 0.40])  # a new value
controls.nudge("ctrl.reach.goal", [0.0, 0.0, 0.01])  # the latest plus this
controls.swap("gentle", 1.0)  # blend to another controller
```

`vmc.interactive.Interactive` wraps the controller and takes what the mailbox holds just before
each control step: the values go through `controller.set`, to every controller that has the
Param live, the swaps through the blend of [Experiments in files](configurations.md). A change is
therefore the same as calling `set` before
the step, and a run with changes is a plain run of the controller. Because only the control
loop touches the controller, the window can be in another thread, or the loop inside a robot's
own node:

```python
from virtualmodelcontrol.interactive import Interactive, Window

interactive = Interactive(controller, swaps={"gentle": other})
window = Window(robot, interactive, goals=["ctrl.reach.goal"])
# the node's loop runs `interactive` like any controller, in its thread
window.show()  # the window, here in the main thread
```

Closing the window stops a run like Ctrl-C does, with the log so far.

## What a change costs

Moving a spring's goal or stiffening it changes the energy it stores at once, and a person at
a slider can pump energy into a controller. `Interactive.injected` adds up what the changes
gave the running controller, as `controller.set` computes it. A goal moved from the tip by a
distance $d$ gives a spring $\tfrac12 K d^2$, a spring stiffer by $\Delta K$ at a deflection
$y$ gives $\tfrac12 \Delta K\,y^2$. For this session:

```{code-cell} python
session.interactive.injected  # [J]
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("injected", float(session.interactive.injected), display=False)
assert session.interactive.injected > 0.1
```

The changes gave the controller {glue:text}`injected:.2f` J.

## Record and repeat

While the Record button is down, the recorder notes every value and swap that is applied, with
its time. `recorder.schedule()` turns them into the entries of a configuration's `schedule`: a
value is held until the next, so each Param becomes a list of points with `step`
interpolation, and a swap is an entry at its time.

```{code-cell} python
session.recorder.save(experiment, "session.yaml")
text = open("session.yaml").read()
print(text[text.index("  schedule:"):])
```

`save` adds the entries to the schedule the configuration already has. The saved file is an
ordinary experiment. Run with nobody at the window, it gives the session back:

```{code-cell} python
spec = vmc.config.files.read("session.yaml")
spec["experiment"]["duration"] = len(rows["t"]) * dt  # [s], as long
replayed = vmc.config.load(spec).run()
gap = vmc.sim.compare(log, replayed)
max(entry["max"] for entry in gap.values())
```

```{code-cell} python
:tags: [remove-cell]
assert max(entry["max"] for entry in gap.values()) == 0.0
glue("signals", len(gap), display=False)
```

Every one of the {glue:text}`signals` signals both logs hold is the same, to the last bit. The
session was played in a thread at an arbitrary pace, but what the recorder keeps is the step at
which each change was applied, not when the person made it. Values that arrive closer than
`min_dt` [s] (0.02 by default) are merged into the last one, to keep a long drag short; a
replay is then exact up to that resolution.

## Keys and joystick

The window takes the keys of `Keyboard`: the arrows move the first goal in the plane shown by
`step` [m], PageUp and PageDown along the third axis, `[` and `]` halve and double the step,
space pauses a simulation, `r` records, and `1` to `9` swap to the n-th controller. A joystick
moves the goal at a speed, with a dead zone; pygame reads it, and is not installed with the
library:

```bash
pip install "virtualmodelcontrol[joystick]"
```

```python
from virtualmodelcontrol.interactive import Joystick, open_joystick

stick = Joystick(session.controls, "ctrl.reach.goal", open_joystick())
window.joystick = stick
```

The left stick moves the goal in the plane at up to `speed` [m/s] (0.1 by default), the right
stick along the third axis; button 0 toggles recording and buttons 4 and 5 swap to the previous
and the next controller. Anything else that has `get_axis` and `get_button` can stand in for
the device.
