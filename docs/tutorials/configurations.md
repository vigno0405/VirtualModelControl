---
file_format: mystnb
kernelspec:
  name: python3
---

# Experiments in files

In this tutorial we write an experiment on the soft arm as a YAML file, run it, let its goal and
its controller change on a schedule, and save it back with a new stiffness.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The file

A configuration file describes an experiment in a few sections. `robot` names a robot template
and its arguments. `coordinates` names coordinates that the controllers share. `controller`
lists the virtual elements: each names its registered `type`, the `coordinate` it acts on and
its gains. This one is the controller of [your first controller](first-controller.md):

```{literalinclude} reach.yaml
:language: yaml
:end-before: swaps:
```

A coordinate is a name from `coordinates`, or a mapping of one kind to its arguments, written as
in Python, for instance `point` (`{s: 1.0}`, a site's name, or `{at, s, offset}`), `joint` (an index, a list,
or `{start, stop}`), `ref` (a live reference with its `name` and `value`), `difference`,
`projection` (`{of, direction}`), `norm`, `slice` (`{of, index}`), `stack`, `plane_distance`,
`sphere_distance` and `state` (the full list is at the end of this page). A plain list in a difference is a live reference named `ref`,
as `tip - [0.1, 0.0, 0.40]` is in Python.

## Run it

The file is [reach.yaml](https://github.com/vigno0405/VirtualModelControl/blob/main/docs/tutorials/reach.yaml): download it into the folder where you run the code.
Its sections are `robot`, `coordinates`, `controller`, `swaps` and `experiment`, in this order,
and the page shows `experiment` before `swaps`. `vmc.config.load` reads the file and builds the robot from its template, the controllers and
the plant. `run` runs the experiment from its start and returns the log. With `run` settings in
the file, it also saves the log, as [Run logs](run-logs.md) shows:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc

experiment = vmc.config.load("reach.yaml")
rows = experiment.run().arrays()
sorted(rows)
```

`experiment.robot`, `experiment.mechanism` and `experiment.controller` are the objects the file
describes, the same ones [your first controller](first-controller.md) builds in Python. So the
other tutorials apply to them unchanged.

## Change values on a schedule

The `experiment` section says how to run it: the plant (here a simulation with the arm's own
stiffness, damping and gravity from `helyx.add_dynamics`), the control rate [Hz], the duration
[s] and a schedule. An `output` list adds output stages to the controller, such as the real
arm's pretension `{type: helyx.output_stage}`. This file has none:

```{literalinclude} reach.yaml
:language: yaml
:start-at: "experiment:"
```

A schedule entry moves one live Param through its `points`: pairs of a time [s] from the start
and a value. The Param is named as in `controller.set` ([Parameters](parameters.md)). Between
points the value moves in a straight line, or jumps with `interpolation: step`. Before the first
point the Param keeps its own value, and after the last it holds the last one. Here the goal
waits half a second, then walks to one side and to the other.

A `swap` entry blends to another controller over its `duration`, with the quintic blend of
`vmc.control.SwapController`. `swaps` lists the controllers to swap to, each named by its key and
with the same elements as `controller`:

```{literalinclude} reach.yaml
:language: yaml
:start-at: "swaps:"
:end-before: "experiment:"
```

The gentle controller's spring pulls the tip towards the middle with a force of at most 0.5 N
on each axis. We follow the tip and the goal along $x$ and $z$:

```{code-cell} python
from virtualmodelcontrol.control import Schedule

kin = vmc.Kinematics(experiment.robot)
tip = np.array([kin.position(q, 1.0) for q in rows["q"]])
t = rows["t"].ravel()
walk, swap = experiment.spec["experiment"]["schedule"]
path = Schedule("goal", walk["points"])
start = np.array([0.0, 0.0, 0.435])  # [m], the goal before the walk
middle = np.array([0.0, 0.0, 0.40])  # [m], the gentle spring's goal


def goal_at(time):  # [m], the goal pulling the tip at a time [s]
    if time >= swap["at"]:
        return middle
    return path.value(time) if time >= walk["points"][0][0] else start


goal = np.array([goal_at(x) for x in t])
walking = (t >= walk["points"][0][0]) & (t < swap["at"])

fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
for ax, k, axis in zip(axes, (0, 2), "xz"):
    ax.axvspan(swap["at"], swap["at"] + swap["duration"], color="0.9")
    ax.plot(t, 100 * goal[:, k], label="goal")
    ax.plot(t, 100 * tip[:, k], "--", label="tip")
    ax.set_ylabel(f"${axis}$ [cm]")
axes[0].legend(loc="upper right", fontsize=18)
axes[1].set_xlabel("time [s]");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue
from virtualmodelcontrol.robots import helyx

first_gap = 100 * float(np.abs(tip[walking, 0] - goal[walking, 0]).max())  # [cm]
glue("gap", first_gap, display=False)
glue("z_end", 100 * float(tip[-1, 2] - middle[2]), display=False)
assert abs(tip[-1, 0]) < 1e-3, "the tip is not back in the middle along x: rewrite"

# The same experiment written in Python gives the same run, bit for bit.
arm = helyx.add_dynamics(helyx.arm("145-145-145"))
point = arm.point(s=1.0)
ctrl, gentle = vmc.Mechanism("ctrl"), vmc.Mechanism("gentle")
ctrl.add("reach", vmc.LinearSpring(point - vmc.Ref("goal", value=start), 300.0))
gentle.add("reach", vmc.TanhSpring(point - [0.0, 0.0, 0.40], 300.0, 0.5))
for m in (ctrl, gentle):
    m.add("damp", vmc.LinearDamper(point, 5.0))
    m.add("gravity", vmc.GravityCompensation(arm))
first, second = (
    vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, m))) for m in (ctrl, gentle)
)
python = vmc.control.ScheduledController(
    first, [Schedule("ctrl.reach.goal", walk["points"])], swaps=[(4.0, second, 1.0)]
)
again = vmc.sim.run(vmc.sim.ModelPlant(arm), python, vmc.sim.SimClock(1 / 330), T=6.0)
for name, values in again.arrays().items():
    assert np.array_equal(values, rows[name]), name
```

The tip follows the walking goal within {glue:text}`gap:.1f` cm along $x$. Along $z$ it stays
above the goal, held by the arm's own stiffness, as in
[your first controller](first-controller.md). From 4 s the gentle controller takes over within a
second. Its spring brings the tip back to the middle along $x$, but with 0.5 N it cannot shorten
the arm: the tip ends {glue:text}`z_end:.1f` cm above its goal.

```{code-cell} python
:tags: [remove-output]
from virtualmodelcontrol import viz

def draw(ax, row):
    viz.draw_goal(ax, goal_at(row["t"]))

viz.animate(experiment.robot, rows, "configurations.mp4", draw=draw,
            trace=1.0)
```

```{video} configurations.mp4
:caption: The goal (red cross) walks from side to side and the tip follows it; from 4 s the gentle controller pulls the tip towards the middle.
```

## Save it back

`save` writes the configuration with the current values of the controllers' Params, so a value
changed in Python reaches the file. Here the spring gets twice the stiffness:

```{code-cell} python
experiment.mechanism.components["reach"].stiffness.value = 600.0  # [N/m]
experiment.save("reach-stiffer.yaml")
stiffer = vmc.config.load("reach-stiffer.yaml")
stiffer.spec["controller"]["elements"]["reach"]
```

```{code-cell} python
tip = np.array([kin.position(q, 1.0) for q in stiffer.run().arrays()["q"]])
gap = 100 * np.abs(tip[walking, 0] - goal[walking, 0]).max()  # [cm], x
float(gap)
```

```{code-cell} python
:tags: [remove-cell]
assert gap < first_gap, "the stiffer spring does not follow closer: rewrite the text"
```

The stiffer spring follows the goal more closely. `save` writes the file anew, without the
comments of the original. It saves the robot as its template call. `controller.set` changes
only the running controller's own copy of a value ([Parameters](parameters.md)), so only the
values of the Params themselves reach the file.

## What a file can name

Every name in a file comes from the library's registry, so a project adds its own robots,
elements or coordinate kinds by registering them ([Extend the library](extend.md)). These are
the library's own:

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown
from virtualmodelcontrol.core import registry

kinds = {
    "robot": "A robot's `template`",
    "dynamics": "A simulation's `dynamics`",
    "output": "An `output` stage",
    "controller": "A controller's `template`",
    "initial_state": "`z0`, the virtual state from the first reading",
    "component": "An element's `type`",
    "coordinate": "A coordinate's kind",
}
lines = []
for kind, where in kinds.items():
    names = ", ".join(f"`{name}`" for name in registry.names(kind))
    lines.append(f"- {where}: {names}.")
Markdown("\n".join(lines))
```
