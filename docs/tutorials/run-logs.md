---
file_format: mystnb
kernelspec:
  name: python3
---

# Run logs

In this tutorial we record a run of the soft arm in detail, save it, load it back, replay its
commands on a model that is slightly wrong, compare the two runs, and let a configuration file
save every run it makes.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## What a run records

`vmc.sim.run` returns a `RunLog`: every signal it recorded, one row per control step. We run the
controller of [your first controller](first-controller.md) and ask `record` for everything it
can add:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
goal = np.array([0.08, 0.0, 0.40])  # [m]
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 300.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))

plant = vmc.sim.ModelPlant(arm)
clock = vmc.sim.SimClock(dt=1 / 330)  # [s]
log = vmc.sim.run(plant, controller, clock, T=2.0,
                  record=vmc.sim.RECORDS)
rows = log.arrays()
len(rows["t"])
```

Without `record`, a log holds the signals down to `z` in this list. `record` adds the others,
one group at a time: `"params"`, `"elements"`, `"energy"` and `"robot"`. The signals are:

- `t`: the time of the step [s].
- `motor_position`, `motor_velocity`: the measured motor angles [rad] and rates [rad/s].
- `q`, `v`: the configuration and velocity, when the plant reports them (a simulator does).
- `motor_torque`: the torques sent to the motors [N·m].
- `law_torque`: the torques the controller's elements asked for, before the output stages
  [N·m].
- `z`: the controller's virtual states, when it has any.
- `param/<name>`: a live Param, in its own shape (`"params"`).
- `element/<element>/y`, `ydot`, `force`: an element's coordinate, its rate and its force
  (`"elements"`).
- `element/<element>/torque`: the element's share of the motor torques [N·m], before the
  output stages (`"elements"`).
- `energy/stored`, `energy/kinetic`: the controller's stored and kinetic energy [J]
  (`"energy"`).
- `power/port`, `power/dissipation`, `power/source`: the controller's powers [W]
  (`"energy"`).
- `robot/<component>/y`, `ydot`, `force`, `torque`: the same of each spring, damper and contact
  of the simulated robot itself, as it feels them (`"robot"`; a real robot reports none).

```{code-cell} python
:tags: [remove-cell]
named = {"t", "motor_position", "motor_velocity", "q", "v", "motor_torque",
         "law_torque"}
elements = {f"element/ctrl.{e}/{k}" for e in ("reach", "damp", "gravity")
            for k in ("y", "ydot", "force", "torque")}
params = {"param/ctrl.reach.stiffness", "param/ctrl.reach.ref",
          "param/ctrl.damp.damping"}
energy = {"energy/stored", "energy/kinetic", "power/port",
          "power/dissipation", "power/source"}
robot = {f"robot/{c}/{k}" for c in ("stiffness", "damping", "gravity")
         for k in ("y", "ydot", "force", "torque")}
assert set(rows) == named | elements | params | energy | robot, "out of date"
```

The spring's force is minus its stiffness times its coordinate `y`, the deflection. The shares
of the elements add up to the torque of the law. We follow the forces, and the torques on one
motor:

```{code-cell} python
t = rows["t"].ravel()
spring = np.linalg.norm(rows["element/ctrl.reach/force"], axis=1)
damper = np.linalg.norm(rows["element/ctrl.damp/force"], axis=1)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
top.plot(t, spring, label="spring")
top.plot(t, damper, "--", label="damper")
top.set_ylabel("force [N]")
top.legend(loc="upper right", fontsize=18)
for element in ("reach", "damp", "gravity"):
    share = rows[f"element/ctrl.{element}/torque"][:, 2]
    bottom.plot(t, 1000 * share, label=element)
bottom.plot(t, 1000 * rows["law_torque"][:, 2], "k:", label="sum")
bottom.set_xlabel("time [s]")
bottom.set_ylabel(r"motor 2 [mN$\cdot$m]")
bottom.legend(loc="lower right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

shares = sum(rows[f"element/ctrl.{e}/torque"] for e in ("reach", "damp", "gravity"))
add_up = float(np.abs(shares - rows["law_torque"]).max())
glue("add_up", add_up, display=False)
assert add_up < 1e-12, "the shares no longer add up: rewrite"
stretch = rows["element/ctrl.reach/y"]
assert np.allclose(rows["element/ctrl.reach/force"], -300.0 * stretch)
glue("spring_start", float(spring[0]), display=False)
glue("spring_end", float(spring[-1]), display=False)
glue("damper_peak", float(damper.max()), display=False)
glue("damper_at", 1000 * float(t[np.argmax(damper)]), display=False)
glue("gravity_end", -1000 * float(rows["element/ctrl.gravity/torque"][-1, 2]),
     display=False)
glue("law_end", -1000 * float(rows["law_torque"][-1, 2]), display=False)
```

The spring pulls with {glue:text}`spring_start:.0f` N at the start and settles at
{glue:text}`spring_end:.1f` N: the arm's own stiffness holds the tip short of the goal. The
damper brakes the first moves, with up to {glue:text}`damper_peak:.1f` N after
{glue:text}`damper_at:.0f` ms. At the end, gravity compensation carries
{glue:text}`gravity_end:.1f` of the {glue:text}`law_end:.1f` mN·m (in magnitude) that the law
sends to motor 2. The shares add up to the law's torque within {glue:text}`add_up:.0e` N·m.

A step on which the guard stops the controller (a measurement is missing or not a number) logs
the zero torque it sent, and NaN for what the controller did not compute. A signal that
exists for part of a run only, such as the elements of a controller that takes over later,
is NaN elsewhere. So every signal keeps one row per step. A loop of your own adds its steps
the same way, with `log.step(t=..., q=...)`.

## Save and load

`save` writes the log as one compressed NumPy file, and `RunLog.load` reads it back:

```{code-cell} python
path = log.save("reach.npz")
again = vmc.sim.RunLog.load(path)
again.meta["params"]["ctrl.reach.stiffness"]
```

```{code-cell} python
:tags: [remove-cell]
for name, values in rows.items():
    np.testing.assert_array_equal(again.arrays()[name], values, err_msg=name)
assert again.meta == log.meta and list(again.meta) == ["library", "start", "params"]
```

The file holds the signals and `meta`, which says what the run was: the library's version, the
start time, and the value every Param had at the start (the controller's own, if `set` had
changed it). A plant that has a hardware profile as `plant.profile` adds it. A real-time run
also fills `info` with its statistics (steps, rate, overruns). `save` writes a temporary file
and renames it, so an interrupted save never leaves half a log. It refuses a name that is
taken, unless you pass `overwrite=True`.

A saved log is a plain NumPy archive, so `np.load("reach.npz")["q"]` reads a signal without the
library.

## Export to CSV

`to_csv` writes one line per step and one column per entry, named `name_i` for the entries of
a vector and `name_i_j` for those of a matrix, with all the digits of each number:

```{code-cell} python
log.to_csv("reach.csv")
header = open("reach.csv").readline().strip().split(",")
header[:3], len(header)
```

The CSV holds the steps only; `meta` and `info` stay in the `.npz`.

## Replay and compare

A log of the real arm holds the commands it received and what its motors measured. `replay`
sends the same commands to a simulator, each held for as long as it was in the run, and returns
what the simulator did. `compare` says how far apart the two runs are. Replayed on the model that
made the log, the commands give the run back:

```{code-cell} python
same = vmc.sim.replay(log, vmc.sim.ModelPlant(arm))
check = vmc.sim.compare(log, same, names=["motor_position", "q"])
max(entry["max"] for entry in check.values())
```

Now the model is wrong: its joints are 30 % stiffer than the arm's. We treat the log as the real
arm's:

```{code-cell} python
stiff = helyx.add_dynamics(helyx.arm("145-145-145"),
                           stiffness=1.3 * helyx.SIM_STIFFNESS)
replayed = vmc.sim.replay(log, vmc.sim.ModelPlant(stiff))
vmc.sim.compare(log, replayed, names=["motor_position"])
```

`compare` gives the largest (`max`) and the root-mean-square (`rms`) difference of each signal
the two logs hold (`names` picks some), reading the second log at the first log's times. A
log of a real arm has no `q` and `v`; `replay` then starts the simulator from the first motor
reading, through the transmission's exact inverse.

```{code-cell} python
fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
angle = rows["motor_position"][:, 0]
model = replayed.arrays()["motor_position"][:, 0]
top.plot(t, angle, label="arm")
top.plot(t, model, "--", label="model")
top.set_ylabel("motor 0 [rad]")
top.legend(loc="lower right", fontsize=18)
bottom.plot(t, model - angle)
bottom.set_xlabel("time [s]")
bottom.set_ylabel("difference [rad]");
```

```{code-cell} python
:tags: [remove-cell]
gap = vmc.sim.compare(log, replayed, names=["motor_position"])["motor_position"]
glue("true_gap", float(max(entry["max"] for entry in check.values())), display=False)
glue("gap_max", gap["max"], display=False)
glue("gap_rms", gap["rms"], display=False)
glue("travel", float(np.ptp(rows["motor_position"], axis=0).max()), display=False)
assert check["q"]["max"] < 1e-9 and gap["max"] > 1e3 * check["q"]["max"]
```

On the true model the largest difference is {glue:text}`true_gap:.0e`. On the wrong one the
stiffer joints hold the arm back. Under the same torques its motors turn up to
{glue:text}`gap_max:.2f` rad less than the arm's ({glue:text}`gap_rms:.2f` rad RMS), against a
travel of {glue:text}`travel:.1f` rad. The differences say what to correct in the model: the
[hanging arm](../examples/hanging-arm.md) fits its stiffness and damping from logged runs.

## Runs from a configuration file

A configuration ([Experiments in files](configurations.md)) saves each of its runs when its
`experiment` section has `run` settings:

```{literalinclude} run-settings.yaml
:language: yaml
```

`name` is the file name (by default the start time of the run), `folder` is relative to the
configuration file, and `record` lists what to add to the log. We add these lines, indented by two
spaces as in the file, at the end of the `experiment` section of that tutorial's file and run it.
Download [reach.yaml](https://github.com/vigno0405/VirtualModelControl/blob/main/docs/tutorials/reach.yaml) and [run-settings.yaml](https://github.com/vigno0405/VirtualModelControl/blob/main/docs/tutorials/run-settings.yaml) into the
folder where you run the code:

```{code-cell} python
from pathlib import Path

settings = Path("run-settings.yaml").read_text()
Path("walk.yaml").write_text(Path("reach.yaml").read_text() + settings)
experiment = vmc.config.load("walk.yaml")
walk = experiment.run()
sorted(path.name for path in Path("logs").iterdir())
```

The run starts only if its name is free. A second `run` with the same `name` raises a
`FileExistsError` before anything moves, so a finished log is never replaced by mistake. With
`overwrite: true` in the settings, it replaces the file instead.

The log holds the configuration it started from, as YAML text in `meta`. Writing it to a file
brings the experiment back as it began, whatever has changed in the original file since:

```{code-cell} python
Path("again.yaml").write_text(walk.meta["configuration"])
before = vmc.config.load("again.yaml")  # the experiment of the log
```

```{code-cell} python
:tags: [remove-cell]
assert before.to_dict() == experiment.to_dict()
```

This experiment moves its goal on a schedule and swaps to a gentler controller at 4 s, which
takes over within a second. The log records the goal Param as it walked, and the elements of
each controller, with NaN where a controller was not the one running:

```{code-cell} python
saved = vmc.sim.RunLog.load("logs/walk.npz")
walked = saved.arrays()
time = walked["t"].ravel()

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
top.plot(time, 100 * walked["param/ctrl.reach.goal"][:, 0])
top.set_ylabel("goal $x$ [cm]")
for name in ("ctrl", "gentle"):
    force = np.linalg.norm(walked[f"element/{name}.reach/force"], axis=1)
    bottom.plot(time, force, label=name)
bottom.set_xlabel("time [s]")
bottom.set_ylabel("spring force [N]")
bottom.legend(loc="upper left", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
first = np.isfinite(walked["element/ctrl.reach/force"][:, 0])
second = np.isfinite(walked["element/gentle.reach/force"][:, 0])
assert (first ^ second).all() and not (first & second).any()
assert np.isnan(walked["param/ctrl.reach.goal"][second]).all()
glue("handover", float(time[second][0]), display=False)
glue("peak", float(np.linalg.norm(walked["element/ctrl.reach/force"][first],
                                  axis=1).max()), display=False)
```

The log holds the first controller's goal and elements until {glue:text}`handover:.1f` s, when
the swap is done, and the gentle controller's from then on. Before the handover the first
spring pulls with up to {glue:text}`peak:.0f` N.
