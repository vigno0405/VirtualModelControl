---
file_format: mystnb
kernelspec:
  name: python3
---

# Parameters

In this tutorial we look at the parameters of the soft arm and of a controller, choose which
ones the compiled controller keeps live, and move a goal while the arm runs.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## Read and change a Param

Every number of a robot or of a controller is a `Param`: a value with a unit, bounds and a
scope. Here is the rest length of the arm's first segment:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
length = arm.params["seg1.L0"]
length, length.bounds
```

`vmc.Kinematics` evaluates at the current values, so lengthening the first segment moves the tip
of the straight arm by the same amount:

```{code-cell} python
kin = vmc.Kinematics(arm)
before = kin.position(np.zeros(9), 1.0)  # the tip [m]
length.value = 0.150  # [m]
after = kin.position(np.zeros(9), 1.0)
length.value = 0.145  # back to the real arm
1000 * (after - before)  # [mm]
```

## Scopes

The scope says how often a value may change: `fixed` never, `design` with the hardware
(lengths, masses, transmissions), `episode` between runs (attachment points, directions, joint
ranges) and `stage` at any control step (stiffnesses, dampings, goals). Each component gives
its Params their scopes:

```{code-cell} python
goal = vmc.Ref("goal", value=[0.0, 0.0, 0.40])  # a named goal [m]
tip = arm.point(s=1.0)

ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - goal, 600.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
{name: p.scope for name, p in system.params.items() if "ctrl" in name}
```

In a system, names read `<mechanism>.<component>.<name>`. The tip's arc parameter belongs to
the first component that uses the point, `reach`. Gravity compensation shares the robot's own
`gravity` Param, which keeps the robot's name, `arm.gravity`.

## What compile keeps live

`compile` keeps the `stage` Params as live inputs of the compiled function and folds every other
Param in as a constant, at its current value. Constants make the function faster. To change
one, change its Param and compile again.

```{code-cell} python
law = vmc.compile(system)
law.live
```

`runtime=` keeps more Params live, by name or by glob pattern. Here, every attachment point of
the controller:

```{code-cell} python
vmc.compile(system, runtime=["ctrl.*.s"]).live
```

## Change a value while running

`controller.set` changes live values, all at once, and returns the jump of the controller's
energy, at the last measured state. It changes the controller's own copy: the `Param` objects
keep their values, and a new controller starts from those.

```{code-cell} python
controller = vmc.VMCController(law)
plant = vmc.sim.ModelPlant(arm)
controller.step(plant.t, plant.read())  # the arm at rest, straight
stored = controller.energy()  # [J]
jump = controller.set({"ctrl.reach.stiffness": 1200.0})
stored, jump
```

Doubling the stiffness doubles the energy of the stretched spring, so the jump equals what it
stored. Goals are live Params too. A plain list on either side of `-` becomes a Param named `ref`:

```{code-cell} python
other = vmc.Mechanism("ctrl")
other.add("reach", vmc.LinearSpring(tip - [0.0, 0.0, 0.40], 600.0))
list(other.params)
```

In a system that is `ctrl.reach.ref`. A goal that changes while running reads better with a
name of its own, given by `vmc.Ref("goal", ...)` as above: `ctrl.reach.goal`.

## Move the goal

The goal swings from side to side while the arm runs. The loop is the one `vmc.sim.run`
runs (read, step, write, advance; its guard is left out) with one `set` before each step. The
log also records the goal, to draw it later.

```{code-cell} python
controller = vmc.VMCController(law)  # a new controller: 600 N/m again
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.RunLog()
dt = 1 / 330  # [s]
controller.reset(plant.t, plant.read())
for _ in range(round(4.0 / dt)):
    target = [0.12 * np.sin(np.pi * plant.t), 0.0, 0.40]  # [m]
    controller.set({"ctrl.reach.goal": target})
    meas = plant.read()
    plant.write(controller.step(plant.t, meas))
    log.step(t=plant.t, q=meas["q"], goal=target)
    plant.advance(dt)
```

We follow the tip and the goal from side to side:

```{code-cell} python
rows = log.arrays()
tip_x = [kin.position(q, 1.0)[0] for q in rows["q"]]

fig, ax = plt.subplots()
ax.plot(rows["t"], 100 * rows["goal"][:, 0], label="goal")
ax.plot(rows["t"], 100 * np.array(tip_x), "--", label="tip")
ax.set_xlabel("time [s]")
ax.set_ylabel("$x$ [cm]")
ax.legend(loc="lower left", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

t = rows["t"].ravel()
paths = np.array([kin.position(q, 1.0) for q in rows["q"]])
second = t >= 2.0  # the second swing, after the start
peak_goal = t[second][np.argmax(rows["goal"][second, 0])]
peak_tip = t[second][np.argmax(paths[second, 0])]
gap = np.linalg.norm(paths - rows["goal"], axis=1)[second]
glue("lag", 1000 * float(peak_tip - peak_goal), display=False)
glue("gap", 100 * float(gap.max()), display=False)
```

The tip follows the goal about {glue:text}`lag:.0f` ms behind and stays within
{glue:text}`gap:.1f` cm of it. The gap is mostly along $z$: the tip stays short of the goal,
where the spring balances the arm's own stiffness.

## Animate

`viz.animate` calls `draw` at every frame with the logged values of that step, so the goal
and its spring move with the run.

```{code-cell} python
:tags: [remove-output]
def draw(ax, row):
    viz.draw_spring(ax, kin.position(row["q"], 1.0), row["goal"])
    viz.draw_goal(ax, row["goal"])

viz.animate(arm, log, "parameters.mp4", draw=draw, trace=1.0)
```

```{video} parameters.mp4
:caption: The goal (red cross), moved by controller.set at every step, and the tip following it on its spring.
```
