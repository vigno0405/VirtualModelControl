---
file_format: mystnb
kernelspec:
  name: python3
---

# Hanging soft arm: reach, avoid, shape

In this example we control the Helyx soft arm that hangs from its base, the `145-290-290`
geometry, in simulation. First its tip reaches a point, then its body bends around an obstacle
on the way, and finally a few springs shape the whole arm. The [soft-arm example](soft-arm.md)
does the same with the arm mounted on its side.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The [soft-arm example](soft-arm.md) introduces the Helyx arm: its tendons, its motors and its
geometries. This one has segments of 145, 290 and 290 mm and hangs from its base, so $z$
points down:

```{code-cell} python
:tags: [remove-input]
from schematics import helyx as schematic
schematic.figure("145-290-290");
```

Its parameters, read from the template:

```{code-cell} python
:tags: [remove-input]
import numpy as np
from schematics import params
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
params.table(arm, {
    "seg1.L0": "rest length of segment 1 (also `seg2.L0`, `seg3.L0`)",
    "gravity": "gravity in the base frame, as mounted",
    "m1.mass": "mass of segment 1, lumped at its middle",
    "stiffness.stiffness": "stiffness of the arm in $\\Delta$, one value per axis",
    "damping.damping": "damping of the arm in $\\Delta$, one value per axis",
})
```

The stiffness and damping come from an identification on a real Helyx arm, the same values as
in the soft-arm example; only the simulator uses them. They were fitted to the torques the motors were commanded, so the simulated arm takes
each torque as the controller sends it: its efficiency is 1, the default of every template.

## Reach a point

We pull the tip ($s = 1$) to a goal with a spring, damp the tip's motion and cancel the weight
of the segments.

```{code-cell} python
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
goal = np.array([0.25, 0.0, 0.55])  # [m]
tip = arm.point(s=1.0)

def reach(arm):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - goal, 600.0))  # [N/m]
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))  # [N·s/m]
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return ctrl

def simulate(arm, ctrl, T=3.0):
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(arm)
    clock = vmc.sim.SimClock(dt=1 / 330)
    return vmc.sim.run(plant, controller, clock, T=T)

log_reach = simulate(arm, reach(arm))
```

The log holds the arm's configuration at every control step, so we can follow the tip:

```{code-cell} python
kin = vmc.Kinematics(arm)
rows = log_reach.arrays()
path = np.array([kin.position(q, 1.0) for q in rows["q"]])
distance = 100 * np.linalg.norm(path - goal, axis=1)  # [cm]

fig, ax = plt.subplots()
ax.plot(rows["t"], distance)
ax.set_xlabel("time [s]")
ax.set_ylabel("tip to goal [cm]");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

travel = distance[0] - distance
t = rows["t"].ravel()
outside = np.abs(travel - travel[-1]) > 0.01 * travel[-1]  # more than 1 % off the end
glue("reach_start", float(distance[0]), display=False)
glue("reach_end", float(distance[-1]), display=False)
glue("reach_99", float(t[np.nonzero(outside)[0][-1] + 1]), display=False)
```

The tip starts {glue:text}`reach_start:.0f` cm from the goal, comes in without overshooting
and is within 1 % of its final distance after {glue:text}`reach_99:.1f` s. It stops
{glue:text}`reach_end:.1f` cm short of the goal, where the virtual spring balances the arm's own
stiffness; a stiffer spring brings it closer.

## Avoid an obstacle

An obstacle sits between the arm and the goal. Six repulsive springs along the body push it
away: each acts on the vector from the obstacle to a point of the arm, and its force fades
with distance.

```{code-cell} python
:tags: [remove-output]
obstacle = np.array([0.12, 0.0, 0.44])  # [m]
radius = 0.035  # [m], for the drawing

ctrl = reach(arm)
for i, s in enumerate(np.linspace(0.4, 0.9, 6)):
    away = arm.point(s=s) - obstacle
    ctrl.add(f"avoid{i}", vmc.GaussianSpring(away, 4000.0, 0.05))
log_avoid = simulate(arm, ctrl)

def draw_obstacle(ax, row=None):
    disc = plt.Circle((obstacle[0], obstacle[2]), radius,
                      color=viz.PALETTE[1], alpha=0.3)
    ax.add_patch(disc)

viz.animate(arm, log_avoid, "hanging-arm-avoid.mp4", springs=[(1.0, goal)],
            trace=1.0, draw=draw_obstacle, invert=True)
```

```{video} hanging-arm-avoid.mp4
:caption: The tip reaches for the goal while six repulsive springs keep the body off the obstacle (red).
```

The two runs end in different shapes:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.6, 5.6))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, log_reach.arrays()["q"][-1],
               color=viz.PALETTE[5], label="reach")
viz.draw_robot(ax, arm, log_avoid.arrays()["q"][-1],
               label="reach and avoid")
viz.draw_goal(ax, goal)
draw_obstacle(ax)
ax.invert_yaxis()  # the arm hangs: z points down
viz.label_axes(ax)
ax.legend(loc="lower left");
```

Without the repulsive springs the body passes through the obstacle. With them it bends around
it, and the tip stops a little further from the goal, where the pull of the goal and the push
of the obstacle balance.

## Shape the whole body

Springs can pull on any point of the arm, not only on its tip, so together they set the shape
of the whole body. Here one spring pulls the middle of the arm ($s = 0.5$) to a point, the
cross, and a second one pulls the tip towards the dashed line at the height $z = 0.62$ m. The
second spring acts vertically only: it stretches with the tip's distance from the line, so it
does not mind where along the line the tip ends.

```{code-cell} python
middle = np.array([0.08, 0.0, 0.33])  # [m]
height = 0.62  # [m]

vertical = vmc.Projection(tip - [0, 0, height], [0, 0, 1])  # distance

ctrl = vmc.Mechanism("ctrl")
ctrl.add("middle", vmc.LinearSpring(arm.point(s=0.5) - middle, 800.0))
ctrl.add("height", vmc.LinearSpring(vertical, 800.0))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
q = simulate(arm, ctrl).arrays()["q"][-1]

fig, ax = plt.subplots(figsize=(4.4, 5.2))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, q)
viz.draw_goal(ax, middle, label="goal of the middle")
ax.axhline(height, color=viz.PALETTE[2], ls="--", label="line for the tip")
ax.invert_yaxis()
viz.label_axes(ax)
ax.legend(loc="lower left");
```

```{code-cell} python
:tags: [remove-cell]
glue("middle_off", 100 * float(np.linalg.norm(kin.position(q, 0.5) - middle)), display=False)
glue("height_off", 100 * float(kin.position(q, 1.0)[2] - height), display=False)
```

The middle of the arm stops {glue:text}`middle_off:.1f` cm from the cross and the tip
{glue:text}`height_off:.1f` cm below the line: as in the first run, the arm's own stiffness
holds both a little short of their goals. Nothing pulls the tip along the line, so where it
ends there follows from the shape the two springs give the body.
