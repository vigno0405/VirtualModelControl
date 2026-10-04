---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft arm: reach, avoid, shape

In this example we control the Helyx soft arm in simulation. First its tip reaches a point,
then its body bends around an obstacle on the way, and finally a few springs shape the whole
arm.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The Helyx arm has three continuum segments. Three tendons run along each segment, and each
tendon is wound on its own motor, so the arm has nine motors. The template describes each
segment by piecewise constant curvature: its configuration is $\Delta = (D_x, D_y, D_l)$
[m], the tendon-length differences that bend it and its elongation.

```{code-cell} python
:tags: [remove-input]
from schematics import helyx as schematic
schematic.figure("145-290-290");
```

The drawing shows the `145-290-290` arm, which hangs from its base, so $z$ points down. The
arc parameter $s$ runs from 0 at the base to 1 at the tip, uniformly in arc length, and
`arm.point(s=...)` gives any point of the body.

Each segment's section, seen along its $z$ axis, shows where its tendons sit and the index of
the motor that pulls each one in the motor vector. A positive motor angle pulls its tendon.

```{code-cell} python
:tags: [remove-input]
from virtualmodelcontrol.robots import helyx
schematic.sections(helyx.arm("145-290-290").params);
```

There are three geometries. Their segment lengths, their mounting and the sign of their
encoders against that convention are:

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown

rows = ["| Geometry | Segments, base to tip | Gravity in the base frame | Encoder sign |", "|---|---|---|---|"]
for name, spec in helyx.GEOMETRIES.items():
    lengths = ", ".join(f"{1000 * L:.0f}" for L in spec["L0"]) + " mm"
    gravity = "[" + ", ".join(f"{g:g}" for g in spec["gravity"]) + "] m/s²"
    rows.append(f"| `{name}` | {lengths} | {gravity} | {helyx.ENCODER_SIGN[name]:+.0f} |")
Markdown("\n".join(rows))
```

The template takes its geometry as arguments, so the same function builds an arm of any
segment lengths, radii, tendon angles or masses, with any number of segments:

```{code-cell} python
import numpy as np
from virtualmodelcontrol.robots import helyx

short = helyx.arm(
    lengths=(0.2, 0.2),  # [m], two segments
    tendon_angles=np.radians([[0, 120, -120], [60, 180, -60]]),
)
short.actuation.motor_sizes(short.space)  # motor angles and rates
```

Every number of the template is a `Param` that you can also change later. These are the ones
of the arm used below, read from the template itself:

```{code-cell} python
:tags: [remove-input]
from schematics import params

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
params.table(arm, {
    "seg1.L0": "rest length of segment 1 (also `seg2.L0`, `seg3.L0`)",
    "seg1.d": "distance of the tendons from the backbone",
    "seg1.delta": "angles of the three tendons around segment 1",
    "seg1.r": "spool radius of the motors",
    "efficiency": "delivered over commanded motor torque",
    "gravity": "gravity in the base frame, as mounted",
    "m1.mass": "mass of segment 1, lumped at its middle",
    "m1.s": "where that mass sits on the arm",
    "stiffness.stiffness": "stiffness of the arm in $\\Delta$, one value per axis",
    "damping.damping": "damping of the arm in $\\Delta$, one value per axis",
}, degrees=("seg1.delta",))
```

The stiffness and damping come from an identification on a real arm. They are referred to the
commanded motor torque: the identified values were divided by the measured transmission
efficiency, 0.12, so the template keeps `efficiency` at 1. Only the simulator uses them.

On the real arm, `helyx.output_stage()` adds a small pretension to every motor command
(0.010 N·m per radian of motor angle). The simulations below leave it out.

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
glue("reach_start", float(distance[0]), display=False)
glue("reach_end", float(distance[-1]), display=False)
glue("reach_99", float(t[np.argmax(travel >= 0.99 * travel[-1])]), display=False)
```

The tip starts {glue:text}`reach_start:.0f` cm from the goal and covers 99 % of its way in
{glue:text}`reach_99:.1f` s. It stops {glue:text}`reach_end:.1f` cm short of the goal, where
the virtual spring balances the arm's own stiffness; a stiffer spring brings it closer.

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

viz.animate(arm, log_avoid, "soft-arm-avoid.mp4", springs=[(1.0, goal)],
            trace=1.0, draw=draw_obstacle, invert=True)
```

```{video} soft-arm-avoid.mp4
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

Springs can act anywhere along the arm. Here one holds the middle of the arm at a point, and
another acts on the tip's height only: it pulls on the projection of the tip on the vertical.

```{code-cell} python
middle = np.array([0.08, 0.0, 0.33])  # [m]
height = 0.62  # [m]

vertical = vmc.Projection(tip - [0, 0, height], [0, 0, 1])

ctrl = vmc.Mechanism("ctrl")
ctrl.add("middle", vmc.LinearSpring(arm.point(s=0.5) - middle, 800.0))
ctrl.add("height", vmc.LinearSpring(vertical, 800.0))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
q = simulate(arm, ctrl).arrays()["q"][-1]

fig, ax = plt.subplots(figsize=(4.4, 5.2))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, q)
viz.draw_goal(ax, middle)
ax.axhline(height, color=viz.PALETTE[2], ls="--", label="tip height")
ax.invert_yaxis()
viz.label_axes(ax)
ax.legend(loc="lower left");
```

The middle of the arm sits on its goal and the tip on the dashed line, free to settle anywhere
along it.
