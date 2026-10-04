---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft arm: reach, avoid, shape

In this example we control the Helyx soft arm in simulation, mounted on its side. First its
tip reaches a point, then its body bends around an obstacle on the way, and finally a few
springs shape the whole arm. The [hanging soft arm](hanging-arm.md) does the same with an arm
that hangs from its base.

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
schematic.figure("145-145-145");
```

The drawing shows the `145-145-145` arm, mounted on its side: gravity acts along $-y$, out
of the drawing, so the arm bends in a horizontal plane. The arc parameter $s$ runs from 0 at
the base to 1 at the tip, uniformly in arc length, and `arm.point(s=...)` gives any point of
the body.

Each segment's section, seen along its $z$ axis, shows where its tendons sit and the index of
the motor that pulls each one in the motor vector. A positive motor angle pulls its tendon.

```{code-cell} python
:tags: [remove-input]
from virtualmodelcontrol.robots import helyx
schematic.sections(helyx.arm("145-145-145").params);
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

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
params.table(arm, {
    "seg1.L0": "rest length of segment 1 (also `seg2.L0`, `seg3.L0`)",
    "seg1.d": "distance of the tendons from the backbone",
    "seg1.delta": "angles of the three tendons around segment 1",
    "seg1.r": "spool radius of the motors",
    "efficiency.c1": "linear coefficient of the delivered over commanded motor torque",
    "gravity": "gravity in the base frame, as mounted",
    "m1.mass": "mass of segment 1, lumped at its middle",
    "m1.s": "where that mass sits on the arm",
    "stiffness.stiffness": "stiffness of the arm in $\\Delta$, one value per axis",
    "damping.damping": "damping of the arm in $\\Delta$, one value per axis",
}, degrees=("seg1.delta",))
```

The stiffness and damping come from an identification on a real Helyx arm, and both
single-arm examples use them; only the simulator does. They were fitted to the torques the
motors were commanded, so the simulated arm takes each torque as the controller sends it: its
efficiency is 1, the default of every template.
The efficiency of the tendons measured against a load cell, `helyx.EFFICIENCY`
({glue:text}`eta:.0f` %), matters for the forces the real arm exerts on its surroundings; the
[efficiency page](../concepts/efficiency.md) explains when to use it.

On the real arm, `helyx.output_stage()` adds a small pretension to every motor command
(0.010 N·m per radian of motor angle). The simulations below leave it out.

## Reach a point

We pull the tip ($s = 1$) to a goal with a spring, damp the tip's motion and cancel the weight
of the segments.

```{code-cell} python
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
goal = np.array([0.15, 0.0, 0.35])  # [m]
tip = arm.point(s=1.0)

def reach(arm, goal):
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

log_reach = simulate(arm, reach(arm, goal))
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
assert distance.min() >= distance[-1] - 1e-6, "the tip overshoots"
glue("reach_start", float(distance[0]), display=False)
glue("reach_end", float(distance[-1]), display=False)
glue("reach_99", float(t[np.nonzero(outside)[0][-1] + 1]), display=False)
assert np.abs(path[:, 1]).max() < 1e-4, "the tip leaves the plane"
glue("eta", 100 * helyx.EFFICIENCY, display=False)
```

The tip starts {glue:text}`reach_start:.0f` cm from the goal, comes in without overshooting
and is within 1 % of its final distance after {glue:text}`reach_99:.1f` s. It stops
{glue:text}`reach_end:.1f` cm short of the goal, where the virtual spring balances the arm's own
stiffness; a stiffer spring brings it closer. Gravity compensation cancels the arm's weight,
so the arm stays in its horizontal plane.

## Avoid an obstacle

An obstacle sits between the arm and a goal. Six repulsive springs along the body push it
away: each acts on the vector from the obstacle to a point of the arm, and its force fades
with distance.

```{code-cell} python
:tags: [remove-output]
target = np.array([0.2, 0.0, 0.3])  # [m]
obstacle = np.array([0.06, 0.0, 0.2])  # [m]
radius = 0.035  # [m], for the drawing

log_free = simulate(arm, reach(arm, target))
ctrl = reach(arm, target)
for i, s in enumerate(np.linspace(0.4, 0.9, 6)):
    away = arm.point(s=s) - obstacle
    ctrl.add(f"avoid{i}", vmc.GaussianSpring(away, 4000.0, 0.05))
log_avoid = simulate(arm, ctrl)

def draw_obstacle(ax, row=None):
    disc = plt.Circle((obstacle[0], obstacle[2]), radius,
                      color=viz.PALETTE[1], alpha=0.3)
    ax.add_patch(disc)

viz.animate(arm, log_avoid, "soft-arm-avoid.mp4", springs=[(1.0, target)],
            trace=1.0, draw=draw_obstacle)
```

```{video} soft-arm-avoid.mp4
:caption: The tip reaches for the goal while six repulsive springs keep the body off the obstacle (red).
```

The two runs end in different shapes:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.6, 5.6))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, log_free.arrays()["q"][-1],
               color=viz.PALETTE[5], label="reach")
viz.draw_robot(ax, arm, log_avoid.arrays()["q"][-1],
               label="reach and avoid")
viz.draw_goal(ax, target)
draw_obstacle(ax)
viz.label_axes(ax)
ax.legend(loc="upper left");
```

```{code-cell} python
:tags: [remove-cell]
def closest(log):  # the body's closest approach to the obstacle's centre [cm]
    points = [kin.position(q, s) for q in log.arrays()["q"][::10]
              for s in np.linspace(0.0, 1.0, 41)]
    return 100 * min(np.linalg.norm(p - obstacle) for p in points)

def miss(log):  # the tip's final distance to the goal [cm]
    return 100 * np.linalg.norm(kin.position(log.arrays()["q"][-1], 1.0) - target)

assert closest(log_free) < 100 * radius < closest(log_avoid)
glue("free_miss", float(miss(log_free)), display=False)
glue("avoid_miss", float(miss(log_avoid)), display=False)
glue("clearance", float(closest(log_avoid) - 100 * radius), display=False)
```

Without the repulsive springs the body passes through the obstacle. With them it bends around
it, at least {glue:text}`clearance:.1f` cm from its surface, and the tip stops
{glue:text}`avoid_miss:.1f` cm from the goal instead of {glue:text}`free_miss:.1f` cm, where the
pull of the goal and the push of the obstacle balance.

## Shape the whole body

Springs can act anywhere along the arm. Here one holds the middle of the arm at a point, and
another acts on the tip's position along $z$ only: it pulls on the projection of the tip on
the $z$ axis.

```{code-cell} python
middle = np.array([-0.05, 0.0, 0.2])  # [m]
reach_z = 0.38  # [m]

along_z = vmc.Projection(tip - [0, 0, reach_z], [0, 0, 1])

ctrl = vmc.Mechanism("ctrl")
ctrl.add("middle", vmc.LinearSpring(arm.point(s=0.5) - middle, 600.0))
ctrl.add("reach_z", vmc.LinearSpring(along_z, 600.0))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
q = simulate(arm, ctrl).arrays()["q"][-1]

fig, ax = plt.subplots(figsize=(4.4, 5.2))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, q)
viz.draw_goal(ax, middle)
ax.axhline(reach_z, color=viz.PALETTE[2], ls="--", label="tip goal in $z$")
viz.label_axes(ax)
ax.legend(loc="lower left");
```

```{code-cell} python
:tags: [remove-cell]
glue("middle_off", 100 * float(np.linalg.norm(kin.position(q, 0.5) - middle)), display=False)
glue("z_off", 100 * float(kin.position(q, 1.0)[2] - reach_z), display=False)
```

The middle of the arm settles {glue:text}`middle_off:.1f` cm from its goal and the tip
{glue:text}`z_off:.1f` cm beyond the dashed line, free to slide along it: as in the first run,
the arm's own stiffness holds both a little short.
