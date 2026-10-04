---
file_format: mystnb
kernelspec:
  name: python3
---

# Hanging soft arm: identify it, reach around an obstacle

In this example we simulate the Helyx soft arm that hangs from its base, in two of its
experiments: its stiffness and damping are identified from step responses, and its tip reaches
a target around an obstacle, pulled by a force-limited spring and pushed by two repulsive
fields.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The arm has three continuum segments, of 145, 290 and 290 mm. Three tendons run along each
segment, and each tendon is wound on its own motor, so the arm has nine motors. The template
describes each segment by piecewise constant curvature: its configuration is
$\Delta = (D_x, D_y, D_l)$ [m], the tendon-length differences that bend it and its elongation.
The arm hangs from its base, so $z$ points down:

```{code-cell} python
:tags: [remove-input]
from schematics import helyx as schematic
schematic.figure("145-290-290");
```

The arc parameter $s$ runs from 0 at the base to 1 at the tip, uniformly in arc length, and
`arm.point(s=...)` gives any point of the body. Each segment's section, seen along its $z$
axis, shows where its tendons sit and the index of the motor that pulls each one in the motor
vector. A positive motor angle pulls its tendon.

```{code-cell} python
:tags: [remove-input]
from virtualmodelcontrol.robots import helyx
schematic.sections(helyx.arm("145-290-290").params);
```

Every number of the template is a `Param`. These are the arm's, read from the template:

```{code-cell} python
:tags: [remove-input]
from schematics import params

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
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

The stiffness and damping come from an identification on a real Helyx arm; only the simulator
uses them. They were fitted to the torques the motors were commanded, so the simulated arm
takes each torque as the controller sends it: its efficiency is 1, the default of every
template. The nine motors and their bus are in `helyx.hardware("145-290-290")`.

## Identify its stiffness and damping

The step experiment holds every motor at a baseline tension of 0.03 N·m, so the tendons stay
taut. Each motor in turn then pulls 0.04 N·m harder for 2 s, followed by each segment's three
motors together, with 2 s back at the baseline after each pull. The angles, rates and torques
of the run go to `identification.fit_stiffness_damping`, which knows the arm's masses and
gravity and fits its diagonal stiffness and damping by least squares, relative to the resting
baseline. Here the simulator plays the arm, so the fit can be checked against the values it
was given.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.identification import fit_stiffness_damping

plant = vmc.sim.ModelPlant(arm)
dt, baseline, pull = 1 / 330, np.full(9, 0.03), 0.04  # [s], [N·m]
log = vmc.sim.RunLog()

def hold(torque, seconds, record=True):
    for _ in range(round(seconds / dt)):
        if record:
            log.append(t=plant.t, q=plant.q, v=plant.v, u=torque)
        plant.write(vmc.Signals(plant.t, motor_torque=torque))
        plant.advance(dt)

hold(baseline, 3.0, record=False)  # settle under the baseline
hold(baseline, 0.5)  # the resting baseline
for motors in [[m] for m in range(9)] + [[0, 1, 2], [3, 4, 5], [6, 7, 8]]:
    u = baseline.copy()
    u[motors] += pull
    hold(u, 2.0)
    hold(baseline, 2.0)

known = helyx.arm("145-290-290")  # masses and gravity, no stiffness
known.add("gravity", vmc.Gravity(known))
K, D = fit_stiffness_damping(known, [log.arrays()], smoothing=11)
```

```{code-cell} python
names = [rf"$D_{{{c},{i}}}$" for i in (1, 2, 3) for c in "xyl"]
x = np.arange(9)
fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
top.bar(x - 0.2, helyx.SIM_STIFFNESS, 0.4, label="simulator")
top.bar(x + 0.2, K, 0.4, label="fitted")
top.set_ylabel("stiffness [N/m]")
top.legend(fontsize=14)
bottom.bar(x - 0.2, helyx.SIM_DAMPING, 0.4)
bottom.bar(x + 0.2, D, 0.4)
bottom.set_ylabel(r"damping [N$\cdot$s/m]")
bottom.set_xticks(x, names);
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("k_err", 100 * float(np.abs(K / helyx.SIM_STIFFNESS - 1).max()), display=False)
glue("d_err", 100 * float(np.abs(D / helyx.SIM_DAMPING - 1).max()), display=False)
t = log.arrays()["t"].ravel()
glue("duration", float(t[-1] - t[0]), display=False)
```

From {glue:text}`duration:.0f` s of steps, the fit recovers every stiffness within
{glue:text}`k_err:.1f` % and every damping within {glue:text}`d_err:.1f` %. On the real arm the
same call fits the logged experiment; the pulls must keep every tendon taut, since a slack
tendon transmits nothing and the model would no longer hold.

## Reach around an obstacle

A tanh spring pulls the tip towards a target with at most 2 N, and a damper slows the tip
down. An obstacle of radius 2 cm sits next to the arm's way, and the arm must keep 2 cm clear
of it. Two Gaussian fields, attached at $s = 0.6$ and at the tip, the ends of the part that
has to get past it, push the arm away from the obstacle's centre.

```{code-cell} python
target = np.array([0.20, 0.0, 0.66])  # [m]
obstacle, radius = np.array([0.155, 0.0, 0.50]), 0.02  # [m]
tip = arm.point(s=1.0)

def reach(avoid):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.TanhSpring(tip - target, 60.0, 2.0))  # [N/m], [N]
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))  # [N·s/m]
    if avoid:
        for i, s in enumerate((0.6, 1.0)):
            away = arm.point(s=s) - obstacle
            ctrl.add(f"avoid{i}", vmc.GaussianSpring(away, 140.0, 0.06))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    system = vmc.VirtualMechanismSystem(arm, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    clock = vmc.sim.SimClock(dt=1 / 330)
    return vmc.sim.run(vmc.sim.ModelPlant(arm), controller, clock, T=5.0)

log_free, log_avoid = reach(False), reach(True)
```

```{code-cell} python
:tags: [remove-output]
def draw_obstacle(ax, row=None):
    disc = plt.Circle((obstacle[0], obstacle[2]), radius, color="0.4")
    ax.add_patch(disc)

viz.animate(arm, log_avoid, "hanging-arm-avoid.mp4",
            springs=[(1.0, target)], trace=1.0, draw=draw_obstacle,
            invert=True)
```

```{video} hanging-arm-avoid.mp4
:caption: The tip reaches for the target while two repulsive fields keep the arm clear of the obstacle (grey).
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.6, 5.6))
kin = vmc.Kinematics(arm)
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, log_free.arrays()["q"][-1],
               color=viz.PALETTE[5], label="reach")
viz.draw_robot(ax, arm, log_avoid.arrays()["q"][-1],
               label="reach and avoid")
viz.draw_goal(ax, target, label="target")
draw_obstacle(ax)
ax.invert_yaxis()  # the arm hangs: z points down
viz.label_axes(ax)
ax.legend(loc="lower left", fontsize=14);
```

```{code-cell} python
:tags: [remove-cell]
def closest(log):  # the body's closest approach to the obstacle's surface [cm]
    points = [kin.position(q, s) for q in log.arrays()["q"][::10]
              for s in np.linspace(0.0, 1.0, 41)]
    return 100 * (min(np.linalg.norm(p - obstacle) for p in points) - radius)

def miss(log):  # the tip's final distance to the target [cm]
    return 100 * np.linalg.norm(kin.position(log.arrays()["q"][-1], 1.0) - target)

assert closest(log_free) < 2.0 < closest(log_avoid)  # the 2 cm clearance
assert miss(log_avoid) > miss(log_free)
glue("free_gap", float(closest(log_free)), display=False)
glue("avoid_gap", float(closest(log_avoid)), display=False)
glue("free_miss", float(miss(log_free)), display=False)
glue("avoid_miss", float(miss(log_avoid)), display=False)
```

Without the fields the body passes {glue:text}`free_gap:.1f` cm from the obstacle's surface,
closer than the 2 cm it must keep, and the tip stops {glue:text}`free_miss:.1f` cm from the
target. With them the body keeps at least {glue:text}`avoid_gap:.1f` cm away, and the tip stops
{glue:text}`avoid_miss:.1f` cm from the target: the clearance costs a little reach. Where the
fields attach and how strong they are set that trade-off.
