---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft arm: reach past an obstacle, limit the force, shape the body

In this example we simulate the Helyx soft arm mounted on its side, in three experiments. In
the first, the tip reaches for a target while two repulsive fields keep the body off an
obstacle. In the second, a force-limited spring pulls the tip against a string. In the third, a
constrained spring holds the tip on a line while another spring shapes the body.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The arm has three continuum segments of 145 mm. Three tendons run along each segment, and each
tendon is wound on its own motor, so the arm has nine motors. The template describes each
segment by piecewise constant curvature: its configuration is $\Delta = (D_x, D_y, D_l)$ [m],
the tendon-length differences that bend it and its elongation.

```{code-cell} python
:tags: [remove-input]
from schematics import helyx as schematic
schematic.figure("145-145-145");
```

The arm is mounted on its side: gravity acts along $-y$, out of the drawing, so the arm bends
in a horizontal plane. The arc parameter $s$ runs from 0 at the base to 1 at the tip,
uniformly in rest length, and `arm.point(s=...)` gives any point of the body.

The cross-sections below, seen along each segment's $z$ axis, show where the tendons sit. The
number beside a tendon is the index of its motor in the motor vector. A positive motor angle
pulls its tendon.

```{code-cell} python
:tags: [remove-input]
from virtualmodelcontrol.robots import helyx
schematic.sections(helyx.arm("145-145-145").params);
```

Every number of the template is a `Param`. These are the arm's, read from the template:

```{code-cell} python
:tags: [remove-input]
from myst_nb import glue
from schematics import params

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
glue("eta", 100 * helyx.EFFICIENCY, display=False)
glue("pretension", helyx.PRETENSION_WEIGHTS, display=False)
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

The stiffness and damping come from an identification on a real Helyx arm. Only the simulator
uses them. The fit used the commanded motor torques, so the simulated arm takes each torque as
the controller sends it: its efficiency is 1, the default of every template. The efficiency of
the tendons, measured against a load cell, is `helyx.EFFICIENCY` ({glue:text}`eta:.0f` %). It
matters for the forces the real arm exerts on its surroundings. The
[efficiency page](../concepts/efficiency.md) explains when to use it.

The nine motors and their bus are in `helyx.hardware("145-145-145")`. On the real arm,
`helyx.output_stage()` adds a small pretension to every motor command
({glue:text}`pretension:.3f` N·m per radian of motor angle). The simulations below leave it out.

## Reach past an obstacle

A spring pulls the tip towards a target beyond an obstacle, while two repulsive Gaussian
fields push the body away from it. The fields are attached a third and two thirds of the way
along the arm. Dampers on the tip and on those two points slow them down. Each run lets the
fields ramp up over 5 s, then walks the target from the tip at rest to its place at 10 cm/s,
and holds it for 3 s. We repeat the run for four strengths of the fields.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
kin = vmc.Kinematics(arm)
tip = arm.point(s=1.0)
rest = kin.position(np.zeros(9), 1.0)  # the tip at rest [m]
target = np.array([-0.50, 0.0, 0.38])  # [m]
obstacle = np.array([-0.15, 0.0, 0.2175])  # [m]

goal = vmc.Ref("goal", value=rest)  # moved while running
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - goal, 30.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(tip, 1.0))  # [N·s/m]
for name, s in (("near", 1 / 3), ("far", 2 / 3)):
    point = arm.point(s=s)
    ctrl.add(name, vmc.GaussianSpring(point - obstacle, 0.0, 0.06))
    ctrl.add(f"damp_{name}", vmc.LinearDamper(point, 1.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
```

The loop is the one `vmc.sim.run` runs, with one `set` before each step for the fields'
strength and the target:

```{code-cell} python
def reach(strength, dt=1 / 330):
    controller = vmc.VMCController(law)
    plant = vmc.sim.ModelPlant(arm)
    controller.reset(plant.t, plant.read())
    walk = np.linalg.norm(target - rest) / 0.10  # [s], at 10 cm/s
    for _ in range(round((5.0 + walk + 3.0) / dt)):
        ramp = min(plant.t / 5.0, 1.0)
        along = min(max(plant.t - 5.0, 0.0) / walk, 1.0)
        controller.set({
            "ctrl.near.strength": ramp * strength,
            "ctrl.far.strength": ramp * strength,
            "ctrl.reach.goal": rest + along * (target - rest),
        })
        meas = plant.read()
        plant.write(controller.step(plant.t, meas))
        plant.advance(dt)
    return plant.read()["q"]

strengths = (0.0, 200.0, 600.0, 1000.0)  # [N/m]
final = {k: reach(k) for k in strengths}
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(5.6, 5.6))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
for (k, q), color in zip(final.items(), viz.PALETTE):
    viz.draw_robot(ax, arm, q, color=color, label=f"{k:.0f} N/m")
ax.plot(obstacle[0], obstacle[2], "o", color="0.3", ms=10, label="obstacle")
viz.draw_goal(ax, target, label="target")
viz.label_axes(ax)
ax.legend(loc="lower left", fontsize=14);
```

```{code-cell} python
:tags: [remove-cell]
def clearance(q):  # the body's closest point to the obstacle [cm]
    body = [kin.position(q, s) for s in np.linspace(0.0, 1.0, 61)]
    return 100 * min(np.linalg.norm(p - obstacle) for p in body)

def miss(q):  # the tip's distance to the target [cm]
    return 100 * np.linalg.norm(kin.position(q, 1.0) - target)

gaps = [clearance(q) for q in final.values()]
misses = [miss(q) for q in final.values()]
assert np.all(np.diff(gaps) > 0) and np.all(np.diff(misses) > 0)
for k, (gap, m) in enumerate(zip(gaps, misses)):
    glue(f"gap{k}", float(gap), display=False)
    glue(f"miss{k}", float(m), display=False)
```

Turning the fields up buys clearance and costs reach. The body's closest point to the obstacle
is {glue:text}`gap0:.1f` cm from it without the fields, then {glue:text}`gap1:.1f`,
{glue:text}`gap2:.1f` and {glue:text}`gap3:.1f` cm as the strength grows to 1000 N/m. The tip
ends {glue:text}`miss0:.1f`, {glue:text}`miss1:.1f`, {glue:text}`miss2:.1f` and
{glue:text}`miss3:.1f` cm from the target. The target lies beyond the arm's reach, so even
without the fields the tip stops short of it.

## Limit the force

A string ties the tip back to where it rests, through a load cell that measures its pull. In
the simulation the string is a stiff constrained spring that acts along $x$ only, and we add it
to the robot mechanism. A spring at the tip pulls towards a goal that moves 5 cm further along
$-x$ every 3 s, up to 50 cm. We try two springs of the same stiffness, 10 N/m. A linear spring
pulls ten times its stretch, ever harder. A tanh spring pulls the same at first but never more
than its maximum force, 1.33 N.

```{code-cell} python
tied = helyx.add_dynamics(helyx.arm("145-145-145"))
k_string = 2000.0  # [N/m]
string = vmc.ConstrainedLinearSpring(tied.point(s=1.0) - rest, k_string,
                                     normal=[1.0, 0.0, 0.0])  # along x
tied.add("string", string)
steps = -np.arange(0.05, 0.51, 0.05)  # [m], the goal's moves along x

def pull(spring, dt=1 / 330):
    goal = vmc.Ref("goal", value=rest)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", spring(tied.point(s=1.0) - goal))
    ctrl.add("damp", vmc.LinearDamper(tied.point(s=1.0), 1.5))
    ctrl.add("gravity", vmc.GravityCompensation(tied))
    system = vmc.VirtualMechanismSystem(tied, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(tied)
    controller.reset(plant.t, plant.read())
    force = []  # the string's pull at the end of each step [N]
    for x in steps:
        controller.set({"ctrl.pull.goal": rest + [x, 0.0, 0.0]})
        for _ in range(round(3.0 / dt)):
            meas = plant.read()
            plant.write(controller.step(plant.t, meas))
            plant.advance(dt)
        tip_x = vmc.Kinematics(tied).position(plant.read()["q"], 1.0)[0]
        force.append(k_string * abs(tip_x - rest[0]))
    return np.array(force)

linear = pull(lambda y: vmc.LinearSpring(y, 10.0))
tanh = pull(lambda y: vmc.TanhSpring(y, 10.0, 1.33))

fig, ax = plt.subplots()
ax.plot(-100 * steps, linear, "o-", label="linear")
ax.plot(-100 * steps, tanh, "s-", label="tanh")
ax.axhline(1.33, color="0.6", ls=":", label="maximum force")
ax.set_xlabel("goal moved [cm]")
ax.set_ylabel("pull on the string [N]")
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
assert tanh[-1] < 1.33 < linear[-1] and np.all(np.diff(tanh) >= 0)
glue("linear_end", float(linear[-1]), display=False)
glue("tanh_end", float(tanh[-1]), display=False)
```

The linear spring's pull grows with the stretch, to {glue:text}`linear_end:.2f` N when the goal
is 50 cm away. The tanh spring's pull levels off at {glue:text}`tanh_end:.2f` N, just under its
maximum, however far the goal moves. A real transmission passes on only part of each motor
torque, so on the real arm the string feels less than either: see the
[efficiency page](../concepts/efficiency.md).

## Shape the body with a constrained spring

Springs can pull on any point of the arm, not only on its tip, so together they set the shape
of the whole body. Here a linear spring pulls the middle of the arm ($s = 0.5$) to a point, the
cross, and a constrained spring pulls the tip towards the dashed line $z = 0.38$ m. A
constrained spring acts along one direction only, here the line's normal $z$. It stretches with
the tip's distance from the line, so it does not mind where along the line the tip ends.

```{code-cell} python
middle = np.array([-0.05, 0.0, 0.2])  # [m]
on_line = np.array([0.0, 0.0, 0.38])  # [m], a point of the line

ctrl = vmc.Mechanism("ctrl")
ctrl.add("middle", vmc.LinearSpring(arm.point(s=0.5) - middle, 600.0))
ctrl.add("line", vmc.ConstrainedLinearSpring(tip - on_line, 600.0,
                                             normal=[0.0, 0.0, 1.0]))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))
clock = vmc.sim.SimClock(dt=1 / 330)
log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, clock, T=3.0)
q = log.arrays()["q"][-1]

fig, ax = plt.subplots(figsize=(4.4, 5.2))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, q)
viz.draw_goal(ax, middle, label="goal of the middle")
ax.axhline(on_line[2], color=viz.PALETTE[2], ls="--",
           label="line for the tip")
viz.label_axes(ax)
ax.legend(loc="lower left");
```

```{code-cell} python
:tags: [remove-cell]
glue("middle_off", 100 * float(np.linalg.norm(kin.position(q, 0.5) - middle)), display=False)
glue("z_off", 100 * float(kin.position(q, 1.0)[2] - on_line[2]), display=False)
```

The middle of the arm stops {glue:text}`middle_off:.1f` cm from the cross and the tip
{glue:text}`z_off:.1f` cm above the line: the arm's own stiffness holds both a little short of
their goals. Nothing pulls the tip along the line, so its place on the line follows from the
shape the two springs give the body.
