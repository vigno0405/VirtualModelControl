---
file_format: mystnb
kernelspec:
  name: python3
---

# Two arms: squeeze an object

In this example we squeeze an object between the tips of two soft arms, with one virtual spring
that couples the arms. Then we hold it with a force we choose, each arm adapting its own spring.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

`bimanual.arms` builds two Helyx arms, the soft arm of the [soft-arm example](soft-arm.md),
standing side by side on one frame and pointing up. As one robot, its configuration stacks the
right arm's $\Delta$ and then the left arm's, and its motors follow the same order.

```{code-cell} python
:tags: [remove-input]
from schematics import bimanual as schematic
schematic.figure();
```

The frame's origin lies midway between the bases, with $z$ up. Both arms keep its orientation,
so the tendons of each sit as in the [soft-arm example](soft-arm.md), and the left arm's motors
are the last nine (indices 9 to 17). These are the Params of the default pair, read from the template; the rest of
the page uses it:

```{code-cell} python
:tags: [remove-input]
import numpy as np
from myst_nb import glue
from schematics import params
from virtualmodelcontrol.robots import bimanual

for name in ("EFFICIENCY", "ENCODER_SIGN", "CONTROL_RATE", "TORQUE_OFFSET",
             "PRETENSION_WEIGHTS", "TORQUE_LIMIT"):
    glue(name, float(getattr(bimanual, name)), display=False)
glue("threshold", float(np.degrees(bimanual.PRETENSION_THRESHOLD)), display=False)
params.table(bimanual.add_dynamics(bimanual.arms()), {
    "right.seg1.L0": "rest length of segment 1 (also `seg2`, `seg3`, and the left arm's)",
    "right.mount.position": "base of the right arm in the frame (also `left.mount.position`)",
    "right.efficiency.c1": "linear coefficient of the delivered over commanded motor torque "
    "(also `left.efficiency.c1`)",
    "gravity": "gravity in the frame: the arms point up",
    "right_m1.mass": "mass of the right arm's segment 1, lumped at its middle",
    "right_stiffness.stiffness": "stiffness of the right arm in $\\Delta$, one value per axis",
    "left_stiffness.stiffness": "stiffness of the left arm in $\\Delta$",
    "right_damping.damping": "damping of the right arm in $\\Delta$, one value per axis",
    "left_damping.damping": "damping of the left arm in $\\Delta$",
})
```

The stiffness and damping come from an identification on the real arms. Only the simulator
uses them. The fit used the commanded motor torques, so the simulated arms take each torque as
the controller sends it: their efficiency is 1, the default of every template. The efficiency
of the tendons, measured against a load cell, is `bimanual.EFFICIENCY`
({glue:text}`EFFICIENCY:.2f`). It matters for the forces the real arms exert on their
surroundings. The [efficiency page](../concepts/efficiency.md) explains when to use it.

The arms' encoders follow the library's sign convention (`bimanual.ENCODER_SIGN` is
{glue:text}`ENCODER_SIGN:+.0f`), and their controller runs at {glue:text}`CONTROL_RATE:.0f` Hz.
On the real arms, `bimanual.output_stage()` adds {glue:text}`TORQUE_OFFSET:.2f` N·m to every
command, a soft stop of {glue:text}`PRETENSION_WEIGHTS:.2f` N·m/rad on motors that release
their tendon past {glue:text}`threshold:.0f`°, and a clip at ±{glue:text}`TORQUE_LIMIT:.1f` N·m.
The simulation below leaves it out.

The template takes its geometry as arguments, one value for both arms or a dict by arm. Here
the segments are 30, 15 and 15 cm long (the default is 29, 14.5 and 14.5 cm) and the bases 30 cm
apart (the default is 25 cm). This pair is only a check of the geometry arguments:

```{code-cell} python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import bimanual

pair = bimanual.arms(lengths=(0.3, 0.15, 0.15),  # [m], both arms
                     base_positions={"right": (0.15, 0.0, 0.0),
                                     "left": (-0.15, 0.0, 0.0)})
kin = vmc.Kinematics(pair)
[kin.position(np.zeros(18), (arm, 1.0)) for arm in bimanual.ARMS]  # tips
```

## Squeeze an object

An object 10 cm wide sits between the tips. In the simulation it belongs to the robot
mechanism, as a [contact](../tutorials/contact.md): a one-sided spring on the distance between
the tips that pushes them apart once they come closer than its width. The controller does not
know about it.

```{code-cell} python
:tags: [remove-output]
arms = bimanual.add_dynamics(bimanual.arms())
right, left = arms.point("right", s=1.0), arms.point("left", s=1.0)
distance = vmc.Norm(right - left)  # between the tips [m]
width, k_object = 0.10, 200.0  # [m], [N/m]
arms.add("object", vmc.ContactSpring(distance - width, k_object))
```

The controller pulls the tips together with a spring of zero rest length between them, slows
their closing with a damper in parallel, and cancels the weight of the segments.

```{code-cell} python
K = 20.0  # [N/m]
ctrl = vmc.Mechanism("ctrl")
ctrl.add("squeeze", vmc.LinearSpring(distance, K))
ctrl.add("damp", vmc.LinearDamper(distance, 1.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arms))

system = vmc.VirtualMechanismSystem(arms, ctrl)
controller = vmc.VMCController(vmc.compile(system))
plant = vmc.sim.ModelPlant(arms)
clock = vmc.sim.SimClock(dt=1 / bimanual.CONTROL_RATE)
log = vmc.sim.run(plant, controller, clock, T=2.0)
```

We follow the distance between the tips, the force of the spring and the force on the
object:

```{code-cell} python
import matplotlib.pyplot as plt
from virtualmodelcontrol import viz

kin = vmc.Kinematics(arms)
rows = log.arrays()

def tips(q):
    return kin.position(q, ("right", 1.0)), kin.position(q, ("left", 1.0))

d = np.array([np.linalg.norm(a - b) for a, b in map(tips, rows["q"])])
on_object = k_object * np.maximum(width - d, 0.0)  # [N]

fig, (top, bottom) = plt.subplots(2, 1, figsize=(4.8, 6.0), sharex=True)
top.plot(rows["t"], 100 * d)
top.axhline(100 * width, color=viz.PALETTE[1], ls="--", label="object")
top.set_ylabel("tip distance [cm]")
top.legend()
bottom.plot(rows["t"], K * d, label="spring")
bottom.plot(rows["t"], on_object, label="on the object")
bottom.set_xlabel("time [s]")
bottom.set_ylabel("force [N]")
bottom.legend();
```

```{code-cell} python
:tags: [remove-cell]
t = rows["t"].ravel()
assert (d <= width).any(), "the tips never reach the object"
glue("touch", float(t[np.argmax(d <= width)]), display=False)
glue("spring", float(K * d[-1]), display=False)
glue("squeeze", float(on_object[-1]), display=False)
```

The tips meet the object after {glue:text}`touch:.2f` s and stop on it. The spring's torques
then pull with {glue:text}`spring:.2f` N and the object feels {glue:text}`squeeze:.2f` N: the
rest holds the soft arms bent. A stiffer spring squeezes harder.

## Hold it with a chosen force

The spring above squeezes with whatever force its stiffness leaves. To hold the object with a
chosen force, we give each arm a spring of its own, towards a goal on a face of the object, and
let a [force tracking law](../tutorials/force.md) move each arm's goal. Each arm pushes along
the line between the tips, the $x$ axis: the right arm towards $-x$, the left one towards $+x$.
The goals start on the object's faces, 5 cm from the middle, at the height of the tips at rest.

```{code-cell} python
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.estimation import ContactForce

push = {"right": np.array([-1.0, 0.0, 0.0]),  # each arm pushes
        "left": np.array([1.0, 0.0, 0.0])}  # towards the other
height = kin.position(np.zeros(18), ("right", 1.0))[2]  # the tips at rest
grip = vmc.Mechanism("grip")
for arm, x in (("right", 0.05), ("left", -0.05)):  # on the object's faces
    tip = arms.point(arm, s=1.0)
    goal = vmc.Ref(f"goal_{arm}", 3, value=[x, 0.0, height])
    grip.add(arm, vmc.LinearSpring(tip - goal, 30.0))  # [N/m]
    grip.add(f"damp_{arm}", vmc.LinearDamper(tip, 2.0))
grip.add("gravity", vmc.GravityCompensation(arms))
grip_system = vmc.VirtualMechanismSystem(arms, grip)
compiled = vmc.compile(grip_system)
```

Each arm needs to know the force it exerts. With a sensor on the object, both read the object's
force. Without one, each arm estimates its own force from the model of the arms alone, a copy
that has no object. The laws start at 1.5 s, once the arms have settled on the faces. `hold`
returns the time, the object's force, what the right arm believes it pushes with, and the
level of the tank, if there is one:

```{code-cell} python
wanted = 1.0  # [N]

def hold(sensor, level=None, step=0.01):
    controller = vmc.VMCController(compiled)
    if level is not None:  # run through a tank of this many joules
        controller = vmc.control.Tank(controller, level=level)
    model = bimanual.add_dynamics(bimanual.arms())  # no object
    law, estimate = {}, {}
    for arm in bimanual.ARMS:
        site = (arm, 1.0)
        law[arm] = ForceTracking(controller, site, f"grip.{arm}.goal_{arm}",
                                 push[arm], max_force_step=step)  # [N]
        estimate[arm] = ContactForce(controller, site, push[arm],
                                     robot=model)
    plant = vmc.sim.ModelPlant(arms)
    dt, log = 1 / bimanual.CONTROL_RATE, []
    for _ in range(int(5.0 / dt)):
        plant.write(controller.step(plant.t, plant.read()))
        a, b = (kin.position(plant.q, (arm, 1.0)) for arm in bimanual.ARMS)
        true = k_object * max(0.0, width - np.linalg.norm(a - b))  # [N]
        for arm in bimanual.ARMS:
            told = true * push[arm] if sensor else estimate[arm](controller)
            if plant.t > 1.5:
                law[arm].step(controller, told, wanted * push[arm])
        believed = estimate["right"](controller) @ push["right"]
        tank = getattr(controller, "level", 0.0)  # [J]
        log.append((plant.t, true, believed, tank))
        plant.advance(dt)
    return np.array(log)

with_sensor, without = hold(sensor=True), hold(sensor=False)
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(4.8, 6.0))
for ax in (top, bottom):
    ax.plot(without[:, 0], without[:, 1], label="estimates only")
    ax.plot(with_sensor[:, 0], with_sensor[:, 1], "--", label="sensor on the object")
    ax.axhline(wanted, color="0.5", linestyle=":", label="wanted")
    ax.set_ylabel("force on the object [N]")
top.legend(loc="lower right")
bottom.set_xlim(4.0, 5.0)  # the last second
bottom.set_ylim(0.9, 1.1)
bottom.set_xlabel("time [s]");
```

```{code-cell} python
:tags: [remove-cell]
late = slice(-500, None)
blind, sensed = without[late, 1], with_sensor[late, 1]
assert abs(blind.mean() - wanted) < 0.02 and abs(sensed.mean() - wanted) < 0.02
half = hold(sensor=True, step=0.005)[late, 1]
assert 0.35 < np.ptp(half) / np.ptp(sensed) < 0.65  # the ripple follows the step
glue("grip_blind", float(blind.mean()), display=False)
glue("grip_sensed", float(sensed.mean()), display=False)
glue("grip_ripple_blind", float(np.ptp(blind)), display=False)
glue("grip_ripple_sensed", float(np.ptp(sensed)), display=False)
glue("grip_ripple_half", float(np.ptp(half)), display=False)
glue("grip_believed", float(without[late, 2].mean()), display=False)
```

The object feels {glue:text}`grip_blind:.2f` N with the estimates alone and
{glue:text}`grip_sensed:.2f` N with the sensor. Peak to peak, the force ripples by
{glue:text}`grip_ripple_blind:.3f` N in the first run and by {glue:text}`grip_ripple_sensed:.3f` N
in the second. With the sensor, the ripple follows `max_force_step` (0.01 N by default): half
that step gives {glue:text}`grip_ripple_half:.3f` N. The right arm's estimate,
{glue:text}`grip_believed:.2f` N, is right too, because the model has no error here. A wrong
mass in the model would give a wrong force, as in the
[force tracking tutorial](../tutorials/force.md).

Moving a goal inward stores energy in the spring, and a [tank](../tutorials/energy.md) pays for
it. The laws take the tank in place of the controller. We run it with 0.2 J and with an empty
tank:

```{code-cell} python
funded, empty = hold(False, level=0.2), hold(False, level=0.0)
```

```{code-cell} python
:tags: [remove-cell]
start = int(1.5 * bimanual.CONTROL_RATE)  # when the laws start
assert abs(funded[late, 1].mean() - wanted) < 0.02 and empty[late, 1].mean() < 0.5 * wanted
glue("tank_funded", float(funded[late, 1].mean()), display=False)
glue("tank_empty", float(empty[late, 1].mean()), display=False)
glue("tank_given", float(empty[start, 3]), display=False)
glue("tank_left", float(funded[-1, 3]), display=False)
```

With 0.2 J the object feels {glue:text}`tank_funded:.2f` N at the end, and {glue:text}`tank_left:.2f` J
are left. The empty tank had {glue:text}`tank_given:.3f` J when the laws started, what the arms'
dampers took on their way to the object, and the object feels only
{glue:text}`tank_empty:.2f` N.

## Animate

`viz.animate` draws only springs from the robot to fixed points. The `draw` callback adds the
object and the spring between the tips. It receives the log's row at each frame.

```{code-cell} python
:tags: [remove-output]
def draw(ax, row):
    a, b = tips(row["q"])
    centre = (a + b) / 2
    ax.add_patch(plt.Circle((centre[0], centre[2]), width / 2,
                            color=viz.PALETTE[1], alpha=0.3))
    viz.draw_spring(ax, b, a)

viz.animate(arms, log, "two-arms.mp4", draw=draw)
```

```{video} two-arms.mp4
:caption: The spring between the tips (teal) closes the arms on the object (red) and squeezes it.
```
