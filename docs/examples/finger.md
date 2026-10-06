---
file_format: mystnb
kernelspec:
  name: python3
---

# Finger: fingertip stiffness and joint limits

In this example we give the ADAPT finger a stiff fingertip with a virtual spring, and keep its
joints inside their ranges with joint-limit springs.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The ADAPT finger has three phalanges and two motors: motor 0 turns the first joint (MCP)
through a pulley, and motor 1 turns the other two (PIP and DIP) together through a cable. Its
configuration is the two motor angles $\theta_0, \theta_1$.

```{code-cell} python
:tags: [remove-input]
from schematics import finger as schematic
schematic.figure();
```

The drawing shows the finger straight, as mounted: its $z$ axis points down, along gravity, and
a positive joint angle bends the finger down. The wedges are the ranges of the joints. These are
the Params of the finger, read from the template:

```{code-cell} python
:tags: [remove-input]
from myst_nb import glue
from schematics import params
from virtualmodelcontrol.robots import adapt

glue("friction", adapt.FRICTION[0], display=False)
glue("speed", adapt.FRICTION[1], display=False)
glue("limit", adapt.TORQUE_LIMIT, display=False)
glue("eta_mcp", 100 * adapt.MOTOR_EFFICIENCY[0], display=False)
glue("eta_pip", 100 * adapt.MOTOR_EFFICIENCY[1], display=False)
params.table(adapt.finger(), {
    "j1.axis": "axis of the MCP joint (also `j2.axis`, `j3.axis` for PIP, DIP)",
    "j2.point": "PIP joint, at the end of the proximal phalanx",
    "j3.point": "DIP joint, at the end of the middle phalanx",
    "tip.position": "fingertip, at the end of the distal phalanx",
    "coupling": "joint angles (MCP, PIP, DIP) per motor angle",
    "m_mcp.mass": "mass of the proximal phalanx",
    "m_pip.mass": "mass of the middle phalanx",
    "m_dip.mass": "mass of the distal phalanx",
    "dip_cog.position": "centre of gravity of the distal phalanx (also `mcp_cog`, `pip_cog`)",
    "gravity": "gravity in the finger's frame, as mounted",
})
```

On the real finger, `adapt.output_stage()` adds a friction feed-forward of
{glue:text}`friction:.2f` N·m in the direction of each command, which fades out once the motor
turns faster than about {glue:text}`speed:.2f` rad/s, then clips the commands to
±{glue:text}`limit:.1f` N·m. The simulations below leave it out.

The motors pass on only part of their torque: {glue:text}`eta_mcp:.0f` % at the MCP and
{glue:text}`eta_pip:.0f` % at the PIP (`adapt.MOTOR_EFFICIENCY`), fitted to measured fingertip
forces. The template's efficiency is 1, the default of every template, so the simulations below
deliver the full torque. With a different efficiency on each of two motors that one virtual
spring couples, the force the finger receives no longer derives from the spring's energy, and a
simulated finger can start to vibrate. `adapt.finger(efficiency=adapt.MOTOR_EFFICIENCY)`
includes the efficiencies.

The template takes its geometry and transmission as arguments. A longer finger with a larger
pulley on the MCP joint, for example, has a new first coupling ratio:

```{code-cell} python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt

longer = adapt.finger(link_lengths=(0.045, 0.035, 0.02),  # [m]
                      pulley_radius=0.003)  # [m], on the MCP joint
longer.params["coupling"].value.round(3)
```

A new pulley comes with a new ratio, which a sweep measures: set the MCP joint to known angles,
read the motor's, and `fit_transmission` gives the joint angle per motor angle, the first entry
of the coupling. Here is the finger's own sweep (the motor angles that went with joints at 0° to
75° in steps of 15°):

```{code-cell} python
from virtualmodelcontrol.identification import fit_transmission

joint = [0, 15, 30, 45, 60, 75]  # [deg], set
motor = [0, 40.07, 69.96, 99.93, 129.99, 170.07]  # [deg], read
ratio = fit_transmission(motor, joint)
radius = 1e3 * adapt.MOTOR_RADIUS * ratio  # [mm]
f"ratio {ratio:.4f}, pulley radius {radius:.2f} mm"
```

## A stiff fingertip

A spring pulls the fingertip to a goal, a damper on the tip slows it down, and gravity
compensation cancels the weight of the phalanges. `adapt.add_dynamics` gives the simulated
finger its gravity and a viscous damping on the motors, whose value here is only illustrative.

```{code-cell} python
finger = adapt.add_dynamics(adapt.finger(), damping=0.001)  # [N·m·s/rad]
tip = finger.point("tip")

def fingertip(goal, limits=False):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("tip", vmc.LinearSpring(tip - goal, 100.0))  # [N/m]
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))  # [N·s/m]
    if limits:
        ctrl.add("limits", adapt.joint_limit_spring(finger))
    ctrl.add("gravity", vmc.GravityCompensation(finger))
    return ctrl

def simulate(ctrl, q0=None):
    system = vmc.VirtualMechanismSystem(finger, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(finger, q0=q0)
    clock = vmc.sim.SimClock(dt=1 / 500)
    return vmc.sim.run(plant, controller, clock, T=1.0)

q = simulate(fingertip([0.0, 0.04, 0.06])).arrays()["q"][-1]  # goal [m]
kin = vmc.Kinematics(finger)
kin.position(q, "tip").round(4), np.degrees(adapt.COUPLING @ q).round(1)
```

The fingertip settles on the goal, and the joint angles (MCP, PIP, DIP, in degrees) stay inside
their ranges, with PIP and DIP turning together.

## Joint limits

A goal above the finger asks it to bend back at the MCP, out of its range.
`adapt.joint_limit_spring` adds springs on the three joint angles. They do nothing inside the
ranges and push a joint back once it leaves its range. We start from where the finger stopped,
without and with them, and plot the joint angles against their ranges (shaded):

```{code-cell} python
import matplotlib.pyplot as plt
from virtualmodelcontrol import viz

above = [0.0, 0.08, -0.02]  # [m], z points down
runs = {"no limits": simulate(fingertip(above), q0=q),
        "limit springs": simulate(fingertip(above, limits=True), q0=q)}
colors = (viz.PALETTE[0], viz.PALETTE[3])
fig, axes = plt.subplots(2, 1, figsize=(4.8, 6.0), sharex=True)
for (label, log), color in zip(runs.items(), colors):
    rows = log.arrays()
    angles = np.degrees(rows["q"] @ adapt.COUPLING.T)  # MCP, PIP, DIP
    for k, ax in enumerate(axes):
        ax.plot(rows["t"], angles[:, k], color=color, label=label)
for ax, joint in zip(axes, ("MCP", "PIP")):
    ax.axhspan(*np.degrees(adapt.JOINT_LIMITS[joint]), color="0.9")
    ax.set_ylabel(f"{joint} [deg]")
axes[1].set_xlabel("time [s]")
axes[0].legend();
```

The paths of the fingertip, with the finger where each run ends:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.8, 4.8))
for (label, log), color in zip(runs.items(), colors):
    qs = log.arrays()["q"]
    path = np.array([kin.position(x, "tip") for x in qs])
    ax.plot(path[:, 1], path[:, 2], color=color, lw=1.5)
    viz.draw_robot(ax, finger, qs[-1], plane="yz", color=color, label=label)
viz.draw_goal(ax, above, plane="yz")
ax.invert_yaxis()  # as mounted: z points down
viz.label_axes(ax, "yz")
ax.legend(loc="lower left");
```

```{code-cell} python
:tags: [remove-cell]
free, held = (log.arrays()["q"][-1] for log in runs.values())
glue("free", float(np.degrees(adapt.COUPLING @ free)[0]), display=False)
glue("held", float(np.degrees(adapt.COUPLING @ held)[0]), display=False)
glue("gap", 1000 * float(np.linalg.norm(kin.position(held, "tip") - above)), display=False)
```

Without limit springs the MCP bends back to {glue:text}`free:.0f`° and the fingertip reaches
the goal. With them, the joints stop just past the ends of their ranges (MCP at
{glue:text}`held:.0f`°), where the limit springs balance the fingertip spring, and the tip stays
{glue:text}`gap:.0f` mm short of the goal. A stiffer limit spring (its `stiffness` argument)
keeps the joints closer to their ranges.

## Animate

`plane="yz"` shows the finger from the side and `invert=True` turns $z$ down, as mounted.
`speed=0.25` slows the run down four times.

```{code-cell} python
:tags: [remove-output]
viz.animate(finger, runs["limit springs"], "finger.mp4", plane="yz",
            invert=True, springs=[("tip", above)], trace="tip", speed=0.25,
            figsize=(5.0, 5.0))
```

```{video} finger.mp4
:caption: The fingertip spring (teal) pulls the tip towards a goal above the finger; the limit springs stop the joints at the ends of their ranges.
```
