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
    "dip_cog.position": "center of gravity of the distal phalanx (also `mcp_cog`, `pip_cog`)",
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

## Press with a chosen force

The lab tracks a fingertip force on a load cell with two laws that both descend the force error
at a fixed learning rate: at every control step a Param moves by the learning rate times its
gradient ([the force tracking tutorial](../tutorials/force.md) explains the law). One law changes
the joint-space stiffness, which changes the slope of the force against the displacement. The
other moves the joint reference and leaves the stiffness alone. Here the load cell is a table,
{glue:text}`table_cm:.1f` cm below the finger's base, that the fingertip touches at joint angles
of 30°. A joint-space spring holds the finger 12° beyond that, which presses on it.

```{code-cell} python
from virtualmodelcontrol.adaptation import ForceTracking

hz = adapt.FINGER_CONTROL_RATE  # [Hz]
rig = adapt.add_dynamics(adapt.finger(), damping=0.001)
touch = np.radians([30.0, 30.0, 30.0])  # joint angles at first contact
q_touch = np.linalg.lstsq(adapt.COUPLING, touch, rcond=None)[0]
table_z = vmc.Kinematics(rig).position(q_touch, "tip")[2]  # [m]
k_table = 1e4  # [N/m]
gap = vmc.PlaneDistance(rig.point("tip"), normal=[0, 0, -1],
                        origin=[0, 0, table_z])
rig.add("table", vmc.ContactSpring(gap, k_table))
rig.add("cushion", vmc.ContactDamper(gap, 5.0))  # [N·s/m]

joints = adapt.joint_angles(rig)
target = vmc.Ref("theta_ref", 3, value=np.radians([42.0] * 3), unit="rad")
ctrl = vmc.Mechanism("ctrl")
K = 0.2 * np.eye(3)  # [N·m/rad]
ctrl.add("hold", vmc.LinearSpring(joints - target, K))
ctrl.add("damp", vmc.LinearDamper(joints, 0.003))  # [N·m·s/rad]
ctrl.add("gravity", vmc.GravityCompensation(rig))
compiled = vmc.compile(vmc.VirtualMechanismSystem(rig, ctrl))
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("table_cm", float(100 * table_z), display=False)
```

The wanted force steps from 0.5 N to 0.8 N after 10 s. The loop reads the table's force, as the
load cell does, and starts the law after a second. The learning rates, 1e-4 for the stiffness
and 5e-5 for the reference, are the lab's for this finger at 900 Hz:

```{code-cell} python
levels = (0.5, 0.8)  # [N]

def track(adapted, learning_rate, hold=10.0):
    controller = vmc.VMCController(compiled)
    law = ForceTracking(controller, "tip", adapted, normal=[0, 0, 1],
                        rate=learning_rate)
    plant = vmc.sim.ModelPlant(rig, q0=0.8 * q_touch, max_step=2e-4)
    kin, log = vmc.Kinematics(rig), []
    for _ in range(int(len(levels) * hold * hz)):
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - table_z  # [m]
        force = np.array([0.0, 0.0, k_table * max(0.0, below)])
        wanted = levels[int(plant.t // hold)]
        if plant.t > 1.0:
            law.step(controller, force, [0.0, 0.0, wanted])
        log.append((plant.t, force[2], wanted))
        plant.advance(1 / hz)
    return np.array(log), controller.live_params()

by_stiffness, stiffness = track("ctrl.hold.stiffness", 1e-4)
by_reference, reference = track("ctrl.hold.theta_ref", 5e-5)
```

```{code-cell} python
:tags: [remove-input]
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(by_stiffness[:, 0], by_stiffness[:, 1], label="stiffness law")
ax.plot(by_reference[:, 0], by_reference[:, 1], "--", label="reference law")
ax.step(by_stiffness[:, 0], by_stiffness[:, 2], color="0.5", linestyle=":",
        label="wanted")
ax.set_ylim(0.0, 1.5)  # leaves out the finger's landing on the table
ax.set_xlabel("time [s]")
ax.set_ylabel("fingertip force [N]")
ax.legend(loc="upper right");
```

```{code-cell} python
:tags: [remove-cell]
n = len(by_stiffness) // 2
for run in (by_stiffness, by_reference):
    for part, level in ((slice(n // 2, n), 0.5), (slice(3 * n // 2, None), 0.8)):
        assert abs(run[part, 1].mean() - level) < 0.03, (level, run[part, 1].mean())
before = by_stiffness[(by_stiffness[:, 0] > 0.8) & (by_stiffness[:, 0] < 1.0), 1]
glue("preload", float(before.mean()), display=False)
glue("k_start", float(K[0, 0]), display=False)
for i, value in enumerate(np.diag(stiffness["ctrl.hold.stiffness"])):
    glue(f"k_end{i}", float(value), display=False)
for i, value in enumerate(np.degrees(reference["ctrl.hold.theta_ref"])):
    glue(f"ref_end{i}", float(value), display=False)
```

The finger lands on the table (the spike at the start, cut off in the plot) and holds
{glue:text}`preload:.2f` N until the laws start. Both laws then reach each level, and they get
there differently. The stiffness law ends with the diagonal of its stiffness at
{glue:text}`k_end0:.3f`, {glue:text}`k_end1:.3f` and {glue:text}`k_end2:.3f` N·m/rad, from
{glue:text}`k_start:.1f` each, and leaves the reference at 42°. The reference law ends with the
joint targets at {glue:text}`ref_end0:.1f`°, {glue:text}`ref_end1:.1f`° and
{glue:text}`ref_end2:.1f`°, and leaves the stiffness alone. A different finger, sensor or rate
needs learning rates of its own.

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
