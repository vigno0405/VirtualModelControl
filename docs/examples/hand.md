---
file_format: mystnb
kernelspec:
  name: python3
---

# Hand: grasp, and the hand on a UR5

In this example we simulate the ADAPT hand grasping a ball, then mount the hand on a UR5 arm.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The robot

The ADAPT hand has a thumb and four fingers, driven through tendons by 13 motors: four for the
thumb's joints, one that spreads the fingers, and two per finger, for its first joint (MCP) and
for its last two joints together (PIP and DIP). $q$ holds the motor angles [rad], in this order:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

adapt.HAND_MOTORS
```

```{code-cell} python
:tags: [remove-input]
from schematics import hand as schematic
schematic.figure();
```

The hand seen from the front at $q = 0$, with $z$ up: beside each joint, the index of the motor
that turns it. The spread motor turns the index, ring and pinky fingers (dashed); the middle
finger stays in place. Every number of the template is a `Param`, for example:

```{code-cell} python
:tags: [remove-input]
from myst_nb import glue
from schematics import params
glue("friction", adapt.HAND_FRICTION[0], display=False)
glue("fade", adapt.HAND_FRICTION[1], display=False)
glue("limit", adapt.HAND_TORQUE_LIMIT, display=False)
params.table(adapt.hand(), {
    "coupling": "joint angles per motor angle",
    "index.j2.axis": "axis of the index MCP joint at $q = 0$ (`j1` is the spread)",
    "m_index_proximal.mass": "mass of the index's first phalanx",
    "gravity": "gravity in the hand's base frame, hand upright",
})
```

`adapt.hand` takes the module's tables as keyword arguments and replaces only the entries
given, here the thumb's fingertip in its last joint frame:

```{code-cell} python
custom = adapt.hand(tip_offsets={"thumb": (-0.025, 0.0, 0.0)})  # [m]
for robot in (adapt.hand(), custom):
    print(vmc.Kinematics(robot).position(np.zeros(13), "thumb/tip"))
```

On the real hand, `adapt.hand_output_stage()` adds a friction feed-forward of
{glue:text}`friction:.1f` N·m that fades out above about {glue:text}`fade:.2f` rad/s, then
clips each command to ±{glue:text}`limit:.1f` N·m. The simulation below leaves it out.

## Grasp a ball

A ball sits in front of the palm, in the robot mechanism: the simulator feels it, while the
controller, compiled from its own components only, does not know it is there (see
[contact](../tutorials/contact.md)). Springs pull three fingertips to the ball's centre. The
motor damping and the ball's stiffness are illustrative.

```{code-cell} python
centre, radius = np.array([-0.02, 0.115, 0.06]), 0.03  # [m]
k_ball = 5000.0  # [N/m], stiffness of the ball's surface
digits = ("thumb", "index", "middle")

hand = adapt.add_dynamics(adapt.hand(), damping=0.01)  # [N·m·s/rad]
ctrl = vmc.Mechanism("ctrl")
for d in digits:
    tip = hand.point(f"{d}/tip")
    ball = vmc.SphereDistance(tip, center=centre, radius=radius)
    hand.add(f"ball_{d}", vmc.ContactSpring(ball, k_ball))
    ctrl.add(f"grasp_{d}", vmc.LinearSpring(tip - centre, 40.0))  # [N/m]
    ctrl.add(f"damp_{d}", vmc.LinearDamper(tip, 0.2))  # [N·s/m]
ctrl.add("limits", adapt.hand_joint_limit_spring(hand))
ctrl.add("gravity", vmc.GravityCompensation(hand))
system = vmc.VirtualMechanismSystem(hand, ctrl)
controller = vmc.VMCController(vmc.compile(system))
plant = vmc.sim.ModelPlant(hand)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=2.5)
```

```{code-cell} python
kin = vmc.Kinematics(hand)
rows = log.arrays()
fig, ax = plt.subplots()
for d in digits:
    tips = np.array([kin.position(q, f"{d}/tip") for q in rows["q"]])
    gap = np.linalg.norm(tips - centre, axis=1) - radius  # [m]
    ax.plot(rows["t"], k_ball * np.maximum(-gap, 0.0), label=d)
ax.set_xlabel("time [s]")
ax.set_ylabel("contact force [N]")
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
t, force = rows["t"].ravel(), [line.get_ydata() for line in ax.get_lines()]  # [s], [N]
glue("fingers", 1000 * float(max(t[np.argmax(f > 0)] for f in force[1:])), display=False)
glue("thumb", float(t[np.argmax(force[0] > 0)]), display=False)
glue("force", float(np.mean([f[-1] for f in force])), display=False)
glue("pull", float(ctrl.params["grasp_index.stiffness"].value) * radius, display=False)
```

The ball pushes a fingertip back once the tip enters it. The index and middle fingertips touch
it within {glue:text}`fingers:.0f` ms, the thumb after {glue:text}`thumb:.1f` s. After a short
peak at impact, each presses with {glue:text}`force:.2f` N: the pull of its spring at the
surface, stiffness times radius ({glue:text}`pull:.1f` N), a little less as the ball gives.

```{code-cell} python
:tags: [remove-output]
def draw_ball(ax, row):
    disc = plt.Circle(centre[1:], radius, color=viz.PALETTE[1], alpha=0.3)
    ax.add_patch(disc)

viz.animate(hand, log, "hand-grasp.mp4", plane="yz", draw=draw_ball,
            springs=[(f"{d}/tip", centre) for d in digits])
```

```{video} hand-grasp.mp4
:caption: The hand seen from the side: springs (green) pull three fingertips to the centre of the ball (red), whose surface stops them.
```

## The hand on a UR5

On the arm, the hand sits on the flange of a UR5, whose six revolute joints are built from its
Denavit-Hartenberg table (`ur5.arm(d=..., a=..., alpha=...)` takes another one):

```{code-cell} python
:tags: [remove-input]
from schematics import ur5 as arm_schematic
arm_schematic.figure()
plt.show()
arm_schematic.dh_table()
```

`ur5.with_hand` builds both as one robot: $q$ holds the six arm joints, then the hand's motors.
The hand sits at `mounting_position` [m] on the flange, turned by `mounting_angle` [rad] about
its $z$ axis. Gravity stays in the arm's base frame, so the hand's weight follows the arm's pose.
The UR5 is position-controlled and has no masses in the template, so there is nothing to
simulate: its joint angles are only measured, and only the hand's torques are sent.

```{code-cell} python
from virtualmodelcontrol.robots import ur5

robot = ur5.with_hand()  # the hand centred on the flange, not turned
ctrl = vmc.Mechanism("ctrl")
ctrl.add("gravity", vmc.GravityCompensation(robot))
system = vmc.VirtualMechanismSystem(robot, ctrl)
controller = vmc.VMCController(vmc.compile(system))

arm = [np.pi, -1.2, 1.4, -np.pi / 2 - 0.2, -np.pi / 2, -np.pi / 2]  # [rad]
x = np.concatenate([arm, np.zeros(13)])
meas = vmc.Signals(0.0, motor_position=x, motor_velocity=np.zeros(19))
u = controller.step(0.0, meas)["motor_torque"]
u[6:].round(4)  # the hand's 13 motor torques [N·m], sent to the hand
```

The figure draws this pose:

```{code-cell} python
fig, ax = plt.subplots(figsize=(6.4, 5.0))
viz.draw_robot(ax, robot, x)
lines = viz.skeleton(robot, x)  # the arm, then the five digits
for digit in lines[1:]:  # the palm: from the flange to each digit
    ax.plot(*viz.project([lines[0][-1], digit[0]]), color="0.6", lw=3)
viz.label_axes(ax);
```

The flange faces down, so the arm holds the hand upside down.
