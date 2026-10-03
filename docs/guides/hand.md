---
file_format: mystnb
kernelspec:
  name: python3
---

# Hand: grasp, limits, and the hand on a UR5

The ADAPT hand has five digits and 13 motors: four for the thumb, one that spreads the fingers,
and two per finger (the last two joints of each finger move together). Its configuration is the
13 motor angles in `adapt.HAND_MOTORS` order.

```{code-cell} python
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt, ur5

viz.use_style(usetex=False, font_size=13)
hand = adapt.hand()
kin = vmc.Kinematics(hand)
q = np.zeros(13)
print(adapt.HAND_MOTORS)
print({d: kin.position(q, f"{d}/tip").round(3) for d in adapt.HAND_DIGITS})
```

## A grasp

Each fingertip is pulled towards a point on the surface of an object; joint-limit springs keep
the joints in range; gravity compensation and friction compensation complete the controller.

```{code-cell} python
centre = np.array([0.03, 0.09, 0.07])          # object centre in the hand frame [m]
ctrl = vmc.Mechanism("ctrl")
for digit in ("thumb", "index", "middle"):
    tip = hand.point(f"{digit}/tip")
    ctrl.add(f"grasp_{digit}", vmc.LinearSpring(tip - centre, 20.0))
    ctrl.add(f"damp_{digit}", vmc.LinearDamper(tip, 0.2))
ctrl.add("limits", adapt.hand_joint_limit_spring(hand))
ctrl.add("gravity", vmc.GravityCompensation(hand))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(hand, ctrl)),
                               output=adapt.hand_output_stage())
u = controller.step(0.0, vmc.Signals(0.0, motor_position=q, motor_velocity=np.zeros(13)))
print("motor torques [N·m]:", u["motor_torque"].round(3))
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, plane in zip(axes, ("xz", "yz")):
    viz.draw_robot(ax, hand, q, plane=plane)
    viz.draw_goal(ax, centre, plane=plane)
    viz.label_axes(ax, plane)
axes[0].set_title("front")
axes[1].set_title("side");
```

## The hand on a UR5

When the hand moves with an arm, gravity changes direction in the hand's frame. Two ways to
handle it:

- **The arm and the hand together.** `ur5.with_hand()` mounts the hand on the UR5 flange: the
  configuration is the 6 arm joints (measured) and the 13 hand motors, gravity stays fixed in
  the arm's base frame, and the hand's gravity follows the arm's pose automatically. Goals can be
  given in the world frame. Send only the last 13 torques (the hand's).
- **The hand alone, with live gravity.** Compile the hand with `runtime=["*.gravity"]` and set
  `adapt.hand_gravity(R_flange)` at each step, from the flange orientation the arm reports.

```{code-cell} python
robot = ur5.with_hand()
qa = np.array([0.0, -1.2, 1.4, -1.8, -1.57, 0.0])          # arm joints [rad]
kin = vmc.Kinematics(robot)
x = np.concatenate([qa, np.zeros(13)])
print("index fingertip in the world frame:", kin.position(x, "hand/index/tip").round(3))

fig, ax = plt.subplots(figsize=(6, 6))
viz.draw_robot(ax, robot, x, linewidth=4)
viz.label_axes(ax)
ax.set_title("UR5 carrying the hand");
```
