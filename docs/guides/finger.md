---
file_format: mystnb
kernelspec:
  name: python3
---

# Finger: fingertip stiffness and joint limits

The ADAPT finger has three phalanges and two motors: one bends the first joint (MCP), the other
bends the last two together (PIP and DIP move as one). Its configuration is the two motor
angles; the coupling to the three joints is built into the model.

```{code-cell} python
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

viz.use_style(usetex=False, font_size=13)
finger = adapt.finger()
kin = vmc.Kinematics(finger)
q = np.array([0.8, 0.6])                      # motor angles [rad]
print("joint angles:", adapt.COUPLING @ q)    # MCP, PIP, DIP
print("fingertip:", kin.position(q, "tip"))
```

## A stiff fingertip

A spring on the fingertip makes it behave like a point held by a spring of 50 N/m; joint-limit
springs keep every joint in its range; gravity compensation removes the weight of the phalanges.

```{code-cell} python
rest = kin.position(q, "tip")
ctrl = vmc.Mechanism("ctrl")
ctrl.add("tip", vmc.LinearSpring(finger.point("tip") - rest, 50.0))
ctrl.add("tip_damping", vmc.LinearDamper(finger.point("tip"), 0.5))
ctrl.add("limits", adapt.joint_limit_spring(finger))
ctrl.add("gravity", vmc.GravityCompensation(finger))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl)),
                               output=adapt.output_stage())   # friction compensation + clip
```

`output=adapt.output_stage()` adds the finger's friction compensation and its ±0.8 N·m clip, as
used on the real finger. Leave it out in simulation.

## What the fingertip feels

Push the fingertip away from its rest point and read the motor torques: they hold the tip with
the spring's stiffness. The arrow is the force the spring applies at the tip.

```{code-cell} python
pushed = q + np.array([0.15, -0.1])
law_torque = controller.step(0.0, vmc.Signals(0.0, motor_position=pushed, motor_velocity=[0, 0]))["law_torque"]
tip = kin.position(pushed, "tip")
force = 50.0 * (rest - tip)

fig, ax = plt.subplots(figsize=(6, 5))
viz.draw_robot(ax, finger, q, plane="yz", color="0.8", label="rest")
viz.draw_robot(ax, finger, pushed, plane="yz", label="pushed")
viz.draw_force(ax, tip, force, plane="yz", scale=0.02)
ax.invert_yaxis()      # the finger's z axis points down
viz.label_axes(ax, "yz")
ax.legend(loc="upper left")
print("motor torques [N·m]:", law_torque.round(4))
```

## Stiffness in joint space instead

For joint-space behaviour, put springs on the joint angles directly:

```python
joints = adapt.joint_angles(finger)  # MCP, PIP, DIP [rad]
ctrl.add("posture", vmc.LinearSpring(joints - [0.5, 0.4, 0.4], [0.05, 0.03, 0.03]))
```
