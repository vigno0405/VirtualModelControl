---
file_format: mystnb
kernelspec:
  name: python3
---

# virtualmodelcontrol

**Control robots by attaching virtual springs, dampers and masses to them.**
`virtualmodelcontrol` is a Python library for Virtual Model Control (VMC): you describe your
robot once, place virtual elements where they should act (a spring pulling a fingertip to a
goal, a repulsive field around an obstacle, a damper along an arm), and the library turns them
into motor torques at control rate, in simulation and on the real robot.

```{code-cell} python
:tags: [remove-input]
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx
%config InlineBackend.figure_formats = ['svg']
viz.use_style(usetex=False, font_size=14)

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
goal = np.array([0.25, 0.0, 0.55])
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 600.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
plant = vmc.sim.ModelPlant(arm)
vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=3.0)
tip = vmc.Kinematics(arm).position(plant.q, 1.0)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
viz.draw_robot(ax, arm, np.zeros(9), color="0.8", label="at rest")
viz.draw_robot(ax, arm, plant.q, label="controlled")
viz.draw_spring(ax, tip, goal)
viz.draw_goal(ax, goal, label="goal")
ax.invert_yaxis()
viz.label_axes(ax)
ax.set_title("A hanging soft arm pulled by a virtual spring")
ax.legend(loc="lower left");
```

## What you can do

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/install
:link-type: doc
Install with `pip install virtualmodelcontrol`, then build your first controller in ten lines.
:::

:::{grid-item-card} Control a soft arm
:link: guides/soft-arm
:link-type: doc
Reach a point, avoid an obstacle, shape the whole body.
:::

:::{grid-item-card} Control a hand
:link: guides/hand
:link-type: doc
Fingertip stiffness, grasps, joint limits, the hand on a UR5.
:::

:::{grid-item-card} Simulate
:link: getting-started/simulate
:link-type: doc
The robot's own masses and springs give its dynamics: test controllers before the hardware.
:::

:::{grid-item-card} Use your robot
:link: how-to/build-a-robot
:link-type: doc
Describe any robot (rigid, continuum, tendon-driven) in a few lines.
:::

:::{grid-item-card} Understand it
:link: concepts/overview
:link-type: doc
How the library works, and how it is organized.
:::
::::

## Ten lines

```python
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")  # a ready-made robot
ctrl = vmc.Mechanism("ctrl")  # the controller: a virtual mechanism
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - [0.1, 0.0, 0.6], 30.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 1.5))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))

meas = vmc.Signals(0.0, motor_position=[0.0] * 9, motor_velocity=[0.0] * 9)
torques = controller.step(0.0, meas)["motor_torque"]  # 9 motor torques [N·m]
```

## Robots included

| Robot | Template | What it is |
|---|---|---|
| Helyx soft arm | `robots.helyx.arm(geometry)` | three tendon-driven continuum segments, three geometries |
| Bimanual Helyx | `robots.bimanual.arms()` | two soft arms on one frame |
| ADAPT finger | `robots.adapt.finger()` | three phalanges, two motors, coupled distal joints |
| ADAPT hand | `robots.adapt.hand()` | five digits, 13 motors, spread coupling |
| Crawling turtle | `robots.turtle.robot()`, `turtle.controller()` | two cranks coordinated by a virtual flywheel |
| UR5 | `robots.ur5.arm()`, `ur5.with_hand()` | six-joint arm, alone or carrying the hand |

```{toctree}
:hidden:
:caption: Getting started

getting-started/install
getting-started/first-controller
getting-started/simulate
```

```{toctree}
:hidden:
:caption: Concepts

concepts/overview
concepts/structure
concepts/coordinates
concepts/components
concepts/parameters
concepts/energy
concepts/pcc
```

```{toctree}
:hidden:
:caption: Guides

guides/soft-arm
guides/two-arms
guides/finger
guides/hand
guides/contact
```

```{toctree}
:hidden:
:caption: Robots

robots/helyx
```

```{toctree}
:hidden:
:caption: How-to

how-to/build-a-robot
how-to/extend
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/conventions
reference/changelog
reference/todo
```

```{toctree}
:hidden:
:caption: Development

development/architecture
development/contributing
```
