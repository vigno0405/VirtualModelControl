---
file_format: mystnb
kernelspec:
  name: python3
---

# Two arms: squeeze an object

The bimanual Helyx is two soft arms on one frame. As one robot, its configuration stacks both
arms (right first), and a single controller can couple them: here a spring between the two tips
squeezes an object.

```{code-cell} python
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import bimanual

viz.use_style(usetex=False, font_size=13)
arms = bimanual.add_dynamics(bimanual.arms())
right_tip = arms.point("right", s=1.0)
left_tip = arms.point("left", s=1.0)
```

## The controller

A spring between the tips pulls them to 15 cm apart; dampers on the tips calm the motion; gravity
compensation acts on both arms.

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
ctrl.add("squeeze", vmc.LinearSpring(right_tip - left_tip - [0.15, 0.0, 0.0], 100.0))
ctrl.add("damp_right", vmc.LinearDamper(right_tip, 2.0))
ctrl.add("damp_left", vmc.LinearDamper(left_tip, 2.0))
ctrl.add("gravity", vmc.GravityCompensation(arms))

controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arms, ctrl)))
plant = vmc.sim.ModelPlant(arms)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / bimanual.CONTROL_RATE), T=2.0)

fig, ax = plt.subplots(figsize=(7, 5))
viz.draw_robot(ax, arms, np.zeros(18), color="0.85")
viz.draw_robot(ax, arms, plant.q)
kin = vmc.Kinematics(arms)
viz.draw_spring(ax, kin.position(plant.q, ("left", 1.0)), kin.position(plant.q, ("right", 1.0)))
viz.label_axes(ax)
ax.set_title("tips pulled towards each other");
```

## What is in the template

- Bases at x = ±0.125 m, both arms pointing up (gravity −z).
- Each arm's identified stiffness and damping (`bimanual.STIFFNESS`, `bimanual.DAMPING`) and the
  transmission efficiency η = 0.12: the arms receive 12% of the commanded torque, which the
  simulator models. Commands are not scaled unless you add `EfficiencyCorrection` to the
  output stage.
- The real arms' output stage (`bimanual.output_stage()`): a torque offset, a soft stop on slack
  tendons, a ±0.5 N·m clip, all opt-in.
