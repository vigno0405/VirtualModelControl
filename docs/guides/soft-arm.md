---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft arm: reach, avoid, shape

This guide controls the Helyx soft arm (three tendon-driven segments, nine motors) in
simulation: first the tip reaches a point, then the body avoids an obstacle on the way, then
several springs shape the whole arm. The same controllers run on the real arm.

```{code-cell} python
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

## 1. The robot

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx

viz.use_style(usetex=False, font_size=13)
arm = helyx.add_dynamics(helyx.arm("145-290-290"))   # geometry, tendons, masses + stiffness, damping, gravity
print(arm)
```

`helyx.arm` builds the arm (kinematics, tendons, masses); `helyx.add_dynamics` adds its
identified stiffness and damping and gravity, which only the simulator needs.

## 2. Reach a point

A spring pulls the tip (`s=1.0`) to the goal, a damper on the tip removes oscillations, and
gravity compensation cancels the weight of the segments.

```{code-cell} python
goal = np.array([0.25, 0.0, 0.55])

def reach_controller(arm):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 600.0))   # [N/m]
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))            # [N·s/m]
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return ctrl

def simulate(arm, ctrl, T=3.0):
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    plant = vmc.sim.ModelPlant(arm)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=T)
    return plant.q, log

q_reach, log = simulate(arm, reach_controller(arm))
```

## 3. Avoid an obstacle

Repulsive Gaussian springs on several points of the body push it away from the obstacle. The
obstacle is a plain point; each spring acts on the distance between a body point and it.

```{code-cell} python
obstacle = np.array([0.12, 0.0, 0.44])

ctrl = reach_controller(arm)
for i, s in enumerate(np.linspace(0.4, 0.9, 6)):
    ctrl.add(f"avoid{i}", vmc.GaussianSpring(arm.point(s=s) - obstacle, 4000.0, 0.05))
q_avoid, _ = simulate(arm, ctrl)

fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
for ax, q, title in zip(axes, (q_reach, q_avoid), ("reach", "reach and avoid")):
    viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
    viz.draw_robot(ax, arm, q)
    viz.draw_goal(ax, goal)
    ax.add_patch(plt.Circle((obstacle[0], obstacle[2]), 0.035, color=viz.PALETTE[1], alpha=0.3))
    viz.label_axes(ax)
    ax.set_title(title)
axes[0].invert_yaxis()    # z points down: the arm hangs
```

Without the repulsive springs the body crosses the obstacle; with them it bends around it, and
the tip settles a little short of the goal, where the pull of the goal and the push of the
obstacle balance. Stronger or wider repulsion keeps more clearance.

## 4. How the tip moved

The run log records every step: time, measured motor angles and rates, commanded torques.

```{code-cell} python
rows = log.arrays()
kin = vmc.Kinematics(arm, coordinates="motors")   # positions from motor angles
distance = [np.linalg.norm(kin.position(theta, 1.0) - goal) for theta in rows["motor_position"]]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(rows["t"], 100 * np.array(distance))
ax.set_xlabel("time [s]")
ax.set_ylabel("tip to goal [cm]");
```

## 5. Shape the whole body

Springs can sit anywhere along the arm. Here the middle of the arm is held at one point and the
tip at another; a cart spring (a projection) only acts vertically on the tip.

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
ctrl.add("middle", vmc.LinearSpring(arm.point(s=0.5) - [0.08, 0.0, 0.33], 800.0))
ctrl.add("tip_height", vmc.LinearSpring(vmc.Projection(arm.point(s=1.0) - [0.0, 0.0, 0.62], [0, 0, 1]), 800.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
q_shape, _ = simulate(arm, ctrl)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
viz.draw_robot(ax, arm, np.zeros(9), color="0.85")
viz.draw_robot(ax, arm, q_shape)
viz.draw_goal(ax, [0.08, 0.0, 0.33])
ax.axhline(0.62, color=viz.PALETTE[2], lw=1.5, ls="--", label="tip height")
ax.invert_yaxis()
viz.label_axes(ax)
ax.legend(loc="lower left");
```

## On the real arm

Nothing changes in the controller: the hardware is another plant that reads motor angles and
writes motor torques. Two things to remember:

- **Motor signs.** The library's convention is θ > 0 pulls a tendon. `helyx.ENCODER_SIGN` gives
  each arm's encoder sign: multiply raw angles and torques by it.
- **Output stage.** The real arm uses a small pretension, opt-in:
  `vmc.VMCController(law, output=helyx.output_stage())`.
