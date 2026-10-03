---
file_format: mystnb
kernelspec:
  name: python3
---

# Contact: stiff springs that only push

A contact is a very stiff spring that acts only while two things touch. The library builds it
from two pieces:

1. a **signed distance** d from a robot point to a surface: positive outside, zero on the
   surface, negative when the point penetrates;
2. a **contact spring** on d: no force while d > 0; once d < 0 it pushes the point back out,
   with energy V = ½ k δ², where δ = max(0, −d) is the penetration. A **contact damper** adds
   damping that acts only during contact.

Where you add them decides what they mean:

| Added to | It models | Who feels it |
| --- | --- | --- |
| the **robot** | the environment: a table, a wall, an object | the simulator; the controller does not know about it, as on the real robot |
| the **controller** | a virtual wall | the robot, through the motor torques the controller sends |

```{code-cell} python
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

## 1. A finger above a table

The finger template has masses; `adapt.add_dynamics` adds its gravity, which the simulator needs.
The table is a plane 6 cm below the finger's base. The finger's z axis points down, so the
table's normal, which points out of the table towards the finger, is −z.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

viz.use_style(usetex=False, font_size=13)

finger = adapt.add_dynamics(adapt.finger())
gap = vmc.PlaneDistance(finger.point("tip"), normal=[0, 0, -1], origin=[0, 0, 0.06])
finger.add("table", vmc.ContactSpring(gap, 1e4))           # stiffness [N/m]
finger.add("table_damping", vmc.ContactDamper(gap, 5.0))   # damping [N·s/m]
```

The table belongs to the robot mechanism, so it is part of the simulated world. A controller
built on `finger` ignores it, because a controller only uses its own components.

## 2. Press with a chosen force

A spring pulls the fingertip towards a goal placed *below* the table's surface. The tip cannot
reach the goal; it stops on the table and pushes on it. At rest the spring's push equals the
table's reaction, so the contact force is

$$
F = K \times \text{depth of the goal below the surface}.
$$

With K = 100 N/m, a goal 5 mm deep presses with 0.5 N, 10 mm with 1 N. (The table itself gives a
little: the two springs act in series, so the force is depth × K k / (K + k), here 1% less.)

```{code-cell} python
K = 100.0                                                     # [N/m]
goal = vmc.Ref("goal", 3, value=[0.0, 0.05, 0.06])           # a named goal, to move it later
ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(finger.point("tip") - goal, K))
ctrl.add("damp", vmc.LinearDamper(finger.point("tip"), 1.0))
ctrl.add("limits", adapt.joint_limit_spring(finger))
ctrl.add("gravity", vmc.GravityCompensation(finger))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl)))
```

The loop below is the run loop written out: read, step, write, advance. Every second it moves the
goal 5 mm deeper with `controller.set`, and it records the contact force k × max(0, −d) from the
tip's height.

```{code-cell} python
plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
kin = vmc.Kinematics(finger)
dt, t, force = 1 / 500, [], []
for k in range(2000):
    if k % 500 == 0:
        depth = 0.005 * (k // 500)                            # 0, 5, 10, 15 mm
        controller.set({"ctrl.press.goal": [0.0, 0.05, 0.06 + depth]})
    plant.write(controller.step(plant.t, plant.read()))
    plant.advance(dt)
    d = 0.06 - kin.position(plant.q, "tip")[2]                # tip height above the table
    t.append(plant.t)
    force.append(1e4 * max(0.0, -d))
```

```{code-cell} python
:tags: [hide-input]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1, 1.5])
ax0.axhspan(0.06, 0.08, color="0.88", zorder=0)
ax0.axhline(0.06, color="0.45", lw=1.2)
ax0.text(0.015, 0.07, "table", color="0.4", ha="center")
viz.draw_robot(ax0, finger, [0.8, 0.8], plane="yz", color="0.75", label="start")
viz.draw_robot(ax0, finger, plant.q, plane="yz", label="pressing")
for depth in (0.005, 0.010, 0.015):
    viz.draw_goal(ax0, [0.0, 0.05, 0.06 + depth], plane="yz", markersize=8, markeredgewidth=2)
tip = kin.position(plant.q, "tip")
viz.draw_force(ax0, tip, [0.0, 0.0, -force[-1]], plane="yz", scale=0.012)
ax0.invert_yaxis()                                            # z points down
viz.label_axes(ax0, "yz")
ax0.legend(loc="upper right", fontsize=11)
ax0.set_title("goals below the surface")

ax1.plot(t, force, color=viz.PALETTE[0], lw=1.6, label="contact force")
for i, f in enumerate((0.0, 0.5, 1.0, 1.5)):
    ax1.hlines(f * 100 / 101, i, i + 1, colors="0.3", linestyles="--", lw=1.2,
               label="K × depth (series)" if i == 1 else None)
ax1.set_xlabel("time [s]")
ax1.set_ylabel("force [N]")
ax1.legend(loc="upper left", fontsize=11)
ax1.set_title("the force follows the goal's depth");
```

The first touch is an impact: the force peaks while the tip stops, then settles. Each later step
overshoots a little and settles on K × depth. To press more gently, lower K or raise the tip
damping.

## 3. A virtual wall

The same two components, added to the **controller** instead of the robot, make a wall the
robot is told to respect. Here the fingertip may not go below z = 6 cm, however deep the goal
is:

```python
floor = vmc.PlaneDistance(finger.point("tip"), normal=[0, 0, -1], origin=[0, 0, 0.06])
ctrl.add("floor", vmc.ContactSpring(floor, 2000.0))  # a stiff virtual floor
ctrl.add("floor_damping", vmc.ContactDamper(floor, 5.0))
```

The tip then stops a little past the floor, by about depth × K / k_floor, where the goal's spring and
the floor balance. The floor's stiffness is a gain: `controller.set` changes it while running
and returns the energy the change adds or removes.

## 4. Other surfaces

**A ball**, for grasps or obstacles. `SphereDistance` is ‖p − c‖ − r. Here a ball of 2 cm
radius that the thumb, index and middle fingertips can touch:

```python
hand = adapt.add_dynamics(adapt.hand())
for digit in ("thumb", "index", "middle"):
    d = vmc.SphereDistance(hand.point(f"{digit}/tip"), center=[0.03, 0.09, 0.07], radius=0.02)
    hand.add(f"ball_{digit}", vmc.ContactSpring(d, 5000.0))
```

**A body, not a point.** A contact acts on one point. For a body that can touch along its length,
such as the soft arm, put a contact on several points:

```python
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
for i, s in enumerate((0.25, 0.5, 0.75, 1.0)):
    d = vmc.PlaneDistance(arm.point(s=s), normal=[-1, 0, 0], origin=[0.15, 0, 0])
    arm.add(f"wall{i}", vmc.ContactSpring(d, 2e4))
    arm.add(f"wall_damping{i}", vmc.ContactDamper(d, 20.0))
```

**Any shape.** Any signed distance works, written with CasADi operations in a `Custom`
coordinate. A vertical pole of radius 2 cm at x = 0.1 m:

```python
import casadi as ca

pole = vmc.Custom(
    lambda p: ca.norm_2(p[0:2] - [0.1, 0.0]) - 0.02, [arm.point(s=1.0)], dim=1, unit="m"
)
arm.add("pole", vmc.ContactSpring(pole, 2e4))
```

## 5. Choosing the numbers

- **Stiffness k.** The penetration at rest is F / k. Values from 10³ to 10⁵ N/m model hard
  surfaces; a soft object is just a lower k.
- **Time step.** `ModelPlant` takes implicit steps, stable for any stiffness. To resolve the
  impact, keep `max_step` well below the contact period 2π √(m / k), where m is the mass that
  hits: here a few grams and 10⁴ N/m give about 4 ms, hence `max_step=1e-4`.
- **Damping D.** Damping ratio ζ = D / (2 √(k m)). Without a contact damper the tip bounces.
- **Smoothing.** `ContactSpring(d, k, smoothing=w)` and `ContactDamper(d, D, smoothing=w)` round
  the corner at the surface over a width w [m], so forces and their derivatives are smooth: use
  it in optimization. Forces change only within a few w of the surface.
- **Friction.** These contacts are frictionless: the force is along the surface normal, and a
  pressing tip can slide along the table. Tangential friction is on the
  [to-do list](https://github.com/vigno0405/VirtualModelControl/blob/main/TODO.md).
