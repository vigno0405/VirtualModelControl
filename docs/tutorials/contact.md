---
file_format: mystnb
kernelspec:
  name: python3
---

# Contact

In this tutorial we press a fingertip on a table with a chosen force, then use the same
elements as a virtual wall that the controller enforces.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## Signed distances and contact springs

A contact needs a signed distance $d$ from a robot point to a surface: positive outside, zero
on the surface, negative inside. A `ContactSpring` on $d$ pushes the point out only while
$d < 0$, with energy $\tfrac12 k \delta^2$ for a penetration $\delta = \max(0, -d)$, and a
`ContactDamper` damps only during contact. Added to the robot mechanism, they model the
environment, which only the simulator feels; added to the controller, they make a virtual
wall.

## A finger above a table

The table is a plane 6 cm below the finger's base. The finger's $z$ axis points down, so the
table's outward normal is $-z$.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

k = 1e4  # [N/m], a hard table
finger = adapt.add_dynamics(adapt.finger())
tip = finger.point("tip")
gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
finger.add("table", vmc.ContactSpring(gap, k))
finger.add("table_damping", vmc.ContactDamper(gap, 5.0))  # [N·s/m]
```

## Press with a chosen force

A spring pulls the fingertip towards a goal below the surface. The tip stops on the table, where
the spring's pull balances the table's push, so the contact force is the spring's stiffness $K$
times the goal's depth (a little less, because the table gives way: $K k / (K + k)$ times the
depth).

```{code-cell} python
K = 100.0  # [N/m]
goal = vmc.Ref("goal", 3, value=[0.0, 0.05, 0.06])  # a goal we move later

ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(tip - goal, K))
ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
ctrl.add("limits", adapt.joint_limit_spring(finger))
ctrl.add("gravity", vmc.GravityCompensation(finger))
system = vmc.VirtualMechanismSystem(finger, ctrl)
controller = vmc.VMCController(vmc.compile(system))
```

We write the run loop by hand so that every second it moves the goal 5 mm deeper with
`controller.set`. Each step records the time, the configuration, the goal's depth and the
table's force $k \max(0, -d)$.

```{code-cell} python
plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
kin = vmc.Kinematics(finger)
log = {"t": [], "q": [], "depth": [], "force": []}
for step in range(2000):  # 4 s at 500 Hz
    depth = 0.005 * (step // 500)  # 0, 5, 10, 15 mm
    if step % 500 == 0:
        controller.set({"ctrl.press.goal": [0.0, 0.05, 0.06 + depth]})
    plant.write(controller.step(plant.t, plant.read()))
    below = kin.position(plant.q, "tip")[2] - 0.06  # [m] under the surface
    row = (plant.t, plant.q, depth, k * max(0.0, below))
    for name, value in zip(log, row):
        log[name].append(value)
    plant.advance(1 / 500)
log = {name: np.array(values) for name, values in log.items()}

fig, ax = plt.subplots()
ax.plot(log["t"], log["force"], label="contact force")
expected = K * k / (K + k) * log["depth"]  # [N]
ax.plot(log["t"], expected, "--", label=r"$Kk/(K+k)$ depth")
ax.set_xlabel("time [s]")
ax.set_ylabel("force [N]")
ax.legend(loc="upper left");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("press_end", float(log["force"][-1]), display=False)
glue("press_goal", float(K * k / (K + k) * log["depth"][-1]), display=False)
glue("press_peak", float(log["force"][:500].max()), display=False)
```

The first touch is an impact: the force peaks at {glue:text}`press_peak:.2f` N while the tip
stops, then settles. Each deeper goal raises the force by one step; the last one settles at
{glue:text}`press_end:.3f` N against {glue:text}`press_goal:.3f` N expected. To press more
gently, lower $K$ or damp the tip more.

```{code-cell} python
:tags: [remove-output]
def draw_table(ax, row):
    ax.axhspan(0.06, 0.09, color="0.88", zorder=0)
    goal = [0.0, 0.05, 0.06 + row["depth"]]
    viz.draw_goal(ax, goal, plane="yz", markersize=10)
    at = kin.position(row["q"], "tip")
    viz.draw_force(ax, at, [0.0, 0.0, -row["force"]], plane="yz",
                   scale=0.015)

viz.animate(finger, log, "contact.mp4", plane="yz", invert=True,
            draw=draw_table, limits=((-0.01, 0.1), (-0.01, 0.085)))
```

```{video} contact.mp4
:caption: The goal (red cross) steps 5 mm deeper every second; the arrow is the table's force.
```

## A virtual wall

The same two components on the controller make a floor that the fingertip may not cross,
however deep its goal. The simulated finger has no table this time: only the controller's floor
stops it.

```{code-cell} python
free = adapt.add_dynamics(adapt.finger())
tip = free.point("tip")
floor = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
k_floor = 2000.0  # [N/m], a gain like any other
depth = 0.015  # [m], the goal below the floor

ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(tip - [0.0, 0.05, 0.06 + depth], K))
ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
ctrl.add("floor", vmc.ContactSpring(floor, k_floor))
ctrl.add("floor_damping", vmc.ContactDamper(floor, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(free))
system = vmc.VirtualMechanismSystem(free, ctrl)
controller = vmc.VMCController(vmc.compile(system))

plant = vmc.sim.ModelPlant(free, q0=[0.8, 0.8], max_step=1e-4)
vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 500), T=1.5)
past = 1000 * (vmc.Kinematics(free).position(plant.q, "tip")[2] - 0.06)
balance = 1000 * K * depth / (K + k_floor)  # [mm]
print(f"tip {past:.3f} mm past the floor;",
      f"K depth / (K + k_floor) = {balance:.3f} mm")
```

The tip stops where the goal's spring and the floor balance. The floor's stiffness is a live
gain: `controller.set` changes it while running and returns the energy that the change adds
or removes.

## Other surfaces

`SphereDistance` gives the distance to a ball, for grasps and obstacles; a body that can touch
along its length, such as a soft arm, takes a contact on several of its points; and any signed
distance written with CasADi operations works through a `Custom` coordinate. Here a vertical
pole of radius 2 cm near the soft arm's tip:

```{code-cell} python
import casadi as ca
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
pole = vmc.Custom(lambda p: ca.norm_2(p[0:2] - [0.1, 0.0]) - 0.02,
                  [arm.point(s=1.0)], dim=1, unit="m")
arm.add("pole", vmc.ContactSpring(pole, 2e4))
vmc.compile_dynamics(arm).energy  # the pole is now part of the arm's world
```

## Choosing the numbers

- Stiffness: the penetration at rest is $F / k$; 10³ to 10⁵ N/m model hard surfaces, and a soft
  object is a lower $k$.
- Time step: `ModelPlant`'s implicit steps stay stable at any stiffness, but resolving the
  impact needs `max_step` well below the contact period $2\pi\sqrt{m/k}$; a few grams on
  10⁴ N/m give a few milliseconds, hence `max_step=1e-4` above.
- Damping: the damping ratio is $D / (2\sqrt{k m})$; without a contact damper the tip bounces.
- Smoothing: `ContactSpring(d, k, smoothing=w)` and `ContactDamper(d, D, smoothing=w)` round the
  corner at the surface over a width $w$ [m], which optimization needs.
- Friction: these contacts push along the normal only, so a pressing tip can slide; tangential
  friction is planned for release 0.5.0.
