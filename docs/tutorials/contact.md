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
$d < 0$, with energy $\tfrac12 k \delta^2$ for a penetration $\delta = \max(0, -d)$. A
`ContactDamper` damps only during contact. Added to the robot mechanism, they model the
environment, which only the simulator feels. Added to the controller, they make a virtual
wall.

## A finger above a table

The table is a plane 6 cm below the finger's base. The finger's $z$ axis points down, so the
table's outward normal is $-z$.

```{code-cell} python
import casadi as ca
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
finger.add("cushion", vmc.ContactDamper(gap, 5.0))  # [N·s/m]
```

## Press with a chosen force

A spring pulls the fingertip towards a goal below the surface. The tip stops on the table, where
the spring's pull balances the table's push. The contact force is then about the spring's
stiffness $K$ times the goal's depth, a little less because the table gives way: $K k / (K + k)$
times the depth.

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

`adapt.joint_limit_spring` is a `LimitSpring` that keeps the finger's joints inside their range.
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
from virtualmodelcontrol.robots import helyx
glue("eta_arm", helyx.EFFICIENCY, display=False)
```

The first touch is an impact: the force peaks at {glue:text}`press_peak:.2f` N while the tip
stops, then settles. Each deeper goal raises the force by one step. The last one settles at
{glue:text}`press_end:.3f` N against {glue:text}`press_goal:.3f` N expected. To press more
gently, lower $K$ or damp the tip more.

This finger delivers the full torque of its motors. A real transmission can pass on less: the
soft arm's tendons deliver a share $\eta$ = {glue:text}`eta_arm:.2f` of each motor torque. The
arm's model does not need $\eta$, because we identified its stiffness and damping from the
commanded torques. Its forces on the surroundings do need it. The spring presses with $\eta K$
times its stretch, so a chosen force needs $1/\eta$ times the stretch, corrected for the arm's
own stiffness and weight. The [efficiency page](../concepts/efficiency.md) explains both.

```{code-cell} python
:tags: [remove-output]
def draw(ax, row):
    ax.axhspan(0.06, 0.09, color="0.88", zorder=0)
    goal = [0.0, 0.05, 0.06 + row["depth"]]
    viz.draw_goal(ax, goal, plane="yz", markersize=10)
    at = kin.position(row["q"], "tip")
    viz.draw_force(ax, at, [0.0, 0.0, -row["force"]], plane="yz",
                   scale=0.015)

viz.animate(finger, log, "contact.mp4", plane="yz", invert=True,
            draw=draw, limits=((-0.01, 0.1), (-0.01, 0.085)))
```

```{video} contact.mp4
:caption: The goal (red cross) steps 5 mm deeper every second; the arrow is the table's force.
```

## Friction

A contact spring keeps the tip out of the table, and friction holds it back along the table.
`ContactFriction(distance, stiffness, friction)` is Coulomb friction, smoothed: a force against
the tip's speed along the surface, of size $\mu F_n$, with $F_n$ the contact spring's own force.
Give it the same `Param` for the stiffness as the spring has, so that they agree. Below a small
speed the force falls linearly with it, so a tip at rest creeps instead of sticking. We press
the tip 1.5 cm below the table and, after one second, move the goal along it at 5 mm/s:

```{code-cell} python
mu = 0.5
rough = adapt.add_dynamics(adapt.finger())
rough_tip = rough.point("tip")
floor = vmc.PlaneDistance(rough_tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
stiffness = vmc.Param("table", 1e4, unit="N/m", scope="stage")  # shared
rough.add("table", vmc.ContactSpring(floor, stiffness))
rough.add("cushion", vmc.ContactDamper(floor, 5.0))
rough.add("rub", vmc.ContactFriction(floor, stiffness, mu))

def goal(t):  # 1.5 cm below the table, moving along it after 1 s
    return ca.vertcat(0.0, 0.05 - 0.005 * ca.fmax(t - 1.0, 0.0), 0.075)

drag = vmc.Mechanism("drag")
drag.add("pull", vmc.LinearSpring(
    rough_tip - vmc.Custom(goal, [vmc.Time()], dim=3, unit="m"), 100.0))
drag.add("damp", vmc.LinearDamper(rough_tip, 1.0))
drag.add("limits", adapt.joint_limit_spring(rough))
drag.add("gravity", vmc.GravityCompensation(rough))
system = vmc.VirtualMechanismSystem(rough, drag)
controller = vmc.VMCController(vmc.compile(system))
plant = vmc.sim.ModelPlant(rough, q0=[0.8, 0.8], max_step=1e-4)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=4.0,
                  record=["elements", "robot"])
rows = log.arrays()
```

`record="robot"` makes the run log what the simulated robot itself feels, component by component,
as `robot/<name>/force`: here the table's push and the friction. The controller's own elements
are logged by `record="elements"`.

```{code-cell} python
t = np.ravel(rows["t"])
normal = rows["robot/table/force"][:, 0]  # [N], the table's push
rub = rows["robot/rub/force"][:, 1]  # [N], opposes the motion
pull = rows["element/drag.pull/force"][:, 1]  # [N], the controller's spring

settled = t > 0.3  # leaves out the first bounces on the table
fig, ax = plt.subplots()
ax.plot(t[settled], -pull[settled], lw=6, alpha=0.5,
        label="the goal's pull")
ax.plot(t[settled], rub[settled], label="friction")
ax.plot(t[settled], mu * normal[settled], "--", label=r"$\mu F_n$")
ax.set_xlabel("time [s]")
ax.set_ylabel("force along the table [N]")
ax.legend(loc="center right");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

slide = t > 3.0
assert abs(rub[slide].mean() - mu * normal[slide].mean()) < 0.03 * mu * normal[slide].mean()
assert rub[(t > 0.5) & (t < 1.0)].mean() < 0.3 * mu * normal[slide].mean()  # small at rest
glue("friction_limit", float(mu * normal[slide].mean()), display=False)
glue("friction_slide", float(rub[slide].mean()), display=False)
```

The goal moves away and the pull grows; friction grows with it and holds the tip, which creeps,
until the pull reaches the limit $\mu F_n$, {glue:text}`friction_limit:.2f` N here. Then the tip slides
at the goal's pace, with a friction of {glue:text}`friction_slide:.2f` N: a little under the limit,
because the sliding speed of 5 mm/s is not far above the smoothing speed of 1 mm/s.

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
ctrl.add("cushion", vmc.ContactDamper(floor, 5.0))
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

`SphereDistance` gives the distance to a ball, for grasps and obstacles. `BoxDistance`,
`CapsuleDistance` and `CylinderDistance` do the same for a box with its sides along the axes, a
capsule (a segment with a radius) and a cylinder. A body that can touch along its length, such
as a soft arm, takes a contact on several of its points. Any other signed distance written with
CasADi operations works through a `Custom` coordinate. Here is a pole of radius 2 cm along $z$,
beside the soft arm's tip:

```{code-cell} python
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
pole = vmc.CylinderDistance(arm.point(s=1.0), center=[0.1, 0.0, 0.3],
                            axis=[0.0, 0.0, 1.0], radius=0.02,
                            half_height=0.5)
arm.add("pole", vmc.ContactSpring(pole, 2e4))
list(arm.components)  # the pole is now one of the arm's components
```

## Between two points

A contact needs no fixed surface. Two points of the robot, or of two arms, meet when the distance
between them falls under a width, and `ContactSpring(vmc.Norm(a - b) - width, k)` pushes them
apart with the stiffness of whatever is between them. That is how the
[two-arms example](../examples/two-arms.md) squeezes an object, and the same spring between two
points of one arm is a self-contact. Here two masses of 0.5 kg are pulled together by a virtual
spring and stop on an object 10 cm wide. The run logs the object's force with `record="robot"`:

```{code-cell} python
pair = vmc.Mechanism("pair", model=vmc.models.JointSpace(2, unit="m"))
a, b = pair.joint(0), pair.joint(1)
for i, coord in enumerate((a, b)):
    pair.add(f"m{i}", vmc.Inertance(coord, 0.5))  # [kg]
distance = (b - a) - 0.10  # [m] to the object's surface; negative inside it
pair.add("object", vmc.ContactSpring(distance, 2e3))  # [N/m]
pair.add("cushion", vmc.ContactDamper(distance, 8.0))

squeeze = vmc.Mechanism("squeeze")
squeeze.add("pull", vmc.LinearSpring(b - a, 50.0))  # [N/m]
system = vmc.VirtualMechanismSystem(pair, squeeze)
controller = vmc.VMCController(vmc.compile(system))
plant = vmc.sim.ModelPlant(pair, q0=[0.0, 0.3], max_step=1e-4)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=1.5,
                  record="robot")
rows = log.arrays()
gap = rows["q"][:, 1] - rows["q"][:, 0]  # [m]
on_object = rows["robot/object/force"][:, 0]  # [N]
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
top.plot(np.ravel(rows["t"]), 100 * gap)
top.axhline(10, color=viz.PALETTE[1], ls="--")
top.set_ylabel("distance [cm]")
bottom.plot(np.ravel(rows["t"]), on_object)
bottom.set_xlabel("time [s]")
bottom.set_ylabel("object's force [N]");
```

```{code-cell} python
:tags: [remove-cell]
assert abs(on_object[-1] - 50.0 * gap[-1]) < 0.05 and gap[-1] < 0.1
glue("squeeze_force", float(on_object[-1]), display=False)
glue("squeeze_sink", float(1e3 * (0.10 - gap[-1])), display=False)
glue("squeeze_gap", float(100 * gap[-1]), display=False)
```

The masses stop {glue:text}`squeeze_sink:.1f` mm into the object, where its force,
{glue:text}`squeeze_force:.2f` N, equals the virtual spring's pull: 50 N/m times the distance between the masses,
{glue:text}`squeeze_gap:.2f` cm.
An object with a mass of its own, or with several coordinates, is a part with its own joints
in the robot, as in [Joints and bodies](joints.md); the contact is the same.

### Friction between two points

Two points that touch can rub too, and the surface they rub on is a sphere around one of them.
`vmc.SphereDistance(b - a, [0, 0, 0], width)` is the same distance as `Norm(b - a) - width`, so
`ContactFriction` takes it, with the stiffness `Param` of the spring. It pushes the two points
with equal and opposite forces: it takes energy out of their sliding and nothing out of their
momentum. A mass of 1 kg hits one of 2 kg off center at 1 m/s, first without friction, then
with a coefficient of 0.4:

```{code-cell} python
def collide(mu):
    robot = vmc.Mechanism("pair", model=vmc.models.JointSpace(6, unit="m"))
    a, b = robot.joint(slice(0, 3)), robot.joint(slice(3, 6))
    robot.add("ma", vmc.PointMass(a, 1.0))  # [kg]
    robot.add("mb", vmc.PointMass(b, 2.0))
    k = vmc.Param("k", 1e4, unit="N/m", scope="stage")
    robot.add("touch", vmc.ContactSpring(vmc.Norm(b - a) - 0.10, k))
    sphere = vmc.SphereDistance(b - a, [0.0, 0.0, 0.0], 0.10)
    robot.add("rub", vmc.ContactFriction(sphere, k, mu, speed=1e-4))
    system = vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))
    idle = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(robot, q0=[0, 0, 0, 0.3, 0.04, 0],
                               v0=[1, 0, 0, 0, 0, 0], max_step=1e-4)
    return vmc.sim.run(plant, idle, vmc.sim.SimClock(1e-3), T=0.3).arrays()

def energy(rows):  # [J], kinetic
    first, second = rows["v"][:, :3], rows["v"][:, 3:]  # 1 kg and 2 kg
    return 0.5 * np.sum(first**2, axis=1) + np.sum(second**2, axis=1)

slick, grippy = collide(0.0), collide(0.4)
```

```{code-cell} python
:tags: [remove-input]
t = np.ravel(slick["t"])
fig, ax = plt.subplots()
ax.plot(t, energy(slick), label=r"$\mu = 0$")
ax.plot(t, energy(grippy), label=r"$\mu = 0.4$")
ax.set_xlabel("time [s]")
ax.set_ylabel("kinetic energy [J]")
ax.legend(loc="lower left");
```

```{code-cell} python
:tags: [remove-cell]
def momentum(rows):
    return rows["v"][:, :3] + 2 * rows["v"][:, 3:]

assert np.ptp(momentum(grippy), axis=0).max() < 1e-6
assert energy(grippy)[-1] < 0.9 * energy(grippy)[0] and energy(slick)[-1] > 0.97 * energy(slick)[0]
glue("kept_slick", float(100 * energy(slick)[-1] / energy(slick)[0]), display=False)
glue("kept_grippy", float(100 * energy(grippy)[-1] / energy(grippy)[0]), display=False)
```

The energy dips while the spring holds some of it, and the frictionless pair gets back
{glue:text}`kept_slick:.0f` % of it (the rest is the integrator's). With friction the pair
leaves with {glue:text}`kept_grippy:.0f` %. The momentum is the same in both.

### A soft object

A soft object is a part with coordinates and springs of its own. Here the object is 10 cm wide:
each face is a point with a mass, the faces are joined by a spring, and each tip touches its
face with a contact spring. The controller pulls the tips together with a spring of 50 N/m.

```{code-cell} python
hold = vmc.Mechanism("hold", model=vmc.models.JointSpace(4, unit="m"))
tip_a, left, right, tip_b = (hold.joint(i) for i in range(4))
for i, coord in enumerate((tip_a, left, right, tip_b)):
    hold.add(f"m{i}", vmc.Inertance(coord, 0.5))  # [kg]
    hold.add(f"d{i}", vmc.LinearDamper(coord, 5.0))  # [N s/m]
hold.add("body", vmc.LinearSpring((right - left) - 0.10, 500.0))  # [N/m]
hold.add("face_a", vmc.ContactSpring(left - tip_a, 2e3))
hold.add("face_b", vmc.ContactSpring(tip_b - right, 2e3))

pull = vmc.Mechanism("pull")
pull.add("spring", vmc.LinearSpring(tip_b - tip_a, 50.0))  # [N/m]
controller = vmc.VMCController(
    vmc.compile(vmc.VirtualMechanismSystem(hold, pull)))
plant = vmc.sim.ModelPlant(hold, q0=[0.0, 0.05, 0.15, 0.2], max_step=1e-3)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=4.0,
                  record="robot")
rows = log.arrays()
```

```{code-cell} python
:tags: [remove-cell]
series = 1 / (1 / 500.0 + 2 / 2e3)  # [N/m]
gap = series * 0.10 / (series + 50.0)
q = rows["q"][-1]
assert abs((q[3] - q[0]) - gap) < 1e-3 * gap
assert abs(rows["robot/body/force"][-1][0] - 50.0 * gap) < 1e-3 * 50.0 * gap
glue("soft_series", float(series), display=False)
glue("soft_gap", float(100 * gap), display=False)
glue("soft_force", float(50.0 * gap), display=False)
glue("soft_squeeze", float(1e3 * (0.10 - (q[2] - q[1]))), display=False)
```

At rest the same force goes through the contact, the object and the other contact: three springs
in series, of {glue:text}`soft_series:.0f` N/m together. The pull of 50 N/m balances them
at a gap of {glue:text}`soft_gap:.2f` cm between the tips and a force of
{glue:text}`soft_force:.2f` N, of which the object itself takes up
{glue:text}`soft_squeeze:.1f` mm of squeeze.

## Choosing the numbers

- Stiffness: the penetration at rest is $F / k$; 10³ to 10⁵ N/m model hard surfaces, and a soft
  object is a lower $k$.
- Time step: `ModelPlant`'s implicit steps stay stable at any stiffness. Resolving the impact
  needs `max_step` well below the contact period $2\pi\sqrt{m/k}$. A few grams on 10⁴ N/m give
  a few milliseconds, hence `max_step=1e-4` above.
- Damping: the damping ratio is $D / (2\sqrt{k m})$; without a contact damper the tip bounces.
- Smoothing: `ContactSpring(d, k, smoothing=w)` and `ContactDamper(d, D, smoothing=w)` round the
  corner at the surface over a width $w$ [m], which optimization needs.
- Friction: a contact spring pushes along the normal only, so a pressing tip can slide. Add a
  `ContactFriction` on the same surface, with the spring's stiffness `Param`; `speed` is the
  sliding speed below which the force falls linearly, so a point at rest creeps.
