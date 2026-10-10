---
file_format: mystnb
kernelspec:
  name: python3
---

# Coordinates and components

In this tutorial we build coordinates on the soft arm: points, joints, differences, projections,
virtual states of the controller, and a cart on a rail. [Components](components.md) says what
to attach to them.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## Points and joints

A coordinate is a quantity that a component acts on. The simplest ones come from the robot's
model: points of its body and entries of its configuration $q$.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-145-145")
tip = arm.point(s=1.0)  # the tip, 3 entries [m]
middle = arm.point(s=0.5)  # halfway along the arm
segment1 = arm.joint(slice(0, 3))  # the first segment's q, 3 entries [m]
tip, middle, segment1
```

On a continuum robot, `s` runs from 0 at the base to 1 at the tip, uniformly in rest length (the arc length while no segment stretches).
Rigid robots and hands name their points instead, as in `finger.point("tip")` or
`hand.point("index/tip")`. `offset=` moves the point within that frame.

## New coordinates from old ones

Subtracting, projecting, measuring, slicing and stacking coordinates gives new ones, and the
library differentiates all of them exactly.

```{code-cell} python
goal = vmc.Ref("goal", value=[0.1, 0.0, 0.40])  # a goal [m]
reach = tip - goal  # where the tip is, minus where it should be
height = vmc.Projection(reach, [0.0, 0.0, 1.0])  # along z only
distance = vmc.Norm(middle - tip)  # [m]
wall = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.45])
for c in (reach, height, distance, wall, reach[0], vmc.Stack(tip, middle)):
    print(c)
```

A projection, a norm and a distance are one number each: a spring on `height` pulls along $z$
only. The constrained elements, such as `vmc.ConstrainedLinearSpring(reach, k, normal=n)`, are
ready-made springs and dampers on such a projection. `PlaneDistance` and `SphereDistance` are
signed distances to a surface, for [contacts](contact.md).

A sum works like a difference: `tip + [0.0, 0.0, 0.05]` is the point 5 cm above the tip. A plain
list works as a goal too, as in `tip - [0.1, 0.0, 0.40]`. It becomes a live parameter named
`ref` ([Parameters](parameters.md)). For anything else, `vmc.Custom` wraps a function
written with CasADi operations ([Extend the library](extend.md)).

## Virtual states

A controller can have degrees of freedom of its own. `add_state` creates one. It needs an
inertance, which is a virtual mass. Springs and dampers connect it to the robot. Here a virtual
point follows the tip on a spring:

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
follower = ctrl.add_state("follower", dim=3, unit="m",
                          initial=[0.0, 0.0, 0.435])  # the tip at rest
ctrl.add("mass", vmc.Inertance(follower, 0.05))  # [kg]
ctrl.add("tether", vmc.LinearSpring(tip - follower, 25.0))  # [N/m]
ctrl.add("drag", vmc.LinearDamper(follower, 0.5))  # [N·s/m]
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
law.z0  # the initial state: positions, then velocities
```

At every step the controller moves the state by its equations of motion: the inertance gives it
a mass, and the tether and the drag push it. The same tether pulls the tip towards the
follower, so the tip drags a virtual mass on a spring.

## A cart on a rail

A virtual state is only a number. To give it a geometry, take a point of any model at the
state's value: `vmc.FramePoint(model, site, q=state)`. The model here is a `SerialChain` with a
[rail](joints.md#a-rail-along-a-path): a curve in front of the arm, with a cart that runs along it. A spring
ties the arm's tip to the cart, so the tip is pulled along the curve, wherever the cart goes.

The coordinate `q` is where the cart is on the rail, and anything can give it: a virtual state
(as above, a free cart that the tip drags along); a `vmc.Ref` (a cart held where the reference
says, which `controller.set` moves); or a function of time, built from `vmc.Time()`. We take
the last, a smooth ramp that starts after 1.5 s and takes 4 s, so the tip follows the curve
slowly:

```{code-cell} python
import casadi as ca
from virtualmodelcontrol.models import SerialChain

arm_up = helyx.add_dynamics(helyx.arm("290-145-145"))  # points up
angle = np.linspace(-0.3, 0.3, 7)  # an arc about the base [rad]
curve = 0.57 * np.column_stack([np.sin(angle), 0 * angle, np.cos(angle)])
rail = SerialChain([("rail", curve)], axes=[None], points=[[0, 0, 0]],
                   sites={"cart": (1, curve[0])})  # at the first point

def ramp(t):  # 0 until 1.5 s, then 0 to 1 smoothly in 4 s
    u = ca.fmin(ca.fmax((t - 1.5) / 4.0, 0.0), 1.0)
    return 3 * u**2 - 2 * u**3

cart = vmc.FramePoint(rail, "cart", q=vmc.Custom(ramp, [vmc.Time()], dim=1))
tip_up = arm_up.point(s=1.0)
follow = vmc.Mechanism("follow")
follow.add("tie", vmc.LinearSpring(tip_up - cart, 600.0))  # [N/m]
follow.add("damp", vmc.LinearDamper(tip_up, 5.0))  # [N·s/m]
follow.add("gravity", vmc.GravityCompensation(arm_up))
system = vmc.VirtualMechanismSystem(arm_up, follow)
controller = vmc.VMCController(vmc.compile(system))

plant = vmc.sim.ModelPlant(arm_up)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=6.5)
rows = log.arrays()
kin = vmc.Kinematics(arm_up)
tip_path = np.array([kin.position(q, 1.0) for q in rows["q"]])
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

along = vmc.Kinematics(rail)
points = np.array([along.position([u], "cart") for u in np.linspace(0, 1, 200)])
away = np.array([np.linalg.norm(points - p, axis=1).min() for p in tip_path])
settled = np.ravel(rows["t"]) > 2.0  # the tip has caught up with the cart
assert away[settled].max() < 0.04 and np.ptp(tip_path[:, 0]) > 0.25
glue("rail_away", float(100 * away[settled].max()), display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 2], "--", label="rail")
ax.plot(tip_path[:, 0], tip_path[:, 2], label="tip")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_aspect("equal")
ax.legend(fontsize=18);
```

After it has caught up with the cart, the tip stays within {glue:text}`rail_away:.1f` cm of the
curve. The arm hardly stretches or shortens, so the curve has to stay about as far from the
base as the arm is long. A goal can move in the same way:
`vmc.Custom(f, [vmc.Time()], dim=3)` is a point that follows $f(t)$, and a damper on the
tip's difference to it feels the goal's velocity.
