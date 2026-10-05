---
file_format: mystnb
kernelspec:
  name: python3
---

# Coordinates and components

In this tutorial we build coordinates on the soft arm, attach components to them, and compare
how the springs of the library push.

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

On a continuum robot, `s` runs from 0 at the base to 1 at the tip, uniformly in arc length.
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

A plain list works as a goal too, as in `tip - [0.1, 0.0, 0.40]`. It becomes a live parameter
named `ref` ([Parameters](parameters.md)). For anything else, `vmc.Custom` wraps a function
written with CasADi operations ([Extend the library](extend.md)).

## Virtual states

A controller can have degrees of freedom of its own. `add_state` creates one. It needs an
inertance, which is a virtual mass. Springs and dampers connect it to the robot. Here a virtual
point follows the tip on a spring:

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
follower = ctrl.add_state("follower", dim=3, unit="m",
                          initial=[0.0, 0.0, 0.725])
ctrl.add("mass", vmc.Inertance(follower, 0.05))  # [kg]
ctrl.add("tether", vmc.LinearSpring(tip - follower, 25.0))  # [N/m]
ctrl.add("drag", vmc.LinearDamper(follower, 0.5))  # [N·s/m]
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
law.z0  # the initial state: positions, then velocities
```

At every step the controller moves the state by its equations of motion: the inertance gives it
a mass, and the tether and the drag push it. The same tether pulls the tip towards the
follower, so the tip drags a virtual mass on a spring.

## Components

A component acts on one coordinate $y$, and it is one of four kinds:

- storage: an energy $V(y)$ and the force $f = -\partial V / \partial y$ (springs; gravity on
  the robot);
- dissipation: a force with $f \cdot \dot y \le 0$ (dampers);
- inertance: a mass or inertia $M$, with kinetic energy $\tfrac12 \dot y^\top M \dot y$;
- source: a force whose power $f \cdot \dot y$ is metered (force sources, gravity compensation).

Each force reaches the robot through its coordinate's Jacobian: $\dot y = J v$ and
$\tau = J^\top f$. The motors then receive the torques $u$ with $B(q)\,u = \tau$, where $B$
comes from the robot's transmission. The [conventions](../concepts/conventions.md) give the
signs. These are the components of the library:

| Component | Kind | Force on its coordinate |
| --- | --- | --- |
| `LinearSpring(y, K)` | storage | $-K y$, with $K$ a scalar, one value per axis, or a matrix |
| `TanhSpring(y, k, F)` | storage | $-F \tanh(k y / F)$ per axis: never more than $F$ |
| `GaussianSpring(y, A, σ)` | storage | $A\, e^{-\lVert y \rVert^2 / 2\sigma^2}\, y$: pushes away from $y = 0$ |
| `SigmoidSpring(y, k_min, k_max, d₀, α)` | storage | $-k(d)\, y$, $k$ rising from $k_\min$ to $k_\max$ around $d = d_0$ |
| `PolynomialSpring(y, K, n, d₀)` | storage | $-K (d / d_0)^n\, y$ |
| `LimitSpring(y, k, lower, upper)` | storage | zero inside $[\text{lower}, \text{upper}]$; outside, $k$ times the overshoot, inwards |
| `ContactSpring(d, k)` | storage | $k \max(0, -d)$ along $d$, only in contact ($d < 0$) |
| `Gravity(robot)` | storage | $m_i g$ on each mass of the robot |
| `LinearDamper(y, D)` | dissipation | $-D \dot y$ |
| `TanhDamper(y, D, F)` | dissipation | $-F \tanh(D \dot y / F)$ per axis |
| `ContactDamper(d, D)` | dissipation | $-D \dot d$, only in contact |
| `PointMass(p, m)` | inertance | a mass $m$ at a point |
| `Inertance(y, M)` | inertance | a mass or inertia $M$ on any coordinate |
| `ForceSource(y, f)` | source | $f$, a live parameter |
| `GravityCompensation(robot)` | source | $-m_i g$ on each mass of the robot |
| `SpeedRegulator(y, b, ω, T)` | source | $b\,(\omega r(t) - \dot y)$, the ramp $r$ rising from 0 to 1 in $T$ |

For the sigmoid and polynomial springs, $d$ is the size of each axis of $y$, or its norm with
`element_wise=False`. A new component only needs its energy or its force
([Extend the library](extend.md)).

## The springs side by side

We put six springs on six independent coordinates of a test mechanism, compile them, and read
their forces over a range of deflections.

```{code-cell} python
:tags: [hide-input]
bar = vmc.Mechanism("bar", model=vmc.models.JointSpace(6, unit="m"))
y = [bar.joint(i) for i in range(6)]  # six deflections [m]
springs = {
    "LinearSpring": vmc.LinearSpring(y[0], 10.0),
    "TanhSpring": vmc.TanhSpring(y[1], 10.0, 0.6),
    "GaussianSpring": vmc.GaussianSpring(y[2], 40.0, 0.05),
    "SigmoidSpring": vmc.SigmoidSpring(y[3], 2.0, 20.0, 0.1, 60.0),
    "PolynomialSpring": vmc.PolynomialSpring(y[4], 10.0, 2, 0.1),
    "LimitSpring": vmc.LimitSpring(y[5], 10.0, -0.1, 0.1),
}
probe = vmc.Mechanism("springs")
for name, spring in springs.items():
    probe.add(name, spring)
law = vmc.compile(vmc.VirtualMechanismSystem(bar, probe))

d = np.linspace(-0.3, 0.3, 241)  # [m]
q, v, z = np.tile(d, (6, 1)), np.zeros((6, 1)), np.zeros((0, 1))
forces = law.tau.map(d.size)(q, v, z, law.live_values(), 0.0).full()

fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.6), sharex=True)
for ax, name, f in zip(axes.flat, springs, forces):
    ax.axhline(0.0, color="0.85", lw=1)
    ax.axvline(0.0, color="0.85", lw=1)
    ax.plot(d, f)
    ax.set_title(name)
for ax in axes[-1]:
    ax.set_xlabel("$y$ [m]")
for ax in axes[:, 0]:
    ax.set_ylabel("$f$ [N]")
```

The tanh spring saturates at its maximum force. The Gaussian spring pushes away only near
$y = 0$, around an obstacle. The sigmoid and polynomial springs stiffen with distance. The limit
spring stays at zero inside its range. On a 3D coordinate the tanh spring saturates each axis on
its own. To bound the force along one direction, put it on a `Projection`.
