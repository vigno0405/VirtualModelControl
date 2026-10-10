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
[rail](build-a-robot.md): a curve in front of the arm, with a cart that runs along it. A spring
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
| `PhaseSpring(y, K, m, φ₀)` | storage | for $y = (e, \varphi)$, a spring on $e$ with the stiffness $K\,(1 + m \cos(\varphi - \varphi_0))$, and the reaction on the phase $\varphi$; it can saturate and steer a gait |
| `ContactSpring(d, k)` | storage | $k \max(0, -d)$ along $d$, only in contact ($d < 0$) |
| `Gravity(robot)` | storage | $m_i g$ on each mass of the robot |
| `LinearDamper(y, D)` | dissipation | $-D \dot y$ |
| `TanhDamper(y, D, F)` | dissipation | $-F \tanh(D \dot y / F)$ per axis |
| `DiodeDamper(y, D, sign)` | dissipation | $-D \dot y$ in one direction of motion only |
| `ContactDamper(d, D)` | dissipation | $-D \dot d$, only in contact |
| `ContactFriction(d, k, μ)` | dissipation | $-\mu F_n$ against the sliding speed along the surface, $F_n$ the contact spring's force |
| `PointMass(p, m)` | inertance | a mass $m$ at a point |
| `Inertance(y, M)` | inertance | a mass or inertia $M$ on any coordinate |
| `RotationalInertia(R, I)` | inertance | the inertia of a rigid body about its frame, on a `FrameRotation` |
| `ForceSource(y, f)` | source | $f$, a live parameter, bounded in force and in power on request |
| `GravityCompensation(robot)` | source | $-m_i g$ on each mass of the robot |
| `SpeedRegulator(y, b, ω, T)` | source | $b\,(\omega r(t) - \dot y)$, the ramp $r$ rising from 0 to 1 in $T$ |

The arguments are listed in the order the constructors take them, with the symbols of the
formulas; the keyword names (`stiffness`, `damping`, `max_force`, ...) are in the
[API reference](../api/index.md).

An `Inertance` on a difference of two coordinates is an inerter: a mass between them, with a force
$M\,(\ddot y_1 - \ddot y_2)$ that opposes their relative acceleration. A `LimitSpring` on a slice of
the joints, with a lower and an upper bound per joint, gives all of them soft limits.

For the sigmoid and polynomial springs, $d$ is the size of each axis of $y$, or its norm with
`element_wise=False`. A new component only needs its energy or its force
([Extend the library](extend.md)).

## A damper that works one way

A `DiodeDamper(y, D, sign)` damps one direction of motion: the positive rates for `sign=1`, the
negative ones for `sign=-1`, and never adds energy. A mass of 1 kg on a spring of 100 N/m, let
go at 0.1 m, swings back and forth. Damped both ways it loses its swing at the same rate on each
side (with 2 N·s/m). Damped on the way up only, with twice that, because it works on half of
each swing, it still swings down almost as far as it started:

```{code-cell} python
def swing(damper):
    """The height of a mass on a spring, let go at 0.1 m."""
    mass = vmc.Mechanism("mass", model=vmc.models.JointSpace(1, unit="m"))
    y = mass.joint(0)
    mass.add("inertia", vmc.Inertance(y, 1.0))
    mass.add("spring", vmc.LinearSpring(y, 100.0))
    mass.add("damper", damper(y))
    plant = vmc.sim.ModelPlant(mass, q0=[0.1], max_step=1e-4)
    out = []
    for _ in range(300):
        plant.advance(0.01)
        out.append(plant.q[0])
    return np.array(out)

swings = {"both ways": swing(lambda y: vmc.LinearDamper(y, 2.0)),
          "upwards only": swing(lambda y: vmc.DiodeDamper(y, 4.0))}
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
for name, x in swings.items():
    ax.plot(0.01 * np.arange(1, 301), x, label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel("position [m]")
ax.legend(fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

one_way = swings["upwards only"]
assert one_way.min() < -0.09 and swings["both ways"].min() > -0.08
glue("lowest", float(one_way.min()), display=False)
```

Its first swing down reaches {glue:text}`lowest:.3f` m.

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
