---
file_format: mystnb
kernelspec:
  name: python3
---

# Components

In this tutorial we list the components of the library, try a damper that works one way, and
compare how the springs push. A component acts on a [coordinate](coordinates.md).

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The list

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
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc

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
