---
file_format: mystnb
kernelspec:
  name: python3
---

# Components

A **component** acts on one coordinate. There are four kinds:

| Kind | What it does | Examples |
|---|---|---|
| storage | stores energy `V(y) ≥ 0`; pushes with `f = −∂V/∂y` | springs, gravity |
| dissipation | removes energy: `f · ẏ ≤ 0` | dampers |
| inertance | gives the coordinate a mass `M` | point masses, a virtual flywheel |
| source | supplies a force; its power is metered | force sources, gravity compensation, speed regulators |

## Springs at a glance

The force each spring exerts on its coordinate, as a function of the deflection `y` (computed by
the library itself):

```{code-cell} python
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
%config InlineBackend.figure_formats = ['svg']
viz.use_style(usetex=False, font_size=13)

robot = vmc.Mechanism("bar", model=vmc.models.JointSpace(1, unit="m"))
y = robot.joint(0)
springs = {
    "LinearSpring": vmc.LinearSpring(y, 10.0),
    "TanhSpring": vmc.TanhSpring(y, 10.0, 0.6),
    "GaussianSpring (repulsive)": vmc.GaussianSpring(y, 40.0, 0.05),
    "SigmoidSpring": vmc.SigmoidSpring(y, 2.0, 20.0, 0.1, 60.0),
    "PolynomialSpring": vmc.PolynomialSpring(y, 10.0, 2, 0.1),
    "LimitSpring": vmc.LimitSpring(y, 10.0, -0.1, 0.1),
}
grid = np.linspace(-0.3, 0.3, 241)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
for ax, (name, spring) in zip(axes.flat, springs.items()):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("k", spring)
    c = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    f = [c.step(0.0, vmc.Signals(0.0, motor_position=[v], motor_velocity=[0.0]))["motor_torque"][0]
         for v in grid]
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.85", lw=1)
    ax.plot(grid, f)
    ax.set_title(name, fontsize=13)
for ax in axes[1]:
    ax.set_xlabel("deflection $y$ [m]")
for ax in axes[:, 0]:
    ax.set_ylabel("force $f$ [N]")
```

| Spring | Force | Parameters | Use it for |
|---|---|---|---|
| `LinearSpring(c, k)` | `f = −k y` | stiffness `k` (scalar, per axis, or matrix) | holding a point or a posture |
| `TanhSpring(c, k, F)` | `f = −F tanh(k y / F)` | slope `k`, saturation `F` | a spring that never pulls harder than `F` |
| `GaussianSpring(c, A, σ)` | `f = A exp(−‖y‖²/2σ²) y` | strength `A`, width `σ` | pushing away from obstacles |
| `SigmoidSpring(c, k_min, k_max, d₀, α)` | `f = −k(d) y` | stiffness from `k_min` to `k_max` around `d₀` | soft near the goal, stiff far away |
| `PolynomialSpring(c, K, n, d₀)` | `f = −K (d/d₀)ⁿ y` | stiffness `K` at `d₀`, order `n` | progressive stiffening |
| `LimitSpring(c, k, lower, upper)` | zero inside the range, `−k·overshoot` outside | range, stiffness | joint limits |

Per-axis (`element_wise=True`, the default) or radial (`element_wise=False`) distances `d` are
available for the sigmoid and polynomial springs.

## Dampers, masses and sources

| Component | Force or role | Use it for |
|---|---|---|
| `LinearDamper(c, D)` | `f = −D ẏ` | damping a point or a joint |
| `TanhDamper(c, D, F)` | `f = −F tanh(D ẏ / F)` | bounded damping |
| `PointMass(c, m)` | mass at a point | the robot's own masses (dynamics, gravity) |
| `Inertance(c, M)` | mass or inertia on a coordinate | a controller's virtual flywheel |
| `ForceSource(c, f)` | constant force | a bias, a push |
| `GravityCompensation(robot)` | `−m g` on each of the robot's masses | cancelling gravity |
| `Gravity(robot)` | `m g` on each mass | the robot's weight, in simulation |
| `SpeedRegulator(c, b, ω, T)` | `b (ω ramp(t) − ẏ)` | driving a virtual oscillator at a speed |

## Acting along one direction

Put any spring on a `Projection` to make it act along one direction only. A tanh spring on a
projection never exceeds its maximum force, whatever the direction:

```python
cart = vmc.Projection(arm.point(s=1.0) - goal, direction=[1.0, 1.0, 0.0])
ctrl.add("cart", vmc.TanhSpring(cart, 50.0, 0.5))
```

## Your own components

A new spring only needs its energy; the force follows automatically. See
[Extend the library](../how-to/extend.md).
