---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft-arm kinematics

A continuum arm such as the Helyx is a chain of segments, each bent into a circular arc
(piecewise constant curvature, PCC). This page gives the equations the `PCC` model computes.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
:tags: [remove-input]
from schematics import pcc
pcc.figure();
```

## One segment

Segment $i$ has the coordinates $\Delta_i = (D_x, D_y, D_l)$ [m]: $D_x$ and $D_y$ bend it
towards $(D_x, D_y)$, and $D_l$ lengthens it. With rest length $L_0$, section radius $d$ and
local arc fraction $s \in [0, 1]$,

$$
\begin{gathered}
D = \sqrt{D_x^2 + D_y^2 + \varepsilon}, \\
\theta(s) = s\,D/d .
\end{gathered}
$$

The frame at $s$ is turned by $\theta$ about the axis $(-D_y, D_x, 0)/D$, and its origin is

$$
t(s) = \frac{d\,(L_0 + D_l)}{D^2}
\begin{bmatrix} D_x (1 - \cos\theta) \\ D_y (1 - \cos\theta) \\ D \sin\theta \end{bmatrix},
$$

the point at arc length $s\,(L_0 + D_l)$ on an arc of radius $d\,(L_0 + D_l)/D$. The tangent
is the frame's $z$ axis. The small $\varepsilon$ ($10^{-12}$ m²) keeps the straight pose
smooth.

## The whole arm

The configuration stacks the segments, base to tip: $q = (\Delta_1, \dots, \Delta_n)$. The arc
parameter $s \in [0, 1]$ of the whole arm is uniform in arc length, so segment $i$ spans
$[b_{i-1}, b_i]$ with $b_i = \sum_{j \le i} L_{0,j} / \sum_j L_{0,j}$. Each segment's frame
starts where the previous one ends. `arm.point(s=...)` accepts a symbolic $s$ as well, so an
attachment point can itself be optimized.

## Tendons

A tendon at angle $\delta$ around the section and distance $d$ from the backbone changes
length by

$$
\Delta L = D_l - D_x \cos\delta - D_y \sin\delta ,
$$

and its motor, with spool radius $r$, turns by $\theta_m = -\Delta L / r$: a positive motor
angle pulls the tendon. With three tendons per segment this map is invertible, so the measured
motor angles give $\Delta$ exactly. The motor torques that realize a generalized force $\tau$
solve $B u = \tau$ with $B = (\partial\theta_m/\partial\Delta)^\top$. The transmission's
[efficiency](efficiency.md), 1 by default, maps the commanded motor torques to the delivered
ones.

## The templates

`robots.helyx` builds Helyx arms from these equations. Three geometries are ready: the soft
arm of the [soft-arm example](../examples/soft-arm.md) (on its side), the
[hanging soft arm](../examples/hanging-arm.md) and the arms of the
[two-arm example](../examples/two-arms.md). Their segment lengths, their mounting and the sign
of their encoders against the convention above are:

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown
from virtualmodelcontrol.robots import helyx

rows = ["| Geometry | Segments, base to tip | Gravity in the base frame | Encoder sign |",
        "|---|---|---|---|"]
for name, spec in helyx.GEOMETRIES.items():
    lengths = ", ".join(f"{1000 * L:.0f}" for L in spec["L0"]) + " mm"
    gravity = "[" + ", ".join(f"{g:g}" for g in spec["gravity"]) + "] m/s²"
    rows.append(f"| `{name}` | {lengths} | {gravity} | {helyx.ENCODER_SIGN[name]:+.0f} |")
Markdown("\n".join(rows))
```

The template takes its geometry as arguments, so the same function builds an arm of any
segment lengths, radii, tendon angles or masses, with any number of segments:

```{code-cell} python
import numpy as np

short = helyx.arm(
    lengths=(0.2, 0.2),  # [m], two segments
    tendon_angles=np.radians([[0, 120, -120], [60, 180, -60]]),
)
short.actuation.motor_sizes(short.space)  # motor angles and rates
```
