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
$[b_{i-1}, b_i]$ with $b_i = \sum_{j \le i} L_{0,j} / \sum_j L_{0,j}$, and each segment's frame
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
motor angles give $\Delta$ exactly, and the motor torques that realize a generalized force
$\tau$ solve $B u = \tau$ with $B = (\partial\theta_m/\partial\Delta)^\top$. A transmission
with efficiency $\eta$ delivers $\eta\,\tau$.
