# Piecewise constant curvature

A continuum robot made of $n$ segments, each bending along a circular arc.

## Coordinates

Segment $i$ has $\Delta_i = [D_x, D_y, D_l]$ [m]: $D_x, D_y$ are tendon-length differences (the
segment bends towards $(D_x, D_y)$) and $D_l$ is the elongation. The configuration is
$q = [\Delta_1, \dots, \Delta_n]$, base to tip. The base frame's $z$ axis runs along the
straight body.

## One segment

With rest length $L_0$, section radius $d$ and local arc fraction $s \in [0, 1]$:

$$
D = \sqrt{D_x^2 + D_y^2 + \varepsilon}, \qquad \theta(s) = s\,D/d .
$$

$R(s)$ rotates by $\theta$ about the axis $(-D_y, D_x, 0)/D$, and

$$
t(s) = \frac{d\,(L_0 + D_l)}{D^2}
\begin{bmatrix} D_x (1 - \cos\theta) \\ D_y (1 - \cos\theta) \\ D \sin\theta \end{bmatrix} .
$$

$t(s)$ is the point at arc length $s (L_0 + D_l)$ on an arc of radius $\rho = d (L_0 + D_l)/D$;
its tangent is the $z$ axis of $R(s)$. The small $\varepsilon$ (default $10^{-12}$ m²) keeps the
straight pose smooth; rotations are then orthonormal up to about $\varepsilon/d^2$.

## The whole body

The global arc parameter $s \in [0, 1]$ is uniform in arc length: segment $i$ spans
$[b_{i-1}, b_i]$ with $b_i = \sum_{j \le i} L_{0,j} / \sum_j L_{0,j}$. A point in segment $i$ is

$$
p(s) = \sum_{j<i} \Big(\prod_{k<j} R_k\Big) t_j + \Big(\prod_{k<i} R_k\Big)\, t_i(s_\text{local}),
\qquad s_\text{local} = \frac{s - b_{i-1}}{b_i - b_{i-1}} .
$$

$s$ may be symbolic (an attachment point to optimize); the segment is then selected with
`casadi.if_else`.

## Tendons

Tendon $j$ of a segment, at angle $\delta_j$ around the section, changes length by

$$
\Delta L_j = D_l - D_x \cos\delta_j - D_y \sin\delta_j ,
$$

and its motor, with spool radius $r$, turns by $\theta_j = -\Delta L_j / r$: positive angles pull.
With three tendons per segment the map is invertible, so motor angles give $\Delta$ exactly, and
the motor torques realizing a generalized force solve $B u = \tau$ with
$B = (\partial\theta/\partial\Delta)^\top$.

## Parameters

| Param | Unit | Scope |
|---|---|---|
| `seg{i}.L0` | m | design |
| `seg{i}.d` | m | design |
| `seg{i}.delta` | rad | design |
| `seg{i}.r` | m | design |
