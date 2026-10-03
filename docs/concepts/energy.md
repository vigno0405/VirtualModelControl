# Mechanisms, energy and the control law

Virtual Model Control replaces abstract gains by physical elements placed on the robot:
springs, dampers and masses attached to points of its body. A **mechanism** is a set of
coordinates plus components acting on them.

## Coordinates

A coordinate is a smooth function $y(q, z, p, t)$ of the robot configuration $q$, the
controller's own states $z$, the parameters $p$ and time. Examples: a point at arc parameter
$s$, a joint angle, a point minus its goal, the projection of a vector on a direction. Its rate
follows by automatic differentiation,

$$
\dot y = J_q(q)\,G(q)\,v + J_z\,\dot z + \partial_t y ,
$$

where $G(q)$ maps the velocity $v$ to $\dot q$ (the identity on flat spaces).

## Components

| Kind | Defined by | Force |
|---|---|---|
| storage | energy $V(y) \ge 0$ | $f = -\partial V / \partial y$ |
| dissipation | $f(y, \dot y)$ | $f \cdot \dot y \le 0$ |
| inertance | inertia $M(y)$ | kinetic energy $\tfrac12 \dot y^\top M \dot y$ |
| source | $f$ (a Param) | power $f \cdot \dot y$ is metered |

The deflection of a spring is $y = x - x_\text{ref}$, so it pulls $x$ towards $x_\text{ref}$.

## The law

Each component's force acts on the robot through its coordinate's Jacobian. The controller's
generalized force and the virtual dynamics are

$$
\tau = \sum_k (J_{q,k} G)^\top f_k, \qquad
M_z(z)\,\ddot z + c_z(z, \dot z) = \sum_k J_{z,k}^\top f_k ,
$$

with $M_z = \sum J_z^\top M_k J_z$ from the controller's inertances and $c_z$ the Coriolis terms
of $T = \tfrac12 \dot z^\top M_z \dot z$. The actuator command solves $B(q)\,u = \tau$; for
tendons, $B = (\partial\theta/\partial q)^\top$.

## Energy balance

With $E = \sum V_k + T$ the controller's energy, the compiled functions satisfy exactly

$$
\dot E = -\tau^\top v + P_\text{diss} + P_\text{source}, \qquad P_\text{diss} \le 0 .
$$

Without sources, the controller can only return energy it has stored: it is passive with
respect to the port $(v, -\tau)$, and so is the closed loop with a passive robot. Changing a
stiffness or a goal while running changes $E$ by a jump that `VMCController.set` returns, so
an energy tank can bound it.

## Saturation along a direction

A tanh spring on a `Projection` acts on the scalar $y = \hat n^\top c$, so the force on the point,
$\hat n\, f$, never exceeds the maximum force whatever the direction $\hat n$.
