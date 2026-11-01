---
file_format: mystnb
kernelspec:
  name: python3
---

# Passivity

A controller built from springs and dampers can only give the robot back the energy it stored.
This page states that property, shows how it makes the robot and its controller stable
together, and lists what weakens it in practice. [Energy and passivity](../tutorials/energy.md)
checks the same balance numerically on a run.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
:tags: [remove-input]
from schematics import energy_flow
energy_flow.figure();
```

## The controller's energy balance

The controller stores $E_c = V + T$: $V$ in its springs, measured from its lowest value so that
$E_c \ge 0$, and $T$ in the inertances of its virtual states. Three powers change it. The port, $P = \tau^\top v$, is the power its torques give the
robot. The dissipation power is $P_D = \sum f \cdot \dot y \le 0$ over its dampers. The source power is
$P_S = \sum f \cdot \dot y$ over its sources. With the [VMC equations](vmc.md),

$$
\dot E_c = -P + P_D + P_S .
$$

The compiled controller computes each term (`compiled.energy`, `compiled.power`), so the balance can be
checked on any run.

## Passive controllers

Without sources, integrating the balance gives

$$
\int_0^t P\,\mathrm{d}t = E_c(0) - E_c(t) + \int_0^t P_D\,\mathrm{d}t \;\le\; E_c(0),
$$

since $E_c(t) \ge 0$ and $P_D \le 0$. Whatever the robot does, the controller gives it at most the
energy it stored at the start: it is passive.

## The robot and its controller together

A robot made of masses, springs, dampers and gravity is passive too: its energy $E_r$ changes
by its input power and its own dissipation. With the default [efficiency](efficiency.md) of 1
and no output stage, its input power is the controller's port $P$, so the two balances add up:

$$
\dot E_r + \dot E_c = P_{D,r} + P_D + P_S .
$$

Without sources the total energy can only fall. It is bounded by its value at the start. Where
it has a minimum, at the equilibrium the springs set, it works as a Lyapunov function: the
robot settles there or stays near it.

Gravity compensation is a source, but its forces are exactly those that cancel the weight of
the robot's masses: its power is minus the power of gravity on them. Added to the balance, it
removes gravity's potential energy from the robot's. The robot then behaves as if it had no
weight.

## Dampers and the Rayleigh function

A damper gives a force $f(y, \dot y)$ on its coordinate with $f \cdot \dot y \le 0$. Every damper of
the library is the gradient of a *Rayleigh dissipation function* $R(\dot y)$:

$$
f = -\frac{\partial R}{\partial \dot y}, \qquad \dot y \cdot \frac{\partial R}{\partial \dot y} \ge 0 .
$$

For a linear damper $R = \tfrac12\,\dot y^\top D\,\dot y$ and $f = -D\dot y$. The tanh damper has
$R = (F^2/d)\ln\cosh(d\dot y/F)$, and the diode, contact and friction dampers are of the same kind.
The robot receives $\tau = J^\top f$. Since $\dot y = J v$, that is also
$\tau = -\partial R/\partial v$: the dissipation function of the whole robot is the sum of its
dampers', and the damping matrix $C = J^\top D J$ is symmetric and positive semidefinite. The power
the dampers take is $f\cdot\dot y = -\dot y^\top \partial R/\partial \dot y$, which is $-2R$ for a
linear damper and never positive for any of them. That is the $P_D \le 0$ of the balance above.

The library never builds $R$. It builds the force $f$ and the power $f \cdot \dot y$ directly, which
are the same numbers. A test checks it for every damper it ships: the Jacobian
$\partial f/\partial \dot y$ is symmetric (the force is the gradient of something), and negative
semidefinite for all but the smooth diode, whose $R$ is slightly bent where its gate opens and
which dissipates all the same. A damper of your own is a Rayleigh damper when that Jacobian is
symmetric; for passivity it is enough that $f \cdot \dot y \le 0$.

The same function explains the damping of a controller on a robot with fewer motors than joints.
Evaluated on the frozen velocity, $R = \tfrac12\,\dot d^\top C\,\dot d$ gives a torque on the
motors that always dissipates ([the theory](underactuated.md)).

## What weakens it

- **Sampling and delay.** A digital controller holds each torque for a control period, so it
  reacts to the past. Stiff springs and strong dampers then inject energy. Above a point the
  loop goes unstable even though every element is passive. [Tuning](../tutorials/tuning.md)
  shows the limit on damping.
- **Changing parameters while running.** Raising a stiffness raises the energy stored in that
  spring at once; `controller.set` returns that jump ([Parameters](../tutorials/parameters.md)),
  and a `Tank` pays for it ([Energy and passivity](../tutorials/energy.md)).
  Swapping elements through a smooth blend (`vmc.control.SwapController`) keeps the torques
  continuous.
- **Sources.** A force source can supply any energy; only gravity compensation is tame.
- **Efficiency.** One constant efficiency $\eta$ on every motor scales the robot's input to
  $\eta P$, and the balance holds with $\eta E_c$. Different efficiencies on motors that one
  element couples make its forces non-conservative, as the [finger example](../examples/finger.md)
  explains.
- **Joints that no motor drives.** Torques that the motors alone can give, $B^+\tau$ at the
  measured state, are not the gradient of a potential, so the controller may not be passive any
  more. The frozen controller of [robots with fewer motors than joints](underactuated.md) is
  passive by construction, and [the tutorial](../tutorials/underactuated.md) runs it.
- **Output stages.** Friction compensation, pretension and torque limits change the torques the
  motors receive, so the robot's input is no longer the controller's port.
