---
file_format: mystnb
kernelspec:
  name: python3
---

# Robots with fewer motors than joints

Some joints have no motor: a finger's last joint follows a spring, a soft arm bends in ways its
tendons do not pull. The motors cannot give every torque a virtual mechanism asks for. This page
explains what they give, the two controllers that follow, and why one of them cannot gain energy.
[Robots with fewer motors than joints](../tutorials/underactuated.md) runs all of it.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## What the motors give

The motors deliver $\tau = B u$: $u$ are the motor torques and $B$ the matrix that maps them to the
joints. With fewer motors than joints no $u$ solves $B u = \tau$ for every wanted torque $\tau$.
The best the motors do is the least-squares answer

$$
u = B^+ \tau , \qquad B^+ = (B^\top B)^{-1} B^\top ,
$$

which is the closest torque $B u$ to $\tau$. What is left, $E\tau$ with $E = I - B B^+$, is the
torque defect. $E$ projects on the passive directions, the ones no motor reaches.

## Two controllers

Both controllers render the same virtual mechanism: its springs, dampers and sources give a wanted
torque $\tau$ at some configuration of the robot. They differ in the configuration they evaluate it
at.

- **Naive.** It evaluates $\tau$ at the measured state $(q, v)$, passive joints included, and sends
  $u = B^+\tau(q, v)$. It is the best projection in torque space and the easiest to use, but it
  needs the passive joints' state. Its torque is not the gradient of anything: moving a passive
  joint changes the torque with nobody paying for it. No energy argument covers it. The
  [passive and tank corrections](../tutorials/underactuated.md) repair that.
- **Frozen.** It ignores the passive joints. The motors give the actuated part of the
  configuration; the passive part is set to a *rest* $\bar q$. It evaluates $\tau$ at that frozen
  configuration $\pi(\theta)$, so it needs the motor angles $\theta$ only. It is passive, as
  shown below.

The frozen controller has two options for the rest, and only that changes:

| rest | the passive joints are taken at | needs |
|---|---|---|
| constant | a fixed $\bar q$: the `q_rest` Param, or the balance at the motor angles `at` | nothing else |
| moving | their static balance at the current motor angles | the robot's model |

## Why the frozen controller is passive

Write $\pi(\theta)$ for the frozen configuration and $P = \partial\pi/\partial\theta$ for how it
moves with the motors. The motors' velocity makes the frozen configuration move at $P\dot\theta$. The
controller evaluates the virtual mechanism there and pulls the torque back through $P$:

$$
u = P^\top \tau\big(\pi(\theta),\, P\dot\theta\big) .
$$

This is the principle of virtual work: a motion of the motors moves the frozen configuration by
$P\,\mathrm{d}\theta$, so the work of $\tau$ is $\tau^\top P\,\mathrm{d}\theta$ and the motor
torque is $P^\top\tau$. Two facts follow.

1. **Springs.** The spring part of $\tau$ is $-\nabla V(\pi)$, with $V$ the virtual potential.
   Then

   $$
   u = -P^\top\nabla V(\pi) = -\nabla_\theta\, V(\pi(\theta)) ,
   $$

   the gradient of a function of the motor angles alone. The motors only store and give back the
   energy $V(\pi(\theta))$. They cannot gain energy by going around a loop.
2. **Dampers.** A damper has a dissipation function $R$, the
   [Rayleigh function](passivity.md#dampers-and-the-rayleigh-function). Here
   $R = \tfrac12\,\dot d^\top C\,\dot d$, with $\dot d = J_\pi P\dot\theta$ the task velocity of the
   frozen configuration and $C$ the damping. The pulled-back damper is

   $$
   u_d = -\frac{\partial R}{\partial\dot\theta} = -P^\top J_\pi^\top C J_\pi P\dot\theta ,
   $$

   so the power the motors give it is $\dot\theta^\top u_d = -\dot d^\top C\,\dot d \le 0$.

Together, the controller stores a function of $\theta$ and dissipates. With a robot that is passive
too, the total energy never rises. None of this uses what $\pi$ is, only that it is a smooth
function of $\theta$. So it holds for a constant rest, for a moving one, and when the model that
places the rest is wrong.

## The constant rest

If the rest $\bar q$ is a constant, the frozen configuration is the motors' part plus a fixed
offset, so $P = (B^+)^\top$ and $P^\top\tau = B^+\tau$. The frozen controller is the naive formula
at $\pi(\theta)$: the same least-squares torque, evaluated at a configuration the motors fix. This
is the simplest passive controller, and it needs no model.

Its price is that the passive joints are held at $\bar q$ while the real ones are wherever the
robot lets them be, a weight or a spring pushes them away. At an equilibrium the two differ by
$E(q - \bar q)$, and the controller's equilibrium differs from the naive one by a term
proportional to that gap. At the configuration where the passive joints really sit at $\bar q$
the two controllers agree. Choose $\bar q$ well and the gap vanishes where you work:

- **Balance it.** The passive joints at rest satisfy $E\nabla V_r = 0$, with $V_r$ the springs and
  gravity of the robot. `at=` solves that once at the motor angles of a pose and keeps it as the
  rest.
- **Optimize it.** `q_rest` is a Param, so a plan can free it and find the rest that puts the
  arm's equilibrium on the goal.

## The moving rest

Let the rest follow the motors instead: the passive joints sit at their static balance
$E\nabla V_r(\pi(\theta)) = 0$ for the current $\theta$. There is then no pose to choose, and the
frozen configuration is the one the real robot takes at rest. The balance depends on $\theta$, so
$P$ gets a second term from the chain rule (the implicit function theorem):

$$
P = (B^+)^\top - e\,(e^\top H e)^{-1} e^\top H\,(B^+)^\top ,
$$

with $e$ a basis of the passive directions and $H = \nabla^2 V_r$ the stiffness of the robot. The
library solves the balance with Newton steps written in CasADi, so it can differentiate it and a
plan can include it: eight steps from the constant rest, each at most 0.5 rad long, which is
plenty for the passive joints of the robot templates. `ua.Moving(actuation, dynamics,
iterations=, step=)` builds the read with other values, for a `StateController(compiled,
read=...)`; a plan and a rollout use the default ones. The balance makes every step of a plan heavier: a
shooting plan of the three-link arm takes about twice the memory and time to build with the
moving rest, one of the five-link arm (three passive joints) about three times. The passivity above holds for any number of steps and any error in
$V_r$; a good model only makes the reach more exact. Everything the robot's model holds at rest
is in $V_r$: its springs, its weight, and any constant load it carries, a payload or a pretension
for instance. A load that is not in the model, or one that changes with time (which acts like a
parameter changed while running, and wants a [tank](../tutorials/energy.md)), is not.

## Gravity

Weight compensation is a source: it pushes each mass with $-mg$, so its torque is $\nabla V_g$,
the gradient of the gravitational potential.

- With a **constant rest**,

  $$
  B^+\nabla V_g(\pi(\theta)) = \nabla_\theta\,V_g(\pi(\theta)) :
  $$

  the gradient of the gravitational potential *projected* on the frozen configuration. It is as passive
  as a spring, and it cancels the weight on the motors exactly when the passive joints really are
  at $\bar q$.
- With a **moving rest**, the compensation is rendered at the balance with $B^+$, without the pull-back.
  There

  $$
  B^+\nabla V_g(\pi) = B^+\nabla V_r(\pi) = \nabla_\theta V_r(\pi(\theta))
  $$

  is the gradient of the robot's own potential on its balance manifold, so it cancels the load on the motors at every
  pose, the passive joint's spring included. Pulling it back through $P$ as well would leave the
  passive spring's own pull uncompensated, and the tip would end a few millimetres short. Sources
  are the one part of a virtual mechanism not covered by the passivity argument, as for any
  controller; this one is the tame one that cancels what the robot carries.

## Which to take

| | reads | needs the passive joints' state | passive | a pose to choose |
|---|---|---|---|---|
| naive | $q$, $v$ | yes | no, repair it with a correction | no |
| frozen, constant rest | motors | no | yes | the rest $\bar q$ (or let a plan find it) |
| frozen, moving rest | motors and the robot's model | no | yes | no |

Take the **naive** controller when the passive joints are measured, or an estimator provides them,
and a correction is acceptable. Take the **frozen** one when only the motors are known, or when a
guarantee matters. Of its two rests, the constant one needs no model; the moving one reaches the goal wherever the robot can, and costs the model and a few
Newton steps. All three are available in a plan, `Problem(system, reads="state")`,
`Problem(system)` and `Problem(system, rest="moving")`, with every other option of the library.
