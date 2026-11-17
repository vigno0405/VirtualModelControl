---
file_format: mystnb
kernelspec:
  name: python3
---

# Friction of the motors

A motor's friction resists its motion: a constant torque against the direction it turns, the
Coulomb friction, and a torque that grows with its speed, the viscous friction. A virtual spring
that pulls with less than the Coulomb torque barely moves the joint, and the robot stops short
of its goal. In this tutorial we model that friction, see what it does to a controller,
compensate it, identify it and plan with it. Everything here is off unless we ask for it.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
from myst_nb import glue
```

## The model

On each motor, turning at the speed $\dot\theta$ [rad/s], friction gives the torque

$$
\tau_f = -F_c \tanh(\dot\theta / \varepsilon) - D \dot\theta ,
$$

with $F_c$ the Coulomb torque [N·m], $D$ the viscous friction [N·m·s/rad], and $\varepsilon$
[rad/s] the speed over which the Coulomb torque turns from one direction to the other. It is
smooth, so a solver can differentiate it, and it only takes energy: $\tau_f \dot\theta \le 0$ at
every speed. It is made of two components the library already has, put on the motor angles
`robot.motor(...)`: a `TanhDamper(motors, F_c / eps, F_c)`, whose slope at rest is
$F_c / \varepsilon$ and which never pushes harder than $F_c$, and a `LinearDamper(motors, D)`.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc

Fc, D = 0.3, 0.05  # Coulomb [N·m], viscous [N·m·s/rad]
rate = np.linspace(-0.1, 0.1, 401)  # motor speed [rad/s]

fig, ax = plt.subplots()
for eps, style in ((0.005, "-"), (0.02, "--")):
    tau = -Fc * np.tanh(rate / eps) - D * rate
    ax.plot(rate, tau, style, label=rf"$\varepsilon$ = {eps}")
ax.set_xlabel(r"motor speed $\dot\theta$ [rad/s]")
ax.set_ylabel(r"friction $\tau_f$ [N$\cdot$m]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
for eps in (0.005, 0.02):
    tau = -Fc * np.tanh(rate / eps) - D * rate
    assert (tau * rate <= 0).all()  # it only takes energy
    coulomb = tau + D * rate
    assert abs(coulomb).max() <= Fc
    assert abs(abs(coulomb[abs(rate) > 4 * eps]) - Fc).max() < 1e-3 * Fc
```

Near rest the Coulomb torque is a steep damper, of slope $F_c / \varepsilon$. A load below
$F_c$ then turns the motor only at the slow speed
$\varepsilon \operatorname{artanh}(\text{load} / F_c)$, so the motor nearly holds: that is its
static friction. A real motor holds exactly, and this model creeps. A smaller $\varepsilon$
creeps slower and is harder for a solver. There is no separate static torque: on the motors we
measured, the torque at which a motor broke away from rest was about its Coulomb torque.

## In a robot

Here is a joint with an inertia, friction on its motor, and a virtual spring of 1 N·m/rad that
pulls it to 1 rad:

```{code-cell} python
from virtualmodelcontrol.models import Direct, JointSpace

def system_of(coulomb=0.0, viscous=0.0, eps=0.005):
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="rad"),
                          actuation=Direct())
    q, motor = robot.joint(0), robot.motor(0)
    robot.add("inertia", vmc.Inertance(q, 0.05))  # [kg·m²]
    if coulomb:
        robot.add("coulomb",
                  vmc.TanhDamper(motor, coulomb / eps, coulomb))
    robot.add("viscous", vmc.LinearDamper(motor, viscous))
    ctrl = vmc.Mechanism("ctrl")
    goal = vmc.Ref("goal", 1, [1.0])  # [rad]
    ctrl.add("spring", vmc.LinearSpring(q - goal, 1.0))
    ctrl.add("damper", vmc.LinearDamper(q, 0.2))
    return vmc.VirtualMechanismSystem(robot, ctrl)

def run(system, output=(), T=20.0):
    plant = vmc.sim.ModelPlant(system.robot)
    controller = vmc.VMCController(vmc.compile(system),
                                   output=list(output))
    clock = vmc.sim.SimClock(dt=1 / 100)
    return vmc.sim.run(plant, controller, clock, T=T).arrays()
```

We run it without friction, and with the friction above:

```{code-cell} python
free, held = run(system_of()), run(system_of(Fc, D))

fig, ax = plt.subplots()
ax.plot(free["t"], free["q"][:, 0], label="no friction")
ax.plot(held["t"], held["q"][:, 0], "--", label="friction")
ax.set_xlabel("time [s]")
ax.set_ylabel("angle [rad]")
ax.legend(loc="lower right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
t = np.ravel(held["t"])
early = 1.0 - held["q"][np.searchsorted(t, 2.0), 0]
short = 1.0 - held["q"][-1, 0]
creep = float(held["v"][-1, 0])
assert abs(1.0 - free["q"][-1, 0]) < 1e-3
assert 0.15 < early < Fc and 0.1 < short < early and 0 < creep < 0.01
glue("early", float(early), display=False)
glue("short", float(short), display=False)
glue("creep", creep, display=False)
```

The joint without friction arrives. With friction, it stops {glue:text}`early:.2f` rad short
within two seconds: the spring pulls with $K e$, which is less than $F_c$ while
$e < F_c / K$. Then it only creeps, at {glue:text}`creep:.3f` rad/s after 20 s, when it is
still {glue:text}`short:.2f` rad short. A stiffer spring narrows the gap.

## The finger's motors

`adapt.add_dynamics` gives the finger its gravity and, when we ask, its motors' friction and
rotor inertia, with the values fitted to step and ramp runs of the real finger held by a joint
spring:

```{code-cell} python
from virtualmodelcontrol.robots import adapt

finger = adapt.add_dynamics(adapt.finger(),
                            coulomb=adapt.MOTOR_FRICTION,
                            damping=adapt.MOTOR_DAMPING,
                            inertia=adapt.ROTOR_INERTIA)
finger.params["coulomb.max_force"].value
```

```{code-cell} python
:tags: [remove-cell]
glue("mcp", adapt.MOTOR_FRICTION[0], display=False)
glue("pip", adapt.MOTOR_FRICTION[1], display=False)
glue("eps", adapt.FRICTION_SMOOTHING, display=False)
found = finger.params["coulomb.max_force"].value
assert found.tolist() == list(adapt.MOTOR_FRICTION)
```

The Coulomb torques are {glue:text}`mcp:.3f` N·m on the MCP motor and {glue:text}`pip:.3f` N·m
on the PIP motor, turning over {glue:text}`eps:.3f` rad/s. With them the simulated runs follow
the measured ones closely, and without the Coulomb term no choice of damping and inertia does.
Two things the model leaves out: the real motors stick where the model creeps, and at slow
speeds the MCP motor moves in jerks (stick-slip), which a friction that does not drop with speed
cannot produce. The efficiency of the motors is a separate thing, in their transmission
([Efficiency](../concepts/efficiency.md)); the fit used the default, 1.

## Compensation

`FrictionCompensation(max_torque, velocity, width=0)` adds a torque $\hat F$ in the direction
of the command while the motor is at rest, and fades it once the motor turns:

$$
u + \hat F e^{-(\dot\theta / v)^2} \sigma(u),
$$

with $\sigma$ the sign of $u$, or with a `width` $w$ the smooth sign $u / \sqrt{u^2 + w^2}$. It
helps a motor break away, and it leaves the friction of a moving motor alone. It is an output
stage: a correction that `VMCController(output=...)` applies to the command after the
controller's law ([the loop](first-controller.md#the-loop)). It adds energy, so nothing turns it
on but us:

```{code-cell} python
from virtualmodelcontrol.control import FrictionCompensation

stages = {f"{share:.0%}": FrictionCompensation(share * Fc, 0.03, width=0.01)
          for share in (0.5, 0.8)}
errors = {name: 1.0 - run(system_of(Fc, D), [stage])["q"][-1, 0]
          for name, stage in stages.items()}
errors
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < errors["80%"] < errors["50%"] < 0.5 * short
glue("half", float(errors["50%"]), display=False)
glue("most", float(errors["80%"]), display=False)
```

Compensating half of the Coulomb torque leaves {glue:text}`half:.3f` rad after 20 s, and 80 %
leaves {glue:text}`most:.3f` rad. How to choose the three numbers:

- **$\hat F$**, from the friction: a fraction of $F_c$, fitted as below. On a robot, more than
  $F_c$ makes a motor jitter around its goal, because the speed it reads is coarse and late. A
  simulation reads the speed exactly, so it does not show that.
- **$v$**, from the speed reading: one or two of its steps, so that a motor at rest reads as at
  rest.
- **$w$**, a choice: commands smaller than $w$ get only part of $\hat F$, so a larger width
  jitters less, responds less, and is easier for a solver. A plan needs one.

## Identification

The friction torques are Params of the robot, so `fit_params` finds them from a run with its
motor torques ([Fit Params to a run](fit.md)). Their names say where they sit:
`coulomb.max_force` is $F_c$, `coulomb.damping` the slope $F_c / \varepsilon$ and
`viscous.damping` is $D$. We drive the joint, held by a spring, with a slow sine of torque, and
fit from a rough guess:

```{code-cell} python
from virtualmodelcontrol.identification import fit_params

dt = 1 / 330
t = np.arange(0.0, 6.0, dt)
command = 0.6 * np.sin(np.pi * t)[:, None]  # [N·m]

def bench(coulomb, viscous):
    robot = system_of(coulomb, viscous).robot
    robot.add("spring", vmc.LinearSpring(robot.joint(0), 2.0))
    return robot

none = vmc.Mechanism("none")  # no controller: the torques are given
system = vmc.VirtualMechanismSystem(bench(Fc, D), none)
log = vmc.sim.rollout(system, [0.0], len(t) * dt, dt, max_step=dt,
                      u=command)

names = ["coulomb.max_force", "coulomb.damping", "viscous.damping"]
fit = fit_params(bench(0.2, 0.02), names, [log], smoothing=11)
{name: fit.values[name].round(3) for name in names}
```

```{code-cell} python
:tags: [remove-cell]
found = {name: float(np.ravel(fit.values[name])[0]) for name in names}
assert abs(found[names[0]] - Fc) < 0.01 and abs(found[names[2]] - D) < 0.01
assert abs(found[names[0]] / found[names[1]] - 0.005) < 5e-4
glue("Fc_fit", found[names[0]], display=False)
glue("D_fit", found[names[2]], display=False)
```

The fit gives {glue:text}`Fc_fit:.3f` N·m for the Coulomb torque and {glue:text}`D_fit:.3f`
N·m·s/rad for the viscous friction, against 0.3 and 0.05 in the simulated robot. The fit is
local: it starts from the robot's values. Without the motor torques, as for the finger, the
friction can still be fitted to runs of the closed loop by simulating them, which is slower.
`fit_stiffness_damping(..., friction=True)` fits a Coulomb torque per motor together with the
stiffness and the damping, from torque steps ([Hanging soft arm](../examples/hanging-arm.md)).

## Planning with it

The friction is in the robot's dynamics, so a plan ([Optimize](optimize.md)), an optimization
and an MPC ([Model predictive control](mpc.md)) see it. A compensation with a width joins them
through `Problem(output=...)`, which applies it to the controller's command as
`VMCController(output=...)` does in a simulation:

```{code-cell} python
from virtualmodelcontrol import optimization as opt

stage = FrictionCompensation(0.8 * Fc, 0.03, width=0.02)
system = system_of(Fc, D, eps=0.02)  # wider: easier for a solver

problem = opt.Problem(system, output=[stage])
problem.add(opt.Shooting([0.0], 3.0, 31, v0=[0.0], substeps=5))
plan = problem.solve()
sim = vmc.sim.rollout(system, [0.0], 3.0, 0.02, v0=[0.0], max_step=0.02,
                      output=[stage])
gap = abs(plan.q[:-1] - sim["q"][::5]).max()
```

```{code-cell} python
:tags: [remove-cell]
assert plan.converged and gap < 1e-8
assert np.isclose(3.0 / (len(plan.q) - 1), 5 * 0.02)  # a point every fifth step
glue("gap", float(gap), display=False)
```

The plan follows the simulation within {glue:text}`gap:.0e` rad: it is the same closed loop,
with the motor's friction and the compensation in it. The plan has a point every 0.1 s, which is
every fifth step of the simulation. `MPC(problem)` takes the same problem, so it plans with
friction too.

## Good to know

- **Off by default.** `adapt.add_dynamics` adds friction only when given `coulomb=`, and no
  template turns a compensation on. `adapt.output_stage()` is the compensation the lab runs on
  the finger, `adapt.COMPENSATION`.
- **The smoothing** $\varepsilon$: smaller is closer to a motor that sticks and harder for a
  solver. A plan may want a wider one than a simulation.
- **Deprecated in 1.6.0, removed in 2.0.0:** `StaticFriction`, `Efficiency(friction=)`,
  `fit_efficiency(..., friction=True)`, `StaticFrictionCompensation`, `adapt.finger_friction()`,
  `adapt.hand_friction()`, and the names `adapt.FRICTION` and `adapt.HAND_FRICTION` (now
  `adapt.COMPENSATION` and `adapt.HAND_COMPENSATION`). The old model took the friction from the
  command, without the speed. To move over, take its breakaway torque as $F_c$ on the motors,
  and replace `StaticFrictionCompensation` by `FrictionCompensation` with a `width`.
