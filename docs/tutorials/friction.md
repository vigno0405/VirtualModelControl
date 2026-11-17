---
file_format: mystnb
kernelspec:
  name: python3
---

# Friction compensation

A motor's friction holds it until the command passes a breakaway torque, and then takes part of
the torque while it moves. A virtual spring that pulls with less than that does not move the joint,
and the robot stops short of its goal. The library has no model of this friction. On the robot an
output stage compensates it: it adds a torque in the direction of the command while the motor is
at rest. This tutorial shows the stage, says why there is no model, and how to choose its numbers.
Nothing here is on unless we ask for it.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
from myst_nb import glue
```

## The stage

`FrictionCompensation(max_torque, velocity, width=0)` turns the command $u$ of each motor into

$$
u + \hat F \, e^{-(\dot\theta / v)^2} \, \sigma(u),
$$

with $\hat F$ the `max_torque` [N·m], $\dot\theta$ the motor's speed and $v$ the `velocity`
[rad/s] past which the stage fades. $\sigma$ is the sign of $u$, or with a `width` $w$ [N·m] the
smooth sign $u / \sqrt{u^2 + w^2}$. Here is what it sends for each command, at rest and moving:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import FrictionCompensation

F_hat, v, w = 0.03, 0.03, 0.01  # [N·m], [rad/s], [N·m]
u = np.linspace(-0.06, 0.06, 601)  # commands of one motor [N·m]

def sent(stage, rate):
    meas = vmc.Signals(0.0, motor_position=np.zeros_like(u),
                       motor_velocity=np.full_like(u, rate))
    return stage(u, meas)

sign = FrictionCompensation(F_hat, v)
smooth = FrictionCompensation(F_hat, v, width=w)

fig, ax = plt.subplots()
ax.plot(u, sent(sign, 0.0), label="sign, at rest")
ax.plot(u, sent(smooth, 0.0), "--", label="smooth, at rest")
ax.plot(u, sent(sign, 3 * v), ":", label=r"moving at $3v$")
ax.set_xlabel(r"command $u$ [N$\cdot$m]")
ax.set_ylabel(r"sent [N$\cdot$m]")
ax.legend(loc="upper left", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
at_rest = sent(sign, 0.0) - u
assert np.allclose(at_rest[u > 1e-9], F_hat) and np.allclose(at_rest[u < -1e-9], -F_hat)
assert (abs(sent(smooth, 0.0) - u) < F_hat).all()  # the smooth sign: never all of it
left = float(abs(sent(sign, 3 * v) - u).max())
assert np.isclose(left, F_hat * np.exp(-9))
glue("F_hat", F_hat, display=False)
glue("w", w, display=False)
glue("left", left, display=False)
```

At rest the sign adds all of $\hat F$ = {glue:text}`F_hat:.2f` N·m to any command, however small:
the output jumps at $u = 0$. The smooth sign adds only part of it to commands smaller than
$w$ = {glue:text}`w:.2f` N·m, so a command that changes sign near a goal does not flip the
motor between $\pm\hat F$. Once the motor turns at three times $v$, the stage adds less than
{glue:text}`left:.0e` N·m: it helps a motor break away and leaves a moving one alone. It is an
output stage, a correction that `VMCController(output=...)` applies to the command after the
controller's law ([the loop](first-controller.md#the-loop)). It adds energy, so nothing turns it
on but us.

## Why there is no friction model

In a robot's runs friction and damping look alike: both take torque against the motion. A fit
tells them apart only with runs over many speeds, slow ones and the start of a motion, and the
runs of an identification rarely have them. So the library models neither apart: what the robot
loses to friction ends up in the damping and the stiffness identified from its runs
([Fit Params to a run](fit.md)), and a simulation or a plan has no friction. Nor does a plan
take the compensation, which would push against a friction the plan does not have:

```{code-cell} python
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import Direct, JointSpace

robot = vmc.Mechanism("robot", model=JointSpace(1, unit="rad"),
                      actuation=Direct())
q = robot.joint(0)
robot.add("inertia", vmc.Inertance(q, 1.0))  # [kg·m²]
ctrl = vmc.Mechanism("ctrl")
ctrl.add("spring", vmc.LinearSpring(q - vmc.Ref("goal", 1, [1.0]), 1.0))
system = vmc.VirtualMechanismSystem(robot, ctrl)
try:
    opt.Problem(system, output=[smooth])
except ValueError as error:
    print(error)
```

```{code-cell} python
:tags: [remove-cell]
try:
    opt.Problem(system, output=[smooth])
    raise AssertionError("a plan took the compensation")
except ValueError as error:
    assert "FrictionCompensation has no symbolic form" in str(error)
```

Friction only takes energy, so a controller that is passive on the model stays passive on the
robot. What we give up is a simulation that stops short of the goal as the robot does.

## Choosing the numbers

- **$\hat F$, under the breakaway torque.** Measure the breakaway $F_b$ of each motor on the
  robot: from rest, raise its command slowly and note the torque at the first motion of the
  encoder, both ways, and keep the smaller; do it at a pose with little static load. Start with
  $\hat F = 0.6\,F_b$. With more than $F_b$ a motor jitters around its goal: the speed it reads
  is coarse and late, so the stage keeps pushing a motor that has already broken away. If the
  robot still stops short and does not jitter, raise it toward $0.8\,F_b$.
- **$v$, from the speed reading:** one or two of its steps, so that a motor at rest reads as at
  rest. A Dynamixel reads speed in steps of 0.229 rpm, which is {glue:text}`step:.3f` rad/s.
- **$w$, from the error we accept:** about the stiffness $K$ of the virtual spring at the motor
  times the position error we accept. A spring of 0.6 N·m/rad and 1° give
  {glue:text}`w_example:.3f` N·m. A larger width jitters less and leaves a slightly larger error.

```{code-cell} python
:tags: [remove-cell]
step = 0.229 * 2 * np.pi / 60  # [rad/s]
glue("step", float(step), display=False)
glue("w_example", float(0.6 * np.radians(1.0)), display=False)
rate = np.linspace(0.0, 5 * v, 10001)
power = F_hat * rate * np.exp(-((rate / v) ** 2))  # what the stage adds at each speed [W]
bound = F_hat * v / np.sqrt(2 * np.e)
assert abs(power.max() - bound) < 1e-9
glue("power_mW", float(1e3 * bound), display=False)
```

The energy it adds is small: at most $\hat F v / \sqrt{2e}$ per motor, the most of
$\hat F |\dot\theta| e^{-(\dot\theta/v)^2}$ over the speed, which is {glue:text}`power_mW:.2f` mW
with the numbers above.

## On the robot

The finger and the hand have the stage in their output stages, `adapt.output_stage()` and
`adapt.hand_output_stage()`, with their values in `adapt.COMPENSATION` and
`adapt.HAND_COMPENSATION`; a torque clip follows it. A controller takes them with `output=`:

```{code-cell} python
from virtualmodelcontrol.robots import adapt

finger = adapt.finger()
ctrl = vmc.Mechanism("ctrl")
goal = vmc.Ref("theta_ref", 3, [0.4, 0.4, 0.4])  # [rad]
angles = adapt.joint_angles(finger)
ctrl.add("hold", vmc.LinearSpring(angles - goal, 0.6 * np.eye(3)))
compiled = vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl))
controller = vmc.VMCController(compiled, output=adapt.output_stage())

meas = vmc.Signals(0.0, motor_position=np.zeros(2),
                   motor_velocity=np.zeros(2))
controller.reset(0.0, meas)
command = controller.step(0.0, meas)
command["law_torque"], command["motor_torque"]
```

```{code-cell} python
:tags: [remove-cell]
law, motor = (np.asarray(command[k], dtype=float) for k in ("law_torque", "motor_torque"))
F_finger = adapt.COMPENSATION[0]
assert np.allclose(motor - law, F_finger * np.sign(law))
glue("F_finger", float(F_finger), display=False)
glue("F_hand", float(adapt.HAND_COMPENSATION[0]), display=False)
```

At rest each motor gets {glue:text}`F_finger:.2f` N·m more than the law asks, in the direction
of the law. The hand's stage adds {glue:text}`F_hand:.1f` N·m. On a robot, read `motor_torque`
and send it to the motors.

## Before 1.6.0

The library had a model of the motors' static friction, a function of the command alone:
`StaticFriction`, in `Efficiency(friction=)`, with `fit_efficiency(..., friction=True)`,
`StaticFrictionCompensation`, `adapt.finger_friction()` and `adapt.hand_friction()`, and a
Coulomb column in `fit_stiffness_damping(..., friction=True)`. Version 1.6.0 deprecates them
with a warning and 2.0.0 removes them; `adapt.FRICTION` and `adapt.HAND_FRICTION` are now
`adapt.COMPENSATION` and `adapt.HAND_COMPENSATION`. To move over, drop the friction model, fit
the damping without it, and compensate on the robot with `FrictionCompensation`; a
`StaticFrictionCompensation` becomes a `FrictionCompensation` with a `width`.
