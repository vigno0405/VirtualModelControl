---
file_format: mystnb
kernelspec:
  name: python3
---

# Track a contact force

In this tutorial the fingertip presses a table with a chosen force. We do not work out how deep
the goal must be. The controller moves its own goal until the measured force is the one we want.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The law

A spring pulls the fingertip to a goal, and the table pushes back. The force on the table is a
function of the goal, so we can ask which way to move the goal to bring the force closer to the
wanted one. `ForceTracking` answers it. It moves live Params of the controller, such as the
spring's goal or its stiffness, by gradient descent of
$\tfrac12 \lVert f - f_\mathrm{des} \rVert^2$. A Param is live when it can change while the
controller runs ([Parameters](parameters.md)). At every control step the law moves them by the
largest step that changes the force by less than `max_force_step` [N], 0.01 N by default.

The force $f$ is the one the motors push through the contact point, a vector of three entries.
The law needs the measured force too, in the same form. On a real robot it comes from a load
cell, or from [an estimate](force-estimates.md). Here it comes from the simulated table first.

## Press with a chosen force

The table is the one of the [contact tutorial](contact.md), a plane 6 cm below the finger's
base. The finger's $z$ axis points down, so the table's outward normal is $-z$, and the finger
pushes on it along $+z$.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.robots import adapt

k = 1e4  # [N/m], a hard table
finger = adapt.add_dynamics(adapt.finger())
tip = finger.point("tip")
gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
finger.add("table", vmc.ContactSpring(gap, k))
finger.add("cushion", vmc.ContactDamper(gap, 5.0))  # [N·s/m]

goal = vmc.Ref("goal", 3, value=[0.0, 0.05, 0.06])  # starts on the table
ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(tip - goal, 100.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
ctrl.add("limits", adapt.joint_limit_spring(finger))
ctrl.add("gravity", vmc.GravityCompensation(finger))
system = vmc.VirtualMechanismSystem(finger, ctrl)
compiled = vmc.compile(system)
```

The loop is the usual one. After the controller's step it reads the table's force, as a load cell
would, and gives it to the law, which changes the Param. The wanted force is 2 N along $+z$, into
the table, and `normal=[0, 0, 1]` tells the law that the force it tracks is the one along $z$.
`adapted` names the Param and `act` replaces the law, in the sections below. The law can also be
told an estimate in place of the table's force: see [Estimate a contact force](force-estimates.md).

```{code-cell} python
wanted = np.array([0.0, 0.0, 2.0])  # [N]
kin = vmc.Kinematics(finger)

def press(controller, adapted="ctrl.press.goal", act=None, steps=3000):
    if act is None:
        law = ForceTracking(controller, "tip", adapted, normal=[0, 0, 1])
        act = law.step
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    names = ["t", "force", "depth", "stiffness", "level"]
    log = {name: [] for name in names}
    for step in range(steps):  # 500 Hz
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - 0.06  # [m]
        table = np.array([0.0, 0.0, k * max(0.0, below)])
        if step > 100:  # let the finger settle on the table first
            act(controller, table, wanted)
        live = controller.live_params()
        level = getattr(controller, "level", 0.0)  # a tank's budget [J]
        depth = live["ctrl.press.goal"][2] - 0.06  # [m]
        stiffness = float(live["ctrl.press.stiffness"])  # [N/m]
        row = (plant.t, table[2], depth, stiffness, level)
        for name, value in zip(names, row):
            log[name].append(value)
        plant.advance(1 / 500)
    return {name: np.array(values) for name, values in log.items()}

free = press(vmc.VMCController(compiled))
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
top.plot(free["t"], free["force"], label="contact force")
top.axhline(wanted[2], color="0.5", linestyle="--", label="wanted")
top.set_ylabel("force [N]")
top.legend(loc="lower right")
bottom.plot(free["t"], 1e3 * free["depth"])
bottom.set_ylabel("goal depth [mm]")
bottom.set_xlabel("time [s]");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

late = free["force"][-500:]
outside = np.nonzero(abs(free["force"] - wanted[2]) > 0.1)[0]
settled = free["t"][outside[-1] + 1]  # [s] from here on, within 0.1 N
assert abs(late.mean() - wanted[2]) < 0.02 and np.ptp(late) < 0.1
glue("settled", float(settled), display=False)
glue("late_mean", float(late.mean()), display=False)
glue("late_ptp", float(np.ptp(late)), display=False)
glue("depth_end", float(1e3 * free["depth"][-1]), display=False)
```

The spike at the start is the finger landing on the table. From {glue:text}`settled:.1f` s on,
the force stays within 0.1 N of the wanted one. At the end it is {glue:text}`late_mean:.2f` N
on average, and it ripples by {glue:text}`late_ptp:.3f` N from peak to peak: every step changes
the force by about `max_force_step`, so near the target it steps over it and back. The goal ends
{glue:text}`depth_end:.1f` mm below the table, a depth we never had to compute. A smaller
`max_force_step` gives a smaller ripple and a slower approach. The law's `max_step` also caps how
far a Param moves in one step, in the Param's own unit (meters for a goal). It has nothing to do
with the `max_step` of `ModelPlant`, the integration step.

## Through a tank

Moving a goal deeper stores energy in the spring. The goal moves only as fast as a
[tank](energy.md) pays for it. Pass the tank in place of the controller, to the loop and to the
law. An empty tank fills from what the controller's dampers take, which happens while the finger
moves. Once it rests, nothing refills it:

```{code-cell} python
def tank(level):  # [J]
    return vmc.control.Tank(vmc.VMCController(compiled), level=level)

stalled = press(tank(0.0))
funded = press(tank(0.05))
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(free["t"], free["force"], label="no tank")
ax.plot(funded["t"], funded["force"], "--", label="tank with 0.05 J")
ax.plot(stalled["t"], stalled["force"], label="empty tank")
ax.axhline(wanted[2], color="0.5", linestyle=":", label="wanted")
ax.set_xlabel("time [s]")
ax.set_ylabel("force [N]")
ax.legend(loc="lower right");
```

```{code-cell} python
:tags: [remove-cell]
assert abs(funded["force"][-500:].mean() - wanted[2]) < 0.05
assert stalled["force"][-1] < 0.8 * wanted[2]
assert funded["level"].min() >= -1e-9 and stalled["level"].min() >= -1e-9
glue("stalled", float(stalled["force"][-1]), display=False)
glue("left", float(funded["level"][-1]), display=False)
glue("empty_level", float(stalled["level"][-1]), display=False)
```

The tank with 0.05 J gets there and ends with {glue:text}`left:.3f` J. The empty one stalls at
{glue:text}`stalled:.2f` N: its level is {glue:text}`empty_level:.3f` J at the end, because the
finger has come to rest and its dampers take nothing. A tank lets the controller's energy rise
only by what it holds, so a law running through one cannot pump energy into the controller,
whatever force it is asked for.

## By stiffness

The law moves any live Param, so the stiffness of the spring can do the work of its goal: the
same call with `"ctrl.press.stiffness"` in place of the goal's name. We fix the goal 20 mm below
the table and start with a spring that is too soft:

```{code-cell} python
soft = vmc.VMCController(compiled)
soft.set({"ctrl.press.stiffness": 40.0,  # [N/m]
          "ctrl.press.goal": [0.0, 0.05, 0.08]})  # [m]
stiff = press(soft, "ctrl.press.stiffness")
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
top.plot(stiff["t"], stiff["force"], label="contact force")
top.axhline(wanted[2], color="0.5", linestyle="--", label="wanted")
top.set_ylim(-0.2, 3.0)  # leaves out the finger's landing on the table
top.set_ylabel("force [N]")
top.legend(loc="lower right")
bottom.plot(stiff["t"], stiff["stiffness"])
bottom.set_ylabel("stiffness [N/m]")
bottom.set_xlabel("time [s]");
```

```{code-cell} python
:tags: [remove-cell]
depth = 0.02  # [m] the goal is below the table
series = wanted[2] * k / (k * depth - wanted[2])  # [N/m]
final = float(stiff["stiffness"][-500:].mean())
assert abs(final - series) < 2.0
assert abs(stiff["force"][-500:].mean() - wanted[2]) < 0.05
glue("k_end", final, display=False)
glue("k_series", float(series), display=False)
```

The stiffness settles at {glue:text}`k_end:.0f` N/m. The spring and the table push in series,
so the force is $K k d / (K + k)$ for a goal at depth $d$, and 2 N at 20 mm needs
{glue:text}`k_series:.0f` N/m. A new value stays within the Param's bounds, so a stiffness never
goes below zero. A stiffness given as a matrix stays symmetric and positive semidefinite, so the
spring cannot store negative energy.

## Laws without a model

Two more laws need no model of the robot, only a force. `ForceRatio` scales a stiffness up when
the force is too small and down when it is too large, by `max_change` (5 % here) times the
relative error at each step: quickly far from the target and gently near it. `Stiffening` sets
the stiffness from the measured force,

$$K(F) = K_\mathrm{low} + (K_\mathrm{high} - K_\mathrm{low})(1 - e^{-\alpha F}),$$

soft until the finger touches something, stiffer as it presses. In the call, `low` and `high`
are $K_\mathrm{low}$ and $K_\mathrm{high}$ and `alpha` is $\alpha$. With the goal fixed 20 mm
below the table and the spring starting soft:

```{code-cell} python
from virtualmodelcontrol.adaptation import ForceRatio, Stiffening

def fresh():
    controller = vmc.VMCController(compiled)
    controller.set({"ctrl.press.stiffness": 40.0,  # [N/m]
                    "ctrl.press.goal": [0.0, 0.05, 0.08]})  # [m]
    return controller

ratio_run = fresh()
ratio_law = ForceRatio(ratio_run, "ctrl.press.stiffness", max_change=0.05)
scaled = press(ratio_run, act=ratio_law.step)

stiff_run = fresh()
stiffen = Stiffening(stiff_run, "ctrl.press.stiffness",
                     low=40.0, high=200.0, alpha=0.2)  # [N/m], [N/m], [1/N]
stiffened = press(stiff_run,
                  act=lambda c, force, wanted: stiffen.step(c, force[2]))
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
top.plot(scaled["t"], scaled["force"], label="ratio law")
top.plot(stiffened["t"], stiffened["force"], "--", label="stiffening")
top.axhline(wanted[2], color="0.5", linestyle=":", label="wanted")
top.set_ylim(-0.2, 3.0)  # leaves out the finger's landing on the table
top.set_ylabel("force [N]")
top.legend(loc="lower right")
bottom.plot(scaled["t"], scaled["stiffness"])
bottom.plot(stiffened["t"], stiffened["stiffness"], "--")
bottom.set_ylabel("stiffness [N/m]")
bottom.set_xlabel("time [s]");
```

```{code-cell} python
:tags: [remove-cell]
assert abs(scaled["force"][-500:].mean() - wanted[2]) < 0.03
glue("ratio_force", float(scaled["force"][-500:].mean()), display=False)
glue("ratio_k", float(scaled["stiffness"][-500:].mean()), display=False)
glue("stiffened_force", float(stiffened["force"][-500:].mean()), display=False)
glue("stiffened_k", float(stiffened["stiffness"][-500:].mean()), display=False)
```

The ratio law ends at {glue:text}`ratio_force:.2f` N with a stiffness of
{glue:text}`ratio_k:.0f` N/m, as the gradient law did. The stiffening has no target. It settles
where the force it produces and the stiffness it asks for agree: {glue:text}`stiffened_force:.2f` N
at {glue:text}`stiffened_k:.0f` N/m. A stiffness that depends on the deflection, in place of the
force, is a nonlinear spring: `vmc.SigmoidSpring` and `vmc.PolynomialSpring`.

## What the calls need

- `ForceTracking(controller, site, params, normal=None, *, max_force_step=0.01, max_step=None,
  rate=None)`: a step is the largest one that changes the force by `max_force_step`, or moves
  a Param by `max_step`, or by a fixed `rate` if one is given. `site` is where the contact is, a name
  such as `"tip"`, an arc parameter `s`, or `(part, s)` for a part of an assembly, as in the
  [two arms](../examples/two-arms.md). `params` are glob patterns of the controller's live Params;
  compile with `runtime=[...]` for the Params that are not live by default.
- `normal` is the direction of the force the robot exerts on its surroundings, opposite to the
  surface's outward normal. Without it the law tracks the full 3D force. The measured and the
  wanted force are vectors of three entries.
- `step(controller, f_meas, f_des)` makes one step and returns the jump of the controller's
  energy [J] that was applied (a tank applies only the part it pays for). `law.alpha` is the step
  size of the last step. `direction(...)` gives the descent direction of each Param without
  applying it.
- `ForceRatio(controller, params, max_change=0.05)` and `Stiffening(controller, params, low,
  high, alpha)` take the controller and glob patterns like `ForceTracking`. `ForceRatio.step`
  takes the measured and the wanted force, `Stiffening.step` the measured force along the
  contact normal, a number. Both return the jump too; `law.ratio` and `law.k` are what they
  asked for.

## Take it to the robot

The loops above run a simulated plant. On the robot the same law runs inside the loop that talks
to the motors, with nothing from `vmc.sim`: the controller steps with the reading, and the law
changes the controller with the force that a load cell or an estimate gives. Here the finger's
controller of the first section runs through a tank that starts empty, as one control period of
your control loop would, with a made-up reading:

```{code-cell} python
from virtualmodelcontrol.control import Tank

tank = Tank(vmc.VMCController(compiled), level=0.0, capacity=0.01)  # [J]
law = ForceTracking(tank, "tip", "ctrl.press.goal", normal=[0, 0, 1])
wanted = np.array([0.0, 0.0, 2.0])  # [N]

def control_step(t, reading, force):
    """What your loop does each period: the torques, then the law."""
    command = tank.step(t, reading)
    law.step(tank, force, wanted)
    return command["motor_torque"]

reading = vmc.Signals(0.0, motor_position=[0.8, 0.8],
                      motor_velocity=[0.0, 0.0])
tank.reset(0.0, reading)
control_step(0.0, reading, force=np.array([0.0, 0.0, 0.5]))  # [N·m]
```
