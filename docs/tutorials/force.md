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
wanted one. `ForceTracking` answers it: it moves live Params of the controller, first the
spring's goal, then its stiffness, by gradient descent of $\tfrac12 \lVert f - f_\mathrm{des} \rVert^2$. At every
control step it moves them by the largest step that changes the force by less than
`max_force_step` [N].

The force $f$ is the one the motors push through the contact point. The law needs the measured
force too, from a load cell on a real robot. Here it comes from the simulated table.

## Press with a chosen force

The table is the one of the [contact tutorial](contact.md), 6 cm below the finger's base.

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

The loop is the usual one with one more line: after the controller's step, the law changes the
goal. The wanted force is 2 N along $+z$, into the table.

```{code-cell} python
wanted = np.array([0.0, 0.0, 2.0])  # [N]
kin = vmc.Kinematics(finger)

def press(controller, adapted="ctrl.press.goal", steps=3000):  # 500 Hz
    law = ForceTracking(controller, "tip", adapted, normal=[0, 0, 1])
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    names = ["t", "force", "depth", "stiffness", "level"]
    log = {name: [] for name in names}
    for step in range(steps):
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - 0.06  # [m]
        measured = np.array([0.0, 0.0, k * max(0.0, below)])
        if step > 100:  # let the finger settle on the table first
            law.step(controller, measured, wanted)
        live = controller.live_params()
        level = getattr(controller, "level", 0.0)  # a tank's budget [J]
        row = (plant.t, measured[2], live["ctrl.press.goal"][2] - 0.06,
               float(live["ctrl.press.stiffness"]), level)
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
assert abs(late.mean() - wanted[2]) < 0.02 and late.std() < 0.03
glue("settled", float(settled), display=False)
glue("late_mean", float(late.mean()), display=False)
glue("late_std", float(late.std()), display=False)
glue("depth_end", float(1e3 * free["depth"][-1]), display=False)
```

The spike at the start is the finger landing on the table. From {glue:text}`settled:.1f` s on,
the force stays within 0.1 N of the wanted one. At the end it is {glue:text}`late_mean:.2f` N,
with a ripple of {glue:text}`late_std:.2f` N: the step always changes the force by about
`max_force_step`, so near the target it steps over it and back. The goal ends
{glue:text}`depth_end:.1f` mm below the table, a depth we never had to compute. A
smaller `max_force_step` gives a smaller ripple and a slower approach. `max_step` [m] also caps
how far a goal moves in one step.

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
ax.axhline(wanted[2], color="0.5", linestyle="--")
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
glue("spent", float(0.05 - funded["level"][-1]), display=False)
```

The tank with 0.05 J gets there, with {glue:text}`spent:.3f` J of it spent on the deeper goal. The
empty one stalls at {glue:text}`stalled:.2f` N, where the finger rests and nothing refills it. A
tank lets the controller's energy rise only by what it holds, so a law running through one
cannot pump energy into the controller, whatever force it is asked for.

## By stiffness

The law moves any live Param, so the stiffness of the spring can do the work of its goal. We
fix the goal 20 mm below the table and start with a spring that is too soft:

```{code-cell} python
soft = vmc.VMCController(compiled)
soft.set({"ctrl.press.stiffness": 40.0,  # [N/m]
          "ctrl.press.goal": [0.0, 0.05, 0.08]})  # [m]
stiff = press(soft, "ctrl.press.stiffness")
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
top.plot(stiff["t"], stiff["force"])
top.axhline(wanted[2], color="0.5", linestyle="--")
top.set_ylim(-0.2, 3.0)  # leaves out the finger's landing on the table
top.set_ylabel("force [N]")
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

## What the law needs

- `ForceTracking(controller, site, params, normal=None)`: `site` is where the contact is, a name
  or an arc parameter `s`, and `params` are glob patterns of the controller's live Params. Compile
  with `runtime=[...]` for the Params that are not live by default.
- `normal` is the direction of the force the robot exerts. Without it the law tracks the full 3D
  force.
- `step(controller, f_meas, f_des)` makes one step and returns the step size and the jump of the
  controller's energy. `direction(...)` gives the descent direction of each Param without
  applying it.
