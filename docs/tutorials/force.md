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
cell, or from an estimate. Here it comes from the simulated table first.

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
`adapted` names the Param, `sensor` replaces the table's force by an estimate, and `act` replaces
the law, in the sections below.

```{code-cell} python
wanted = np.array([0.0, 0.0, 2.0])  # [N]
kin = vmc.Kinematics(finger)

def press(controller, adapted="ctrl.press.goal", sensor=None, act=None,
          steps=3000):
    if act is None:
        law = ForceTracking(controller, "tip", adapted, normal=[0, 0, 1])
        act = law.step
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    names = ["t", "force", "told", "depth", "stiffness", "level"]
    log = {name: [] for name in names}
    for step in range(steps):  # 500 Hz
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - 0.06  # [m]
        table = np.array([0.0, 0.0, k * max(0.0, below)])
        told = table if sensor is None else sensor(controller)
        if step > 100:  # let the finger settle on the table first
            act(controller, told, wanted)
        live = controller.live_params()
        level = getattr(controller, "level", 0.0)  # a tank's budget [J]
        depth = live["ctrl.press.goal"][2] - 0.06  # [m]
        stiffness = float(live["ctrl.press.stiffness"])  # [N/m]
        row = (plant.t, table[2], told[2], depth, stiffness, level)
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
`max_force_step` gives a smaller ripple and a slower approach. `max_step` also caps how far a
Param moves in one step, in the Param's own unit (metres for a goal).

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

## Without a force sensor

A load cell is not always there. The controller knows what it commands, and a model knows what
the arm's own stiffness and weight hold. The rest of the balance is the contact. `ContactForce`
computes it, for the controller's Params as they are at its last step. The plant has the table,
so the model it uses is a second finger without one. The law is told the estimate, and never
the table's force:

```{code-cell} python
from virtualmodelcontrol.estimation import ContactForce

def blind(model):
    controller = vmc.VMCController(compiled)
    estimate = ContactForce(controller, "tip", [0, 0, 1], robot=model)
    return press(controller, sensor=estimate)

exact = blind(adapt.add_dynamics(adapt.finger()))
```

An estimate is as good as its model. We run it again with the finger's last phalanx 15 g in the
model and 25 g in the plant:

```{code-cell} python
light = adapt.add_dynamics(adapt.finger())
light.params["m_dip.mass"].value = 0.015  # [kg]
wrong = blind(light)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(exact["t"], exact["force"], label="exact model")
ax.plot(wrong["t"], wrong["force"], label="last phalanx 10 g too light")
ax.axhline(wanted[2], color="0.5", linestyle=":", label="wanted")
ax.set_ylim(-0.2, 3.0)
ax.set_xlabel("time [s]")
ax.set_ylabel("table's force [N]")
ax.legend(loc="lower right");
```

```{code-cell} python
:tags: [remove-cell]
good, bad = exact["force"][-500:].mean(), wrong["force"][-500:].mean()
told = wrong["told"][-500:].mean()
assert abs(good - wanted[2]) < 0.02 and abs(told - wanted[2]) < 0.02
assert 0.05 < bad - wanted[2] < 0.15
glue("blind_exact", float(good), display=False)
glue("blind_wrong", float(bad), display=False)
glue("blind_told", float(told), display=False)
```

With the exact model the table's force settles at {glue:text}`blind_exact:.2f` N. With the wrong
one the law believes it holds {glue:text}`blind_told:.2f` N, and the table feels
{glue:text}`blind_wrong:.2f` N: the model's error goes straight into the force. A load cell
closes this loop, because what it tells the law is the true force. It is the same call with the
cell's reading in place of the estimate.

## An object's compliance

How soft is the object the finger touches? Press it at a gentle setting of the controller's
stiffness, then at stiffer ones. The tip sinks a little further and pushes a little harder, and
the ratio of the two changes, the distance over the force, is the object's compliance in m/N.
Both are known without a sensor on the object: the tip's position from the joint angles, and
the force from `ContactForce`. `object_compliance` takes the median of the samples of each
setting and gives that ratio.

Here three objects of stiffness 4000, 1000 and 250 N/m are pressed 1.5 cm in by the spring of
the controller, at 50 N/m first and then at 100 and 200 N/m:

```{code-cell} python
from virtualmodelcontrol.estimation import ContactForce, object_compliance

def probe(stiffness, press, model=None, steps=750):
    """Samples of the tip's position and of the estimated force [N]."""
    world = adapt.add_dynamics(adapt.finger())
    tip = world.point("tip")
    surface = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
    world.add("object", vmc.ContactSpring(surface, stiffness))
    world.add("cushion", vmc.ContactDamper(surface, 5.0))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("press", vmc.LinearSpring(tip - [0.0, 0.05, 0.075], press))
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("limits", adapt.joint_limit_spring(world))
    ctrl.add("gravity", vmc.GravityCompensation(world))
    system = vmc.VirtualMechanismSystem(world, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    model = model or adapt.add_dynamics(adapt.finger())
    estimate = ContactForce(controller, "tip", [0, 0, 1], robot=model)
    plant = vmc.sim.ModelPlant(world, q0=[0.8, 0.8], max_step=1e-4)
    kin, position, force = vmc.Kinematics(world), [], []
    for _ in range(steps):  # 500 Hz
        plant.write(controller.step(plant.t, plant.read()))
        position.append(kin.position(plant.q, "tip"))
        force.append(estimate(controller))
        plant.advance(1 / 500)
    return np.array(position[-150:]), np.array(force[-150:])  # last 0.3 s

found = {}
for stiffness in (4000.0, 1000.0, 250.0):  # [N/m]
    gentle = probe(stiffness, 50.0)
    stiff = [probe(stiffness, press) for press in (100.0, 200.0)]
    found[stiffness] = [1e3 * object_compliance(*s, *gentle) for s in stiff]
    print(f"{1e3 / stiffness:5.2f} mm/N: {found[stiffness][0]:5.2f} and "
          f"{found[stiffness][1]:5.2f} mm/N")
```

```{code-cell} python
:tags: [remove-cell]
for stiffness, values in found.items():
    assert np.allclose(values, 1e3 / stiffness, rtol=0.03), (stiffness, values)
glue("worst_compliance", float(max(abs(np.array(v) * s / 1e3 - 1).max()
                                   for s, v in found.items()) * 100),
     display=False)
```

Each line gives the object's true compliance, $1/k$, then the estimates from the two stiffer
settings. With the exact model of the finger they agree to {glue:text}`worst_compliance:.1f` %.

```{code-cell} python
lighter = adapt.add_dynamics(adapt.finger())
lighter.params["m_dip.mass"].value = 0.015  # [kg], 10 g under the plant's
baseline = probe(1000.0, 50.0, lighter)
off = 1e3 * object_compliance(*probe(1000.0, 200.0, lighter), *baseline)
```

```{code-cell} python
:tags: [remove-cell]
glue("wrong_model", float(off), display=False)
assert abs(off - 1.0) < 0.03
```

A model that is wrong by a constant, here a last phalanx 10 g too light, moves the forces of both
settings by the same amount, and the difference does not see it: the same object of 1.00 mm/N
gives {glue:text}`wrong_model:.2f` mm/N.

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
                  act=lambda c, told, wanted: stiffen.step(c, told[2]))
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

- `ForceTracking(controller, site, params, normal=None)`: `site` is where the contact is, a name
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
- `ContactForce(controller, site, normal=None, robot=None)` estimates the force from the
  controller's command and the model of the robot. Call it with the controller. `robot` is the
  model that holds the arm at rest, without the surroundings: it gives only the arm's own
  stiffness and weight, and the motors' efficiency is the system's. The estimate uses the
  controller's law, before the output stages, and leaves velocities out. A Param that acts only
  on a virtual state does not change the force at once, and the laws leave it.
