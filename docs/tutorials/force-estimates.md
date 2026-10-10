---
file_format: mystnb
kernelspec:
  name: python3
---

# Estimate a contact force

In this tutorial the fingertip presses a table with a chosen force and nothing measures it: we
estimate it from the controller's command and a model of the robot, follow it while the finger
moves, and read the compliance of an object from it.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.robots import adapt
```

## The finger and the table

They are those of [Track a contact force](force.md): a plane 6 cm below the finger's base, a hard
spring of $10^4$ N/m, and a spring of 100 N/m pulling the tip to a goal that the law moves until the
force is 2 N. Below the law is told an estimate, never the table's force:

```{code-cell} python
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
compiled = vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl))
wanted = np.array([0.0, 0.0, 2.0])  # [N]
kin = vmc.Kinematics(finger)
```

## Without a force sensor

A load cell is not always there. The controller knows what it commands, and a model knows what
the arm's own stiffness and weight hold. The rest of the balance is the contact. `ContactForce`
computes it, for the controller's Params as they are at its last step. The plant has the table,
so the model it uses is a second finger without one. The law is told the estimate:

```{code-cell} python
from virtualmodelcontrol.estimation import ContactForce

def blind(model, steps=3000):
    """One run: the table's force and what the law was told [N]."""
    controller = vmc.VMCController(compiled)
    estimate = ContactForce(controller, "tip", [0, 0, 1], robot=model)
    law = ForceTracking(controller, "tip", "ctrl.press.goal",
                        normal=[0, 0, 1])
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    t, table, told = [], [], []
    for step in range(steps):  # 500 Hz
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - 0.06  # [m]
        guess = estimate(controller)
        if step > 100:  # let the finger settle on the table first
            law.step(controller, guess, wanted)
        t.append(plant.t)
        table.append(k * max(0.0, below))
        told.append(guess[2])
        plant.advance(1 / 500)
    return {"t": np.array(t), "force": np.array(table),
            "told": np.array(told)}

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
from myst_nb import glue

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

## While it moves

`ContactForce` assumes the robot is at rest, and so it is wrong while the finger lands on the
table. A `MomentumObserver` has no such assumption. It works from the robot's momentum
$M(q)\dot q$ and what the model says would change it, so it needs no acceleration. What
the model does not explain it takes for a force from outside. Its estimate follows the true force
like a low-pass filter of bandwidth `gain` [1/s]: the higher, the faster, and the more noise it
lets through. Like `ContactForce`, it takes the model of the finger without the table. Its
arguments are the system, the step between calls [s], the `gain`, and that model:

```{code-cell} python
from virtualmodelcontrol.estimation import ContactForce, MomentumObserver

world = adapt.add_dynamics(adapt.finger())
tip = world.point("tip")
gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
world.add("table", vmc.ContactSpring(gap, 1e4))
world.add("cushion", vmc.ContactDamper(gap, 5.0))
ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(tip - [0.0, 0.05, 0.075], 100.0))
ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
ctrl.add("limits", adapt.joint_limit_spring(world))
ctrl.add("gravity", vmc.GravityCompensation(world))
controller = vmc.VMCController(
    vmc.compile(vmc.VirtualMechanismSystem(world, ctrl)))
model = adapt.add_dynamics(adapt.finger())
at_rest = ContactForce(controller, "tip", [0, 0, 1], robot=model)
moving = MomentumObserver(controller.compiled.system, 1 / 500, 300.0,
                          robot=model)  # system, step [s], gain [1/s]

plant = vmc.sim.ModelPlant(world, q0=[0.8, 0.8], max_step=1e-4)
t, true, rest, flow = [], [], [], []
for _ in range(500):  # 500 Hz
    u = controller.step(plant.t, plant.read())["motor_torque"]
    plant.write(vmc.Signals(plant.t, motor_torque=u))
    moving(plant.q, plant.v, u, plant.t)  # q, v and the torques just sent
    pushes = plant.elements()  # what each plant component pushes with
    t.append(plant.t)
    true.append(pushes["table"]["force"][0] + pushes["cushion"]["force"][0])
    rest.append(at_rest(controller)[2])
    flow.append(moving.force("tip", [0, 0, 1])[2])
    plant.advance(1 / 500)
t, true, rest, flow = (np.array(x) for x in (t, true, rest, flow))
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(t, true, label="table's force")
ax.plot(t, rest, label="at rest")
ax.plot(t, flow, label="momentum observer")
ax.set_xlim(0, 0.2)
ax.set_ylim(-3, 8)
ax.set_xlabel("time [s]")
ax.set_ylabel("force on the table [N]")
ax.legend(loc="upper right");
```

```{code-cell} python
:tags: [remove-cell]
early = t < 0.2
error = lambda estimate: float(np.abs(estimate - true)[early].mean())
assert error(flow) < 0.6 * error(rest)
assert np.abs(flow - true)[t > 0.5].max() < 0.02 and abs(rest[0] - true[0]) > 3
glue("rest_start", float(rest[0]), display=False)
glue("rest_error", error(rest), display=False)
glue("flow_error", error(flow), display=False)
```

Before the finger arrives, the estimate at rest already reads {glue:text}`rest_start:.1f` N: it
sees the controller pulling and takes the finger to be held. The observer starts from no
force and follows the true one a little late, by about $1/\text{gain}$ = 3 ms. In the first
0.2 s its error is on average {glue:text}`flow_error:.2f` N, against {glue:text}`rest_error:.2f` N
at rest. Once the finger rests, all three agree.

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
settings. With the exact model of the finger they agree to within {glue:text}`worst_compliance:.0e` %.

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

## What the calls need

- `ContactForce(controller, site, normal=None, robot=None)` estimates the force from the
  controller's command and the model of the robot. Call it with the controller. `robot` is the
  model of the arm alone, without the surroundings: it gives the arm's own stiffness and weight,
  which the command balances first. The transmission, with its efficiency, is the system's.
  The estimate uses the torques the controller's law asks for, before any output stage, and
  leaves velocities out. A Param that acts only on a virtual state does not change the force at
  once, so `ForceTracking` finds no direction for it and leaves it where it is.
