---
file_format: mystnb
kernelspec:
  name: python3
---

# Shape the stiffness of the tip

In this tutorial we ask how stiff the soft arm is at its tip, choose the stiffness we want, and
let the controller find the springs that give it. Then we correct a position that the arm's own
stiffness holds back.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The stiffness at the tip

A spring pulls the tip to a goal, as in [your first controller](first-controller.md). The
spring's stiffness is a matrix, so that it can differ along each axis. We let the arm settle:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
tip = arm.point(s=1.0)
goal = np.array([0.08, 0.0, 0.40])  # [m]

ctrl = vmc.Mechanism("ctrl")
reach = vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=goal),
                         300.0 * np.eye(3))  # [N/m]
ctrl.add("reach", reach)
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
compiled = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
controller = vmc.VMCController(compiled)
plant = vmc.sim.ModelPlant(arm)

def settle(seconds):
    dt = 1 / helyx.CONTROL_RATE
    for _ in range(int(seconds / dt)):
        plant.write(controller.step(plant.t, plant.read()))
        plant.advance(dt)
    plant.write(controller.step(plant.t, plant.read()))

settle(2.0)
```

A push $\delta$ of the tip along $x$ is met by a force $K_{xx}\delta$ along $x$ and, if $K_{xz}$
is not zero, by a force $K_{xz}\delta$ along $z$. `TaskStiffness` gives the matrix $K$ from
the controller and the model of the arm: the stiffness of everything that holds the tip, the
arm's own and the springs', with the changes of the geometry counted.

```{code-cell} python
from virtualmodelcontrol.estimation import TaskStiffness

stiffness = TaskStiffness(controller, site=1.0)  # the tip
K = stiffness(controller)  # [N/m]
K.round(0)
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

assert K[2, 2] > 3 * K[1, 1] and abs(K - K.T).max() < 1e-6
glue("kxx", float(K[0, 0]), display=False)
glue("kyy", float(K[1, 1]), display=False)
glue("kzz", float(K[2, 2]), display=False)
glue("kxz", float(K[0, 2]), display=False)
```

Along $x$ the tip is {glue:text}`kxx:.0f` N/m stiff and along $y$ {glue:text}`kyy:.0f` N/m,
for a spring of 300 N/m. Along $z$ it is {glue:text}`kzz:.0f` N/m: the arm is stiff along its
length. The arm is bent a little, so the axes mix ({glue:text}`kxz:.0f` N/m between $x$ and
$z$). Only the matrix of the springs is ours to change.

## Ask for a stiffness

Say we want the tip stiffer than it is by 200 N/m along $x$ and $y$ and by 600 N/m along $z$.
`StiffnessTracking` takes the live stiffness Params and finds the ones that give a wanted
matrix. The task-space stiffness is affine in the springs' stiffness, so it solves a linear
system, and takes the solution of smallest norm.

```{code-cell} python
from virtualmodelcontrol.adaptation import StiffnessTracking

wanted = K + np.diag([200.0, 200.0, 600.0])  # [N/m]
law = StiffnessTracking(controller, site=1.0, params="ctrl.reach.stiffness")
history = [np.diag(K)]
for _ in range(3):
    law.step(controller, wanted)
    settle(2.0)  # the arm moves a little, and the matrix changes with it
    history.append(np.diag(stiffness(controller)))
history = np.array(history)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
for i, axis in enumerate("xyz"):
    line, = ax.plot(history[:, i], "o-", label=f"$K_{{{axis}{axis}}}$")
    ax.axhline(wanted[i, i], color=line.get_color(), linestyle=":")
ax.set_xticks(range(4))
ax.set_xlabel("steps")
ax.set_ylabel("stiffness at the tip [N/m]")
ax.legend(loc="center right");
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(history[-1], np.diag(wanted), rtol=0.01)
spring = controller.live_params()["ctrl.reach.stiffness"]
glue("k_after", float(np.abs(history[-1] / np.diag(wanted) - 1).max()), display=False)
glue("spring_z", float(spring[2, 2]), display=False)
```

Each step moves the Params all the way to what the model says gives the wanted stiffness. The arm
then moves a little, which changes the matrix, so a few steps close the gap: after three the
diagonal is within {glue:text}`k_after:.1%` of the wanted one. The spring along $z$ ends at
{glue:text}`spring_z:.0f` N/m. The stiffness the arm has by itself stays, so a stiffness below
that is out of reach: the springs cannot be negative, and a step stops at zero. The step
keeps every stiffness matrix symmetric and positive semidefinite. `fraction` moves only a part of the
way at each step, and a [tank](energy.md) takes the controller's energy into account when
passed in place of the controller.

## Put the tip where we want it

A spring with a stiffness of 300 N/m does not bring the tip to its goal: the arm's own stiffness
holds it back.

```{code-cell} python
:tags: [remove-cell]
controller.set({"ctrl.reach.stiffness": 300.0 * np.eye(3), "ctrl.reach.goal": goal})
settle(2.0)
```

```{code-cell} python
kin = vmc.Kinematics(arm)
miss = np.linalg.norm(goal - kin.position(plant.q, 1.0))  # [m]
```

The goal is only a place where the spring would be at rest. `PositionRegulation` moves it by
integral action: at every step it adds a share of the position error to the goal, until the tip
is at its target.

```{code-cell} python
from virtualmodelcontrol.adaptation import PositionRegulation

regulate = PositionRegulation(controller, {"ctrl.reach.goal": 1.0},
                              gain=0.02)
errors = []
for _ in range(int(4.0 * helyx.CONTROL_RATE)):
    plant.write(controller.step(plant.t, plant.read()))
    regulate.step(controller, {"ctrl.reach.goal": goal})
    errors.append(np.linalg.norm(goal - kin.position(plant.q, 1.0)))
    plant.advance(1 / helyx.CONTROL_RATE)
```

```{code-cell} python
:tags: [remove-cell]
assert errors[-1] < 1e-3 < miss
glue("miss", float(1e3 * miss), display=False)
glue("end", float(1e3 * errors[-1]), display=False)
glue("moved", float(1e3 * np.linalg.norm(controller.live_params()["ctrl.reach.goal"] - goal)),
     display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(np.arange(len(errors)) / helyx.CONTROL_RATE, 1e3 * np.array(errors))
ax.set_xlabel("time [s]")
ax.set_ylabel("distance to the target [mm]");
```

The tip starts {glue:text}`miss:.0f` mm from the target and ends {glue:text}`end:.2f` mm from it.
To get there the goal moved {glue:text}`moved:.0f` mm away from the target. No model of the
arm's stiffness or weight was used, only the position of the tip.
