---
file_format: mystnb
kernelspec:
  name: python3
---

# Simulate a closed loop

The robot's own components (masses, stiffness, damping, gravity) define its dynamics, so the
same robot object that the controller is built on can also be simulated.

## The robot with its dynamics

`helyx.add_dynamics` gives the soft arm its stiffness and damping in Δ and gravity (default
values: the simulated arm's).

```{code-cell} python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
list(arm.components)
```

## A controller

A spring pulls the tip to a goal, a damper on the tip adds damping, and gravity compensation
cancels the weight of the lumped masses.

```{code-cell} python
goal = np.array([0.08, 0.0, 0.68])
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 300.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
```

## Run

`ModelPlant` integrates the dynamics with linearly implicit Euler steps (stable for the arm's
stiff tendons). It reads like the hardware: motor angles and rates. `run` reads, commands and
advances at the control rate; its guard sends zero torque if a reading is missing or not finite.

```{code-cell} python
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=2.0)

rows = log.arrays()
tip = evaluate_frame(arm.model, plant.q, 1.0)[1]
print("steps:", rows["t"].shape[0])
print("tip at rest [m]:", tip.round(4))
print("distance to goal [m]:", round(float(np.linalg.norm(tip - goal)), 4))
```

The tip stops short of the goal: the virtual spring balances the arm's own stiffness, and a
stiffer virtual spring brings it closer.

## Energy of the robot

```{code-cell} python
dyn = plant.dynamics
T, V = dyn.energy(plant.q, plant.v, plant.p, plant.t)
print(f"kinetic {float(T):.2e} J, stored {float(V):.4f} J")
```
