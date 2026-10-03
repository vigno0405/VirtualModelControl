---
file_format: mystnb
kernelspec:
  name: python3
---

# Your first controller

This page builds a controller for the three-segment soft arm, evaluates its torques and changes
a gain while it runs. Every cell runs when the documentation is built.

## The robot

A robot is a `Mechanism` with a kinematic model. The `helyx` module builds the soft arm: a
piecewise-constant-curvature (PCC) model with three segments, three tendons per segment, a lumped
mass per segment and the gravity vector of its mounting.

```{code-cell} python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")
print(arm)
print(list(arm.params)[:6], "...")
```

Every number is a `Param` with a unit and a scope, geometry included:

```{code-cell} python
arm.params["seg1.delta"]
```

## The controller

The controller is a second mechanism. Its components act on coordinates of the robot: here a
spring pulls the tip towards a goal, and a damper slows the middle of the arm.

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
tip = arm.point(s=1.0)            # arc parameter s: 0 at the base, 1 at the tip
middle = arm.point(s=0.5)
ctrl.add("drag", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=[0.1, 0.0, 0.6]), 30.0))
ctrl.add("damp", vmc.LinearDamper(middle, 1.5))
ctrl.add("gravity", vmc.GravityCompensation(arm))
```

## Compile and run

`compile` turns the robot and the controller into CasADi functions. Gains and goals (scope
`stage`) stay live inputs; geometry and attachment points are folded in.

```{code-cell} python
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
print(law.live)
```

The controller reads motor angles and rates and returns motor torques. Motor angles follow the
library convention: θ > 0 pulls a tendon.

```{code-cell} python
controller = vmc.VMCController(law)
meas = vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.zeros(9))
controller.step(0.0, meas)["motor_torque"].round(4)
```

## Change a gain while running

`set` changes live Params at once and returns the jump of the controller's stored energy, which
energy-based safety layers need.

```{code-cell} python
jump = controller.set({"ctrl.drag.stiffness": 60.0})
print(f"energy jump: {jump:.3e} J")
controller.step(0.003, meas)["motor_torque"].round(4)
```

Attachment points are frozen by default. To move one while running, compile with
`runtime=["ctrl.drag.s"]` (glob patterns work, for example `"*.s"`).
