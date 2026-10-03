---
file_format: mystnb
kernelspec:
  name: python3
---

# Parameters

Every number in a robot or a controller is a `Param`: a value with a unit, bounds and a scope.
That includes gains and goals, and also every geometric number of a robot: segment lengths,
tendon angles, spool radii, joint axes, mounting poses, masses.

```{code-cell} python
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm()
arm.params["seg2.delta"]          # the tendon angles of segment 2 [rad]
```

Change a value by assigning to it; models built from it use the new value from then on:

```{code-cell} python
import numpy as np
arm.params["seg1.L0"].value = 0.150          # a longer first segment [m]
print(vmc.Kinematics(arm).position(np.zeros(9), "tip"))   # the tip moved by 5 mm
```

## Scopes: how often a value may change

| Scope | Meaning | Examples |
|---|---|---|
| `fixed` | never changes | physical constants |
| `design` | changes with the hardware | lengths, radii, masses, tendon angles |
| `episode` | changes between runs | attachment points, projection directions, joint ranges |
| `stage` | may change at every control step | stiffness, damping, goals |

## Live and frozen parameters

When you compile, `stage` parameters stay **live**: the controller takes them as inputs, so you
can change them at every step. The others are **frozen** into the compiled function, which
makes it faster; change one of those and compile again (a few tens of milliseconds).

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - [0.1, 0.0, 0.6], 30.0))
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
law.live            # the parameters you can change while running
```

To keep a frozen parameter live, name it (glob patterns work):

```{code-cell} python
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.reach.s"])
law.live
```

## Changing values while running

```{code-cell} python
controller = vmc.VMCController(law)
controller.step(0.0, vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.zeros(9)))
jump = controller.set({"ctrl.reach.stiffness": 60.0, "ctrl.reach.s": 0.8})
print(f"energy jump: {jump:.4f} J")
```

`set` changes several values at once and returns the exact change in the controller's stored
energy, which energy-based safety checks need.

## Names

Parameter names are `<mechanism>.<component>.<name>` once compiled (`ctrl.reach.stiffness`,
`arm.seg1.L0`). In a robot or a controller alone, they drop the mechanism's name
(`arm.params["seg1.L0"]`).
