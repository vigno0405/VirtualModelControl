---
file_format: mystnb
kernelspec:
  name: python3
---

# Extend the library

Everything is built from a few small interfaces, so new elements plug in without editing the
library. This page shows each extension point with code that runs.

| To add | Implement | Register as |
|---|---|---|
| a spring, damper or source | a `Component` subclass (`energy` and/or `force`) | `"component"` |
| a quantity to act on | a `Coordinate` subclass, or `Custom` | none |
| a robot | a kinematic model (see [Build a robot](build-a-robot.md)) | `"model"` |
| a transmission | the `Actuation` protocol | `"actuation"` |
| a simulator or hardware | the `Plant` protocol (`read`, `write`, `close`) | none |

## A new component

A storage component defines its energy `V(y)`; the force `f = −∂V/∂y` then follows by
automatic differentiation (define `force` too when a closed form is cheaper). Parameters become
`Param`s through `_param`, with a unit and a scope.

```{code-cell} python
import casadi as ca
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.mechanisms import Component


@vmc.register("component", "quartic_spring")
class QuarticSpring(Component):
    """Spring with V = ¼ k ‖y‖⁴: soft near the goal, stiff far from it."""

    kind = "storage"

    def __init__(self, coord, stiffness):
        super().__init__(coord)
        self.stiffness = self._param("stiffness", stiffness, unit="N/m^3", scope="stage")

    def energy(self, ctx, y):
        return 0.25 * ctx.param(self.stiffness) * ca.sumsqr(y) ** 2
```

It works with every coordinate, and `compile` handles it like any built-in component:

```{code-cell} python
from virtualmodelcontrol.robots import helyx

arm = helyx.arm()
ctrl = vmc.Mechanism("ctrl")
ctrl.add("soft", QuarticSpring(arm.point(s=1.0) - [0.1, 0.0, 0.6], 2.0e4))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
meas = vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.zeros(9))
controller.step(0.0, meas)["motor_torque"].round(5)
```

A dissipation component defines `force(ctx, y, yd)` with `f · ẏ ≤ 0`; a source defines `force`
and its power is metered. The kinds are listed in `virtualmodelcontrol.mechanisms.KINDS`.

## A new coordinate

`Custom` wraps any CasADi expression of other coordinates. Here, the height of the tip above the
middle of the arm:

```{code-cell} python
height = vmc.Custom(lambda tip, mid: tip[2] - mid[2], [arm.point(s=1.0), arm.point(s=0.5)],
                    dim=1, unit="m")
ctrl = vmc.Mechanism("ctrl")
ctrl.add("keep_height", vmc.LinearSpring(height - [0.36], 50.0))
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
law.live
```

For a reusable coordinate, subclass `Coordinate`: give it `dim` and `unit`, implement
`value(ctx)` with CasADi operations, and return its Params from `params()` and its inputs from
`children()`.

## A new transmission

An actuation maps the configuration to motor angles and motor torques to generalized forces.
The `Actuation` protocol lists the methods; `Direct` (one motor per coordinate) and
`TendonTransmission` are the built-in ones.

```{code-cell} python
from virtualmodelcontrol.models import Actuation, Direct

[name for name in Actuation.__dict__ if not name.startswith("_")]
```

## A new plant

Hardware and simulators share one interface: `read()` returns `Signals` with `motor_position`
[rad] and `motor_velocity` [rad/s]; `write(cmd)` applies `cmd["motor_torque"]` [N·m]. A
simulator also has `t`, `reset` and `advance(dt)`, and then `vmc.sim.run` drives it.

## From a project repository

A project can register its own components, models or transmissions without forking the library.
Declare an entry point in the project's `pyproject.toml`:

```toml
[project.entry-points."virtualmodelcontrol.plugins"]
my_project = "my_project.vmc_plugins"
```

Importing `my_project.vmc_plugins` must run its `register(...)` decorators. The library loads
every plugin the first time a name is looked up and not found
(`virtualmodelcontrol.core.get(kind, name)`), so configuration files can refer to plugin names.
