---
file_format: mystnb
kernelspec:
  name: python3
---

# Extend the library

In this tutorial we add a new spring, a new coordinate, a new kind of model and a new plant,
and register them so that other projects can use them by name.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

| To add | Write | Register as |
|---|---|---|
| a spring, damper or source | a `Component` subclass | `"component"` |
| a quantity to act on | a `Custom` coordinate, or a `Coordinate` subclass | |
| a kind of robot | a class with `space`, `params`, `sites` and `frame` | `"model"` |
| a transmission | the `Actuation` protocol | `"actuation"` |
| a simulator or hardware | `read`, `write` and `close` (and, simulated, `t`, `reset`, `advance`) | |

## A new component

A storage component defines its energy $V(y)$; automatic differentiation gives the force
$f = -\partial V / \partial y$. Its numbers are Params, created with `_param`.

```{code-cell} python
import casadi as ca
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.mechanisms import Component


@vmc.register("component", "quartic_spring")
class QuarticSpring(Component):
    """V = ¼ k ‖y‖⁴: soft near the goal, stiff far from it."""

    kind = "storage"

    def __init__(self, coord, stiffness):
        super().__init__(coord)
        self.stiffness = self._param("stiffness", stiffness,
                                     unit="N/m^3", scope="stage")

    def energy(self, ctx, y):
        return 0.25 * ctx.param(self.stiffness) * ca.sumsqr(y) ** 2
```

It acts on any coordinate, and `compile` treats it like the built-in springs:

```{code-cell} python
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-145-145")
tip = arm.point(s=1.0)
ctrl = vmc.Mechanism("ctrl")
ctrl.add("soft", QuarticSpring(tip - [0.1, 0.0, 0.40], 2.0e4))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))
meas = vmc.Signals(0.0, motor_position=np.zeros(9),
                   motor_velocity=np.zeros(9))
controller.step(0.0, meas)["motor_torque"].round(4)  # [N·m]
```

A dissipation component defines `force(ctx, y, yd)` with $f \cdot \dot y \le 0$; a source
defines `force`, and its power is metered.

## A new coordinate

`Custom` turns a CasADi expression of other coordinates into a coordinate. Here, the height of
the tip above the middle of the arm, held by a spring:

```{code-cell} python
height = vmc.Custom(lambda tip, mid: tip[2] - mid[2],
                    [arm.point(s=1.0), arm.point(s=0.5)], dim=1, unit="m")
ctrl = vmc.Mechanism("ctrl")
ctrl.add("keep_height", vmc.LinearSpring(height - [0.36], 50.0))
vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)).live  # its live Params
```

A reusable coordinate subclasses `Coordinate`: it states its `dim` and `unit`, computes
`value(ctx)` with CasADi operations, and returns its Params from `params()` and its inputs
from `children()`.

## A new kind of model

A model gives a configuration space, its Params, its named sites, and `frame(q, at, p)`, the
rotation and position of a site written with CasADi operations. Everything else (coordinates,
components, compile, simulation) then works unchanged. A pendulum swinging about $z$:

```{code-cell} python
@vmc.register("model", "pendulum")
class Pendulum:
    """Pendulum about z; q = angle [rad]. Params: ``L`` length [m]."""

    q_unit = "rad"

    def __init__(self, length):
        self.space = vmc.Euclidean(1)
        self.params = vmc.ParamSet(
            [vmc.Param("L", length, unit="m", scope="design")])
        self.sites = ("bob",)

    def frame(self, q, at, p):
        c, s = ca.cos(q[0]), ca.sin(q[0])
        R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0),
                       ca.horzcat(0, 0, 1))
        return R, p["L"] * ca.vertcat(c, s, 0)


robot = vmc.Mechanism("pendulum", model=Pendulum(0.5))
vmc.Kinematics(robot).jacobian([0.0], "bob")  # [m/rad]
```

Keep `frame` smooth: no numpy and no Python `if` on symbols (use `casadi.if_else`), and
regularize singular poses. A new model then passes the tests of
[Build your own robot](build-a-robot.md).

## A new plant

A plant is anything with `read`, `write` and `close`; a simulated one also has a time `t` and
`advance(dt)`, and `vmc.sim.run` drives it. This wrapper applies each command one control
period late, as a real loop with its communication delay does:

```{code-cell} python
class OneStepLate:
    """A simulated plant that applies each command one step late."""

    def __init__(self, plant):
        self.plant, self.u = plant, np.zeros_like(plant.u)

    @property
    def t(self):
        return self.plant.t

    def read(self):
        return self.plant.read()

    def write(self, cmd):
        self.plant.write(vmc.Signals(self.t, motor_torque=self.u))
        self.u = np.array(cmd["motor_torque"], dtype=float)

    def advance(self, dt):
        self.plant.advance(dt)

    def close(self):
        self.plant.close()


arm = helyx.add_dynamics(helyx.arm("145-145-145"))
ctrl = vmc.Mechanism("ctrl")
goal = [0.1, 0.0, 0.40]  # [m]
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 300.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))
late = OneStepLate(vmc.sim.ModelPlant(arm))
log = vmc.sim.run(late, controller, vmc.sim.SimClock(dt=1 / 330), T=0.5)
log.arrays()["motor_torque"].shape  # one row per step
```

```{code-cell} python
:tags: [remove-output]
from virtualmodelcontrol import viz

viz.animate(arm, log, "extend.mp4", springs=[(1.0, goal)], trace=1.0)
```

```{video} extend.mp4
:caption: The arm driven through the late plant: each torque arrives one control period late.
```

[Tuning](tuning.md) uses the same idea to show how the delay limits damping.

## From another project

A project registers its own components, models or transmissions without changing the library,
through an entry point in its `pyproject.toml`:

```toml
[project.entry-points."virtualmodelcontrol.plugins"]
my_project = "my_project.vmc_plugins"
```

Importing `my_project.vmc_plugins` must run its `register(...)` decorators. The library loads
the plugins the first time a name is looked up and not found, so configuration files can use
the plugins' names: a robot template as `"robot"`, a simulation's dynamics as `"dynamics"`, an
output stage as `"output"`, a controller template as `"controller"`, a component as
`"component"`, and a coordinate kind as `"coordinate"`, a function `(args, scope, path)` that
builds it ([Experiments in files](configurations.md)).
