---
file_format: mystnb
kernelspec:
  name: python3
---

# Coordinates

A **coordinate** is a quantity of the robot that an element acts on: the position of a point,
a joint angle, the distance between two points, the component of a vector along a direction.
Every component (spring, damper, mass, source) acts on one coordinate.

Coordinates are written with CasADi, so the library differentiates them exactly: their rate
`ẏ = J v` and the torque `τ = Jᵀ f` that a force `f` on them produces come out automatically.

## The coordinates

| Coordinate | Value | Create it with | Typical use |
|---|---|---|---|
| point of the robot | position [m] | `robot.point("tip")`, `robot.point(s=0.5)` | attach springs and dampers to the body |
| joint | joint angles [rad] | `robot.joint(2)`, `robot.joint(slice(0, 3))` | joint-space stiffness, joint limits |
| reference | a goal, an obstacle | `vmc.Ref("goal", 3, value=[...])`, or a plain list | where a spring pulls to |
| difference | `a − b` | `point - goal` | the deflection of a spring |
| projection | `n̂ᵀ c` (one number) | `vmc.Projection(c, direction)` | act along one direction only (a cart) |
| norm | `‖c‖` | `vmc.Norm(c)` | distances |
| slice, stack | some entries; several stacked | `c[0]`, `vmc.Stack(a, b)` | build bigger coordinates |
| virtual state | a controller's own degree of freedom | `ctrl.add_state("flywheel")` | oscillators, followers, filters |
| custom | any CasADi expression | `vmc.Custom(fn, [children], dim=...)` | anything else |

On a continuum robot, `robot.point(s=...)` takes an arc parameter `s` from 0 (base) to 1
(tip), uniform in arc length. On a rigid robot or a hand, points are named sites such as
`"tip"` or `"index/tip"`.

## Building a deflection

The deflection of a spring is "where the point is, minus where it should be": `y = x − x_ref`.

```{code-cell} python
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm()
tip = arm.point(s=1.0)                      # a point: 3 numbers [m]
goal = vmc.Ref("goal", 3, value=[0.1, 0.0, 0.6])
deflection = tip - goal                     # a Difference
vertical = vmc.Projection(deflection, [0.0, 0.0, 1.0])  # only the vertical part
print(deflection, vertical, sep="\n")
```

A plain list on either side of `-` becomes a live reference automatically, so
`tip - [0.1, 0.0, 0.6]` works too; its value can be changed while the controller runs.

## Attachment points are parameters

`s=0.5` becomes a parameter of the point, so the attachment point can be moved later (or
optimized). It is frozen when the controller is compiled, unless you ask for it to stay live
(see [Parameters](parameters.md)).
