---
file_format: mystnb
kernelspec:
  name: python3
---

# Build your own robot

In this tutorial we describe a two-link arm from scratch, give it masses and simulate it. Then we
pack it as a template whose geometry is a set of arguments, test it, and take its controller to
a robot. [Joints and bodies](joints.md) and [Soft arms and assemblies](soft-and-assemblies.md)
go on with the other kinds of joints and models.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## A rigid arm from its DH table

`SerialChain.from_dh` takes a standard Denavit–Hartenberg table and returns the chain, with
the last frame as the site `tool`. Here two links of 0.30 m and 0.25 m turn about $z$, so the
arm moves in the $x$-$y$ plane:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.models import SerialChain

dh = SerialChain.from_dh(d=[0.0, 0.0], a=[0.30, 0.25], alpha=[0.0, 0.0])
dh.sites
```

## The same arm by product of exponentials

The library stores every chain as a product of exponentials: an axis and a point per joint, and
named sites that move with the joint they follow. Writing the arm this way lets us name the
elbow and the tip:

```{code-cell} python
chain = SerialChain(
    ["revolute", "revolute"],
    axes=[[0, 0, 1], [0, 0, 1]],  # at q = 0
    points=[[0, 0, 0], [0.30, 0, 0]],  # [m], one point on each axis
    sites={"elbow": (1, [0.30, 0, 0]), "tip": (2, [0.55, 0, 0])},
)
q = np.array([0.4, 0.9])  # [rad]
a = vmc.Kinematics(dh).position(q, "tool")
b = vmc.Kinematics(chain).position(q, "tip")
np.abs(a - b).max()  # [m], the two descriptions agree
```

## Masses, gravity, simulation

A robot is a mechanism around its model. Its physical components are point masses at sites,
gravity, and here a little viscous friction in the joints. The simulator needs all of them; the
controller needs only the point masses, for `GravityCompensation`. The static friction of the
motors, which holds a joint until the command passes a breakaway torque, is a part of the
transmission: see [static friction of the motors](friction.md).

```{code-cell} python
arm = vmc.Mechanism("arm", model=chain)
arm.add_param(vmc.Param("gravity", [0.0, -9.81, 0.0], unit="m/s^2"))
arm.add("m1", vmc.PointMass(arm.point("elbow"), 1.0))  # [kg]
arm.add("m2", vmc.PointMass(arm.point("tip"), 0.8))
arm.add("gravity", vmc.Gravity(arm))
joints = arm.joint([0, 1])
arm.add("friction", vmc.LinearDamper(joints, 0.05))  # [N·m·s/rad]

goal = np.array([0.25, 0.30, 0.0])  # [m]
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - goal, 200.0))
ctrl.add("damp", vmc.LinearDamper(arm.point("tip"), 30.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))

plant = vmc.sim.ModelPlant(arm, q0=[-1.2, 0.3])
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 500), T=2.0)
```

```{code-cell} python
:tags: [remove-output]
viz.animate(arm, log, "build-a-robot.mp4", plane="xy",
            springs=[("tip", goal)], trace="tip")
```

```{video} build-a-robot.mp4
:caption: The two-link arm, built on this page, pulled from below to its goal.
```

## A template

A template is a function that returns the robot with its geometry as arguments, so the same code
builds every variant. Each number also becomes a `Param` that can be changed or optimized later.

```{code-cell} python
def two_link(lengths=(0.30, 0.25), masses=(1.0, 0.8),
             gravity=(0.0, -9.81, 0.0), name="arm"):
    """A planar two-link arm; lengths [m], masses [kg], gravity [m/s²]."""
    a, b = lengths
    model = SerialChain(
        ["revolute", "revolute"],
        axes=[[0, 0, 1], [0, 0, 1]],
        points=[[0, 0, 0], [a, 0, 0]],
        sites={"elbow": (1, [a, 0, 0]), "tip": (2, [a + b, 0, 0])},
    )
    robot = vmc.Mechanism(name, model=model)
    robot.add_param(vmc.Param("gravity", gravity, unit="m/s^2"))
    robot.add("m1", vmc.PointMass(robot.point("elbow"), masses[0]))
    robot.add("m2", vmc.PointMass(robot.point("tip"), masses[1]))
    return robot

long = two_link(lengths=(0.4, 0.3))
vmc.Kinematics(long).position([0.0, 0.0], "tip")  # [m]
```

The templates in `virtualmodelcontrol.robots` follow the same pattern, with a separate
`add_dynamics` for what only the simulator needs.

## The tests a new robot needs

A model is right when its derivatives match finite differences, its rotations are rotations, it
stays finite at the poses where it is singular, it comes back from `to_dict` as it was and,
without friction or control, it keeps its energy. `check_model` makes these checks in one call
and raises an `AssertionError` that lists the ones that fail. Call it from your model's own
tests. It returns the worst error of each check:

```{code-cell} python
from virtualmodelcontrol.testing import check_model

free = two_link()
free.add("gravity", vmc.Gravity(free))  # masses and gravity, no friction
worst = check_model(free)
{name: f"{error:.0e}" for name, error in worst.items()}
```

The checks run at the neutral pose, where soft and rigid arms are often singular, and at a few
random ones, for every site (`at=` picks some, and `s=` the points of a continuous body, whose
positions must not jump). The energy check runs the simulator twice, the second time with a
step four times smaller. The drift of its implicit steps must fall with the step, and a force
that does work would not let it fall.

## Take it to the robot

A robot you describe is controlled like any other, and the controller needs nothing from
`vmc.sim`. The `controller` of the section on simulation above can drive a real two-link arm. One
control period of your control loop is a `step` with the motors' reading: the joint angles and
rates, since this arm has no transmission:

```{code-cell} python
reading = vmc.Signals(0.0, motor_position=np.array([-1.2, 0.3]),  # [rad]
                      motor_velocity=np.zeros(2))  # [rad/s]
controller.reset(0.0, reading)
torques = controller.step(0.0, reading)["motor_torque"]  # [N·m]
```

A robot with a transmission, such as a tendon-driven arm, reads and commands its motors instead:
see [Soft arms and assemblies](soft-and-assemblies.md#take-the-soft-arm-to-the-robot). To run a
model without CasADi, as numpy or PyTorch code, see [Use a model outside CasADi](outside.md).
