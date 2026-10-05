---
file_format: mystnb
kernelspec:
  name: python3
---

# How it works

This page explains the idea of Virtual Model Control, the two mechanisms behind every controller
in the library, and the four steps from a robot to its motor torques.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The idea

Virtual Model Control builds a controller from virtual physical elements attached to the robot.
To hold a fingertip at a point, we attach a spring between the fingertip and the point; to calm
a motion, a damper. The controller computes the forces these elements would exert, and the
motors apply them.

Every number of such a controller has a unit and a physical meaning, a stiffness in N/m or a
damping in N·s/m, so we read and tune it like a mechanism. Springs and dampers only store and
dissipate energy. A controller made of them can only give the robot back the energy it stored.

## Two mechanisms

The robot and its controller are two separate mechanisms, each a set of coordinates and of
components acting on them. Here is a finger, once as the robot and once as its controller:

```{code-cell} python
:tags: [remove-input]
from schematics import mechanism
mechanism.figure();
```

The robot mechanism describes the hardware. Its kinematics give the coordinates, here the joint
angles $\theta_1$ to $\theta_3$ and any point such as the tip. Its motors drive them. Its
components are physical: the masses of the phalanges, gravity and, on a soft robot, its own
stiffness and damping.

The controller is a virtual mechanism on the same coordinates. Here a spring pulls the tip to a
goal, a damper slows the tip, spiral springs keep each joint in its range and gravity
compensation cancels the weight of the phalanges. A controller can also have degrees of freedom
of its own, its virtual states.

The two never mix. A simulator moves the robot mechanism with its own components. The
controller's torques come from the controller's components alone, through the robot's
kinematics and motors. On the real robot, the hardware takes the place of the simulator.

## Four steps

```{code-cell} python
:tags: [remove-input]
from schematics import four_steps
four_steps.figure();
```

1. Describe the robot: a ready-made template from `virtualmodelcontrol.robots`, or a model of
   our own.
2. Place the virtual elements: components on the robot's coordinates, in a `vmc.Mechanism`.
3. Compile: `vmc.compile` turns the two mechanisms into one CasADi function from motor angles
   and rates to motor torques, with exact derivatives.
4. Run on a plant: `vmc.VMCController` evaluates that function at every control step. The plant
   is a simulator built from the robot mechanism alone (`vmc.sim.ModelPlant`), or the real
   robot.

## Where to go next

- [Your first controller](first-controller.md) runs the four steps on the soft arm.
- [Coordinates and components](coordinates-and-components.md) lists what we can attach, and
  where.
- [Parameters](parameters.md) shows which numbers can change while the robot runs.
- [Energy and passivity](energy.md) follows the energy through a run.
- [Build a robot](build-a-robot.md) describes a robot of our own.
- [Virtual Model Control](../concepts/vmc.md) and [Passivity](../concepts/passivity.md) give the
  equations behind these steps.
- [How the library is organized](../concepts/library.md) maps the packages, and the
  [glossary](../concepts/conventions.md) defines the terms.
