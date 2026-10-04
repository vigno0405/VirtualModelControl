---
file_format: mystnb
kernelspec:
  name: python3
---

# virtualmodelcontrol

`virtualmodelcontrol` is a Python library for Virtual Model Control. You build a controller by
attaching virtual springs, dampers and masses to your robot, and the library turns them into
motor torques at the control rate, in simulation and on the real robot.

```{video} index-hero.mp4
:caption: A soft arm pulled to a point by one virtual spring, simulated with the code below.
```

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

The robot and its controller are two separate mechanisms. The robot mechanism describes the
hardware: its kinematics, its motors, its masses and, for simulation, its own stiffness and
damping. The controller, a virtual mechanism, holds only the virtual elements we place on the
robot. Here they are a spring from
the tip to a goal and a damper on the tip; `compile` turns them into one fast function:

```{code-cell} python
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))  # the robot
goal = [0.25, 0.0, 0.55]  # [m]

ctrl = vmc.Mechanism("ctrl")  # the controller
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 600.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))

system = vmc.VirtualMechanismSystem(arm, ctrl)
controller = vmc.VMCController(vmc.compile(system))
```

The same `controller` runs on the hardware. Here it runs on a simulation built from the robot's
mechanism alone:

```{code-cell} python
:tags: [remove-output]
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=3.0)
vmc.viz.animate(arm, log, "index-hero.mp4", springs=[(1.0, goal)],
                trace=1.0, invert=True)
```

## Install

```bash
pip install virtualmodelcontrol
```

[Installation](installation.md) covers virtual environments, conda, ROS 2 and updates.

## Where to start

[How it works](tutorials/introduction.md) explains the idea, and
[your first controller](tutorials/first-controller.md) puts it to work on the soft arm. The
[tutorials](tutorials/coordinates-and-components.md) then go through the library step by step,
up to building your own robot, and the examples apply it to complete tasks on ready-made robots:

- [a soft arm](examples/soft-arm.md) that reaches a point, avoids an obstacle and changes shape;
- [two soft arms](examples/two-arms.md) that squeeze an object between them;
- [a finger](examples/finger.md) with a stiff fingertip and soft joint limits;
- [a hand](examples/hand.md) that grasps, alone or mounted on a UR5 arm;
- [a crawling turtle](examples/turtle.md) whose two cranks follow a virtual flywheel.

## Authors

`virtualmodelcontrol` is developed by Lorenzo Vignoli at the [CREATE Lab](https://www.epfl.ch/labs/create/) of EPFL (Prof.
Josie Hughes), in a collaboration with the University of Cambridge (Prof. Fulvio Forni). To cite it, use the
`CITATION.cff` file of the [repository](https://github.com/vigno0405/VirtualModelControl).

```{toctree}
:hidden:
:caption: Getting started

installation
tutorials/introduction
tutorials/first-controller
troubleshooting
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/coordinates-and-components
tutorials/parameters
tutorials/energy
tutorials/kinematics
tutorials/tuning
tutorials/contact
tutorials/build-a-robot
tutorials/extend
```

```{toctree}
:hidden:
:caption: Examples

examples/soft-arm
examples/two-arms
examples/finger
examples/hand
examples/turtle
```

```{toctree}
:hidden:
:caption: Concepts

concepts/library
concepts/pcc
concepts/conventions
```

```{toctree}
:hidden:
:caption: Reference

api/index
development/changelog
Roadmap <development/roadmap>
development/contributing
```
