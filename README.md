# VirtualModelControl

[![PyPI](https://img.shields.io/pypi/v/virtualmodelcontrol)](https://pypi.org/project/virtualmodelcontrol/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://vigno0405.github.io/VirtualModelControl/)
[![Python](https://img.shields.io/pypi/pyversions/virtualmodelcontrol)](https://pypi.org/project/virtualmodelcontrol/)

`virtualmodelcontrol` is a Python library for Virtual Model Control. You build a controller by
attaching virtual springs, dampers and masses to your robot, and the library turns them into
motor torques at the control rate, in simulation and on the real robot.

![A soft arm pulled by a virtual spring, and a finger pressing on a table with 1 N](https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/readme-hero.png)

## Install

```bash
pip install virtualmodelcontrol
```

The newest code, from GitHub, before it is released:

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

Python 3.10 or newer, on Linux, macOS or Windows; pip installs every dependency. The same two
lines update an existing install. Virtual environments, conda, ROS 2 and pinned versions are
covered in [Installation](https://vigno0405.github.io/VirtualModelControl/installation.html).

## A first controller

The robot and the controller are two separate mechanisms: the robot mechanism describes the
hardware, and the controller holds the virtual elements placed on it.

```python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")  # a soft arm with nine tendon motors
tip = arm.point(s=1.0)

ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - [0.1, 0.0, 0.6], 300.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))

meas = vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.zeros(9))
torques = controller.step(0.0, meas)["motor_torque"]  # [N·m], one per motor
```

The same controller runs on a simulation built from the robot's own masses, stiffness and
damping:

```python
arm = helyx.add_dynamics(arm)
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=2.0)
```

## Robots included

Each template takes its geometry as arguments, so the same function builds variants of the
robot.

| Robot | Template |
| --- | --- |
| Helyx soft arm, three geometries | `robots.helyx.arm()` |
| Two Helyx arms on one frame | `robots.bimanual.arms()` |
| ADAPT finger and hand | `robots.adapt.finger()`, `robots.adapt.hand()` |
| UR5, alone or carrying the hand | `robots.ur5.arm()`, `robots.ur5.with_hand()` |
| Crawling turtle (two cranks, a virtual flywheel) | `robots.turtle.robot()`, `turtle.controller()` |

## Documentation

**[vigno0405.github.io/VirtualModelControl](https://vigno0405.github.io/VirtualModelControl/)**:
tutorials, an example for each robot, the concepts and the API. What changed between versions
is in [CHANGELOG.md](CHANGELOG.md), what comes next in [TODO.md](TODO.md).

## Develop

```bash
git clone https://github.com/vigno0405/VirtualModelControl.git
cd VirtualModelControl
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
.venv/bin/python -m pytest
```

The checks, the documentation rules and the release steps are in
[Contributing](https://vigno0405.github.io/VirtualModelControl/development/contributing.html).

## Authors

Lorenzo Vignoli, in a collaboration between EPFL (Prof. Josie Hughes) and the University of
Cambridge (Prof. Fulvio Forni). To cite the library, see [CITATION.cff](CITATION.cff). The code
is under the [MIT license](LICENSE).
