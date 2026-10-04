# VirtualModelControl

[![PyPI](https://img.shields.io/pypi/v/virtualmodelcontrol?color=3c5488)](https://pypi.org/project/virtualmodelcontrol/)
[![Python](https://img.shields.io/pypi/pyversions/virtualmodelcontrol?color=3c5488)](https://pypi.org/project/virtualmodelcontrol/)
[![Docs](https://img.shields.io/badge/docs-online-3c5488)](https://vigno0405.github.io/VirtualModelControl/)
[![License](https://img.shields.io/badge/license-MIT-3c5488)](https://github.com/vigno0405/VirtualModelControl/blob/main/LICENSE)

`virtualmodelcontrol` is a Python library for Virtual Model Control. You build a controller by
attaching virtual springs, dampers and masses to your robot, and the library turns them into
motor torques at the control rate, in simulation and on the real robot.

<p align="center">
  <img src="https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/readme-hero.svg" width="820" alt="A soft arm whose tip a virtual spring pulls towards a goal, and the distance from the tip to the goal over time">
</p>

## Install

```bash
pip install virtualmodelcontrol
```

The newest code, from GitHub, before it is released (the same line updates an existing install):

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

Virtual environments, conda, ROS 2 and pinned versions are covered in [Installation][install].

## Requirements

- Python 3.10 or newer, tested with 3.10 to 3.14 on Ubuntu 22.04 and 24.04.
- pip installs everything else: numpy (1.26 or 2.x), SciPy, CasADi, matplotlib, PyYAML and
  imageio-ffmpeg, and the Dynamixel SDK and pyserial for the hardware.
- Nothing to compile; ROS and LaTeX are not needed.

## The example in the figure

The robot and the controller are two separate mechanisms: the robot mechanism describes the
hardware, and the controller holds the virtual elements placed on it.

```python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")  # a soft arm with nine tendon motors
tip = arm.point(s=1.0)

ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - [0.25, 0.0, 0.55], 600.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))

meas = vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.zeros(9))
torques = controller.step(0.0, meas)["motor_torque"]  # [N·m], one per motor
```

The same controller runs on a simulation built from the robot's own masses, stiffness and
damping; [readme_figure.py][figure] draws the figure from this run.

```python
arm = helyx.add_dynamics(arm)
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=3.0)
```

## Examples

Each example builds a ready-made robot, controls it in simulation and animates the run:

- [Soft arm][soft-arm]: reach a point, avoid an obstacle, shape the whole arm.
- [Two arms][two-arms]: squeeze an object between the tips of two soft arms.
- [Finger][finger]: a stiff fingertip and soft joint limits.
- [Hand][hand]: grasp a ball, alone or mounted on a UR5 arm.
- [Turtle][turtle]: two cranks that follow a virtual flywheel.

## Documentation

The [documentation][docs] holds the tutorials, the examples, the concepts and the API. What
changed is in the [changelog][changelog], what comes next in the [roadmap][roadmap], and how to
contribute in [Contributing][contributing].

## Authors

Lorenzo Vignoli, at EPFL, in a collaboration between EPFL (Prof. Josie Hughes) and the
University of Cambridge (Prof. Fulvio Forni). To cite the library, see [CITATION.cff][cite].
The code is under the [MIT license][license].

[install]: https://vigno0405.github.io/VirtualModelControl/installation.html
[figure]: https://github.com/vigno0405/VirtualModelControl/blob/main/docs/scripts/readme_figure.py
[soft-arm]: https://vigno0405.github.io/VirtualModelControl/examples/soft-arm.html
[two-arms]: https://vigno0405.github.io/VirtualModelControl/examples/two-arms.html
[finger]: https://vigno0405.github.io/VirtualModelControl/examples/finger.html
[hand]: https://vigno0405.github.io/VirtualModelControl/examples/hand.html
[turtle]: https://vigno0405.github.io/VirtualModelControl/examples/turtle.html
[docs]: https://vigno0405.github.io/VirtualModelControl/
[changelog]: https://vigno0405.github.io/VirtualModelControl/development/changelog.html
[roadmap]: https://vigno0405.github.io/VirtualModelControl/development/roadmap.html
[contributing]: https://vigno0405.github.io/VirtualModelControl/development/contributing.html
[cite]: https://github.com/vigno0405/VirtualModelControl/blob/main/CITATION.cff
[license]: https://github.com/vigno0405/VirtualModelControl/blob/main/LICENSE
