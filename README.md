<a href="https://www.epfl.ch/labs/create/"><img align="right" height="72" src="https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/create-lab.png" alt="CREATE Lab, EPFL"></a>

# VirtualModelControl

`virtualmodelcontrol` is a Python library for Virtual Model Control. You build a controller by
attaching virtual springs, dampers and masses to your robot, and the library turns them into
motor torques at the control rate, in simulation and on the real robot.

<p align="center">
  <a href="https://vigno0405.github.io/VirtualModelControl/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/logo-light.svg"><img src="https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/logo.svg" height="64" align="middle" alt="virtualmodelcontrol logo"></picture></a>
  &nbsp;
  <a href="https://vigno0405.github.io/VirtualModelControl/"><img src="https://img.shields.io/badge/Documentation-tutorials%20%C2%B7%20examples%20%C2%B7%20API-3c5488?style=for-the-badge&labelColor=13294b" align="middle" alt="Documentation: tutorials, examples, API"></a>
</p>

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

Python 3.10 or newer, tested with 3.10 to 3.14 on Ubuntu 22.04 and 24.04. Nothing to compile;
ROS and LaTeX are not needed.

## Dependencies

pip installs them with the library:

| Package | Version | Used for |
| --- | --- | --- |
| [numpy](https://numpy.org) | 1.26 or newer, 2.x included | arrays |
| [CasADi](https://web.casadi.org) | 3.6 or newer | the symbolic models, their derivatives and the compiled controllers |
| [SciPy](https://scipy.org) | 1.11 or newer | rotations of chains built from DH tables; fitting stiffness and damping |
| [matplotlib](https://matplotlib.org) | 3.8 or newer | figures and animations |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | 0.5 or newer | MP4 videos of the animations |
| [PyYAML](https://pyyaml.org) | 6 or newer | robot and hardware files |
| [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) | 3.7 or newer | Dynamixel motors |
| [pyserial](https://github.com/pyserial/pyserial) | 3.5 or newer | the serial port of the Dynamixel SDK |

## The example in the figure

The robot and the controller are two separate mechanisms: the robot mechanism describes the
hardware, and the controller holds the virtual elements placed on it.

```python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-145-145")  # a soft arm with nine tendon motors
tip = arm.point(s=1.0)

ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - [0.15, 0.0, 0.35], 600.0))  # [N/m]
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

- [Soft arm][soft-arm]: reach past an obstacle, and limit the force with a tanh spring.
- [Hanging soft arm][hanging-arm]: identify its stiffness and damping, and reach around an
  obstacle.
- [Two arms][two-arms]: squeeze an object between the tips of two soft arms.
- [Finger][finger]: a stiff fingertip and soft joint limits.
- [Hand][hand]: grasp a ball, alone or mounted on a UR5 arm.
- [Turtle][turtle]: two cranks that follow a virtual flywheel.

## Documentation

The [documentation][docs] holds the tutorials, the examples, the concepts and the API. What
changed is in the [changelog][changelog], what comes next in the [roadmap][roadmap], and how to
contribute in [Contributing][contributing].

## Authors

Lorenzo Vignoli, at the [CREATE Lab][lab] of EPFL (Prof. Josie Hughes), in a collaboration
with the University of Cambridge (Prof. Fulvio Forni). To cite the library, see [CITATION.cff][cite].
The code is under the [MIT license][license].

[lab]: https://www.epfl.ch/labs/create/
[install]: https://vigno0405.github.io/VirtualModelControl/installation.html
[figure]: https://github.com/vigno0405/VirtualModelControl/blob/main/docs/scripts/readme_figure.py
[soft-arm]: https://vigno0405.github.io/VirtualModelControl/examples/soft-arm.html
[hanging-arm]: https://vigno0405.github.io/VirtualModelControl/examples/hanging-arm.html
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
