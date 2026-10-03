# VirtualModelControl

[![PyPI](https://img.shields.io/pypi/v/virtualmodelcontrol)](https://pypi.org/project/virtualmodelcontrol/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://vigno0405.github.io/VirtualModelControl/)
[![Python](https://img.shields.io/pypi/pyversions/virtualmodelcontrol)](https://pypi.org/project/virtualmodelcontrol/)

**Control robots by attaching virtual springs, dampers and masses to them.**

`virtualmodelcontrol` is a Python library for Virtual Model Control (VMC). You describe your robot
once, place virtual elements where they should act (a spring pulling a fingertip to a goal, a
repulsive field around an obstacle, a damper along a soft arm), and the library turns them into
motor torques at control rate, in simulation and on the real robot.

![A soft arm pulled by a virtual spring, and a finger pressing on a table with 1 N](https://raw.githubusercontent.com/vigno0405/VirtualModelControl/main/docs/_static/readme-hero.png)

A collaboration between EPFL (Prof. Josie Hughes) and the University of Cambridge
(Prof. Fulvio Forni).

## What it does

- **Any robot:** rigid chains (from DH or product-of-exponentials data), soft continuum arms
  (piecewise constant curvature), tendon drives, coupled joints, robots mounted on robots.
- **Controllers as mechanisms:** springs (linear, saturating, repulsive, joint limits, contact),
  dampers, virtual masses, sources and gravity compensation, attached to points, joints, distances
  or any expression you write.
- **Fast:** one compiled CasADi function per controller, about 20 µs per control step on a
  three-segment soft arm.
- **Exact energy bookkeeping:** stored energy, dissipated power and source power of every element,
  and the exact energy jump when you change a gain while running.
- **Simulation from the same description:** the robot's own masses, stiffness and damping give
  its dynamics, with implicit steps that stay stable for stiff contacts.
- **Kinematics with derivatives:** positions, rotations, Jacobians and Hessians of any point, by
  automatic differentiation.
- **Ready-made robots:** the Helyx soft arm, the bimanual Helyx, the ADAPT finger and hand, a
  crawling turtle, and the UR5 (alone or carrying the hand).
- **Hardware details as opt-in stages:** friction compensation, pretension, torque offsets,
  efficiency correction, torque limits.

Planned next (hardware and ROS 2 drivers, URDF and meshes, numpy and sympy exports, simulators,
identification, estimation, energy tanks, optimization, MPC, learning): see [TODO.md](TODO.md).

## Install

Python 3.10 or newer, on Linux, macOS or Windows:

```bash
pip install virtualmodelcontrol
```

This installs every dependency (numpy, SciPy, CasADi, matplotlib, PyYAML, the Dynamixel SDK and
pyserial). Check it with:

```bash
python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

On Debian and Ubuntu, a new virtual environment needs the `python3-venv` package
(`sudo apt install python3-venv`); a conda environment works too:

```bash
conda create -n vmc python=3.12 -y
conda activate vmc
pip install virtualmodelcontrol
```

If ROS 2 is sourced in your shell, prefix pip and python with `env -u PYTHONPATH`, so ROS's own
Python packages stay out of the environment.

## Update

An installed copy does not update by itself. Pick the line that matches how you installed it.

**From PyPI** (the latest release):

```bash
pip install --upgrade virtualmodelcontrol
```

**The current development state** (the `main` branch on GitHub, newer than the last release):

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

Every commit on `main` has its own version number (for example `0.1.1.dev12+g3a4b5c6`), so pip
sees the change and reinstalls. Running the same line again later updates you to the newest
`main`. If pip ever answers "Requirement already satisfied" although `main` has changed, add
`--force-reinstall --no-deps`.

**From a clone you develop in** (an editable install, `pip install -e .`): pull, and the change is
live; reinstall only when the dependencies changed.

```bash
cd VirtualModelControl
git pull
pip install -e .
```

**A fixed version**, so an experiment always runs with the same code: pin it, and move the pin
on purpose.

```bash
pip install "virtualmodelcontrol==0.1.0"
```

In a `requirements.txt`, write `virtualmodelcontrol==0.1.0`; in a conda `environment.yml`, list it
under `pip:`. To see what you have, run `pip show virtualmodelcontrol`; to remove it,
`pip uninstall virtualmodelcontrol`.

New releases reach PyPI when a version tag (for example `v0.2.0`) is pushed to GitHub: the release
workflow builds the package and uploads it. What changed in each release is in
[CHANGELOG.md](CHANGELOG.md).

## A first controller

```python
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.arm("145-290-290")  # a ready-made robot
ctrl = vmc.Mechanism("ctrl")  # the controller: a virtual mechanism
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - [0.1, 0.0, 0.6], 30.0))  # [N/m]
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 1.5))  # [N·s/m]
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))

meas = vmc.Signals(0.0, motor_position=[0.0] * 9, motor_velocity=[0.0] * 9)
torques = controller.step(0.0, meas)["motor_torque"]  # 9 motor torques [N·m]
```

The same controller runs in simulation:

```python
arm = helyx.add_dynamics(arm)  # the arm's own stiffness, damping, gravity
plant = vmc.sim.ModelPlant(arm)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=3.0)
```

## Robots included

| Robot | Template | What it is |
| --- | --- | --- |
| Helyx soft arm | `robots.helyx.arm(geometry)` | three tendon-driven continuum segments, three geometries |
| Bimanual Helyx | `robots.bimanual.arms()` | two soft arms on one frame |
| ADAPT finger | `robots.adapt.finger()` | three phalanges, two motors, coupled distal joints |
| ADAPT hand | `robots.adapt.hand()` | five digits, 13 motors, spread coupling |
| Crawling turtle | `robots.turtle.robot()`, `turtle.controller()` | two cranks coordinated by a virtual flywheel |
| UR5 | `robots.ur5.arm()`, `ur5.with_hand()` | six-joint arm, alone or carrying the hand |

Every number in a template (lengths, radii, tendon angles, masses, gains) is a parameter you can
change or optimize.

## How the library is organized

| Package | What is in it |
| --- | --- |
| `core` | parameters with units and scopes, spaces, signals, the plugin registry |
| `mechanisms` | coordinates (points, joints, distances, references) and components (springs, dampers, masses, sources) |
| `models` | kinematic models: PCC continuum, serial chains, couplings, assemblies, actuation |
| `compile`, `VMCController` | turn a robot and a controller into one fast function |
| `compile_dynamics` | the robot's equations of motion from its components |
| `sim` | simulated plants and the run loop shared with the hardware |
| `viz` | figures in the lab style |
| `robots` | ready-made robots and their calibration data |
| `control.output` | opt-in output stages for real hardware |

A controller needs only `mechanisms`, `models` and `compile`; simulation and figures are separate
and optional.

## Documentation

**[vigno0405.github.io/VirtualModelControl](https://vigno0405.github.io/VirtualModelControl/)**: install, a first controller, concepts with
figures, guides for each robot, how to build your own robot, and the full API.

## Develop

```bash
git clone https://github.com/vigno0405/VirtualModelControl.git
cd VirtualModelControl
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
.venv/bin/python -m pytest
```

The checks CI runs:

```bash
cd VirtualModelControl
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/lint-imports
.venv/bin/sphinx-build -W -b html docs docs/_build/html
```

The built documentation opens from `docs/_build/html/index.html`. On Windows, use
`.venv\Scripts\` instead of `.venv/bin/`.

## Releasing

Update `CHANGELOG.md`, push `main`, then tag:

```bash
cd VirtualModelControl
git tag -a v0.2.0 -m "virtualmodelcontrol 0.2.0"
git push origin v0.2.0
```

The release workflow tests the tag, builds the package, publishes it to PyPI and creates the
GitHub release.

## Citing

See [CITATION.cff](CITATION.cff).

## License

See [LICENSE](LICENSE).
