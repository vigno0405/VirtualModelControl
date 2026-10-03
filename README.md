# VirtualModelControl

`virtualmodelcontrol` is a Python library for Virtual Model Control (VMC). Robots and
controllers are described as mechanisms: coordinates plus springs, dampers, inertances and
sources. One CasADi model serves the real-time controller, simulation and optimization.

A collaboration between EPFL (Prof. Josie Hughes) and the University of Cambridge
(Prof. Fulvio Forni).

## Install

Python ≥ 3.10 on Linux, macOS or Windows:

```bash
pip install virtualmodelcontrol
```

Then `import virtualmodelcontrol as vmc`. The latest development version installs with
`pip install "git+https://github.com/vigno0405/VirtualModelControl.git"`.

## Develop

On Linux or macOS (Windows: see the install page in the docs):

```bash
git clone https://github.com/vigno0405/VirtualModelControl.git
cd VirtualModelControl
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
.venv/bin/python -m pytest
```

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`); a conda environment works too. If ROS 2 is sourced in your
shell, prefix these commands with `env -u PYTHONPATH` so ROS's Python packages stay out of the
environment.

## Documentation

https://vigno0405.github.io/VirtualModelControl/

To build it locally: `sphinx-build docs docs/_build/html`, then open
`docs/_build/html/index.html`.

## Citing

See `CITATION.cff`.
