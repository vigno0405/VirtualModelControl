# Install

The library needs Python ≥ 3.10 and runs on Linux, macOS and Windows. Its core depends only
on numpy, scipy and CasADi.

## Into an existing environment

```bash
pip install virtualmodelcontrol
```

The latest development version, straight from GitHub:

```bash
pip install "git+https://github.com/vigno0405/VirtualModelControl.git"
```

## Into a new virtual environment

Linux and macOS:

```bash
python3 -m venv ~/venvs/vmc
~/venvs/vmc/bin/pip install virtualmodelcontrol
~/venvs/vmc/bin/python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

Windows (PowerShell):

```powershell
py -m venv $HOME\venvs\vmc
& $HOME\venvs\vmc\Scripts\pip install virtualmodelcontrol
& $HOME\venvs\vmc\Scripts\python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`). A conda environment works the same way: create one with any
Python ≥ 3.10, activate it and run `pip install virtualmodelcontrol`.

## Update

An installed copy does not update by itself. Pick the line that matches how you installed it.

The latest release, from PyPI:

```bash
pip install --upgrade virtualmodelcontrol
```

The current development state (the `main` branch on GitHub, newer than the last release):

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

Every commit on `main` has its own version number, so pip sees the change and reinstalls. If pip
answers "Requirement already satisfied" although `main` has changed, add
`--force-reinstall --no-deps`.

A clone installed with `pip install -e .`: pull, and the change is live (reinstall only when the
dependencies changed):

```bash
cd VirtualModelControl
git pull
pip install -e .
```

A fixed version, so an experiment always runs the same code:

```bash
pip install "virtualmodelcontrol==0.1.0"
```

Check what you have with `pip show virtualmodelcontrol`, remove it with
`pip uninstall virtualmodelcontrol`. New releases reach PyPI when a version tag is pushed to
GitHub; the changes are listed in the {doc}`changelog <../reference/changelog>`.

## ROS 2

The library never needs ROS. Its optional ROS 2 parts use only `rclpy` and `std_msgs`, so any
ROS 2 distribution works. ROS comes from the system's ROS install, not from pip.

When ROS is sourced in your shell, `PYTHONPATH` points at ROS's own Python packages. Keep them
out of the library's environment for installs and tests:

```bash
env -u PYTHONPATH ~/venvs/vmc/bin/python -c "import virtualmodelcontrol"
```
