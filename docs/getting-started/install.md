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

## ROS 2

The library never needs ROS. Its optional ROS 2 parts use only `rclpy` and `std_msgs`, so any
ROS 2 distribution works. ROS comes from the system's ROS install, not from pip.

When ROS is sourced in your shell, `PYTHONPATH` points at ROS's own Python packages. Keep them
out of the library's environment for installs and tests:

```bash
env -u PYTHONPATH ~/venvs/vmc/bin/python -c "import virtualmodelcontrol"
```
