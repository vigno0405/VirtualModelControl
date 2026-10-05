# Installation

The library needs Python 3.10 or newer and runs on Linux, macOS and Windows. pip installs
everything it uses: numpy, SciPy, CasADi, matplotlib, imageio-ffmpeg (to write videos), PyYAML,
the Dynamixel SDK and pyserial (the last two leave in 0.4.0).

## With pip

```bash
pip install virtualmodelcontrol
python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

The second line prints the installed version.

## In a virtual environment

A virtual environment keeps the library and its dependencies apart from the rest of your
system. On Linux and macOS:

```bash
python3 -m venv ~/venvs/vmc
~/venvs/vmc/bin/pip install virtualmodelcontrol
~/venvs/vmc/bin/python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

On Ubuntu (22.04 and 24.04 alike), `python3 -m venv` needs the `python3-venv` package. Install
it once:

```bash
sudo apt install python3-venv
```

On Windows, in PowerShell:

```powershell
py -m venv $HOME\venvs\vmc
& $HOME\venvs\vmc\Scripts\pip install virtualmodelcontrol
& $HOME\venvs\vmc\Scripts\python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

## With conda

```bash
conda create -n vmc python=3.12 -y
conda activate vmc
pip install virtualmodelcontrol
```

## With ROS 2 in the same shell

The library never needs ROS. If ROS 2 is sourced in your shell, though, `PYTHONPATH` points at
ROS's own Python packages, and they can shadow the versions in your environment. Unset it for
the commands that install or run the library:

```bash
env -u PYTHONPATH ~/venvs/vmc/bin/pip install virtualmodelcontrol
env -u PYTHONPATH ~/venvs/vmc/bin/python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

## The development version

Between releases, the newest code is on the `main` branch on GitHub. pip installs it directly:

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

## Update

An installed copy does not update by itself. Run the line that matches how you installed it.
From PyPI, the latest release:

```bash
pip install --upgrade virtualmodelcontrol
```

From GitHub, the newest `main`:

```bash
pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"
```

Every commit on `main` has its own version number, such as `0.1.1.dev9+gb8695c46f`, so pip
notices a newer commit and reinstalls; the output then ends with `Successfully installed
virtualmodelcontrol-...`. If nothing changed on `main`, pip lists the dependencies as already
satisfied and installs nothing.

From a clone of the repository installed with `pip install -e .`, pull: the change is live at
once. Reinstall only when the dependencies changed.

```bash
cd VirtualModelControl
git pull
pip install -e .
```

## A fixed version

An experiment should always run the same code. Pin the version, and move the pin on purpose:

```bash
pip install "virtualmodelcontrol==0.1.0"
```

In a `requirements.txt` file, write `virtualmodelcontrol==0.1.0`; in a conda `environment.yml`,
list it under `pip:`. `pip show virtualmodelcontrol` tells you which version you have, and the
[changelog](development/changelog.md) what changed between versions.

## Uninstall

```bash
pip uninstall virtualmodelcontrol
```
