# Installation

The library needs Python 3.10 or newer and runs on Linux, macOS and Windows. pip installs
everything it uses: numpy, SciPy, CasADi, matplotlib, imageio-ffmpeg (to write videos) and
PyYAML.

## With pip

```bash
pip install virtualmodelcontrol
python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

The second line prints the installed version.

On a recent Ubuntu or Debian (Ubuntu 24.04, for example), pip refuses to install into the system
Python and stops with `error: externally-managed-environment`. That is not a fault of the library:
use the virtual environment of the next section.

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
conda create -n vmc -c conda-forge --override-channels python=3.12 -y
conda activate vmc
pip install virtualmodelcontrol
python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

conda installs Python, and pip installs the library and everything it uses into the environment.
The channel is conda-forge on purpose. On a new installation, conda's default channels refuse to
create an environment (`CondaToSNonInteractiveError`) until you accept Anaconda's terms of
service. conda-forge asks for nothing. If ROS 2 is sourced in your shell, set the environment up
as shown in the next section.

## With ROS 2 in the same shell

The library never needs ROS. If ROS 2 is sourced in your shell, though, `PYTHONPATH` points at
ROS's own Python packages. pip then takes them for installed: it prints an error about their
dependencies, such as `generate-parameter-library-py requires jinja2`, although the library itself
is installed (`Successfully installed ...` follows). The error is confusing, and it is avoidable. They also come first on
the import path, so they can shadow the versions in your environment. Unset `PYTHONPATH` for the
commands that install or run the library:

```bash
env -u PYTHONPATH ~/venvs/vmc/bin/pip install virtualmodelcontrol
env -u PYTHONPATH ~/venvs/vmc/bin/python -c "import virtualmodelcontrol as vmc; print(vmc.__version__)"
```

In a conda environment, empty `PYTHONPATH` for the environment once, before you first activate
it. conda gives ROS's value back when you leave the environment. Each time you enter it, conda
prints a warning about overwriting `PYTHONPATH`: that warning is the empty value taking effect.

```bash
conda create -n vmc -c conda-forge --override-channels python=3.12 -y
conda env config vars set PYTHONPATH= -n vmc
conda activate vmc
pip install virtualmodelcontrol
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

Every commit on `main` has its own version number, such as `1.2.2.dev3+g1a2b3c4d5`, so pip
notices a newer commit and reinstalls. The output then ends with
`Successfully installed virtualmodelcontrol-...`. If nothing changed on `main`, pip lists the
dependencies as already satisfied and installs nothing.

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
pip install "virtualmodelcontrol==1.2.1"
```

In a `requirements.txt` file, write `virtualmodelcontrol==1.2.1`. In a conda `environment.yml`,
list it under `pip:`. `pip show virtualmodelcontrol` tells you which version you have, and the
[changelog](development/changelog.md) what changed between versions.

## Uninstall

```bash
pip uninstall virtualmodelcontrol
```

## Run the examples

The tutorials are notebooks: each page is a sequence of code cells that you run in order, in a
Jupyter notebook, in IPython, or pasted into one script. In a notebook or IPython, a cell shows the
value of its last line. In a script, wrap what you want to see in `print(...)`, and end a plot with
`plt.show()` or `plt.savefig("figure.png")`. Videos (`vmc.viz.animate`) are written to the folder
you run from, and any video player opens them. Run the code with the environment's Python (for
the venv above, `~/venvs/vmc/bin/python`).
