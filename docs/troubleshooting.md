---
file_format: mystnb
kernelspec:
  name: python3
---

# Troubleshooting

## Import errors with ROS 2 sourced

When ROS 2 is sourced, `PYTHONPATH` points at ROS's own Python packages, which can shadow numpy,
matplotlib or other packages of your environment and break imports. Unset it for the commands
that install or run the library:

```bash
env -u PYTHONPATH ~/venvs/vmc/bin/python -c "import virtualmodelcontrol"
```

For a conda environment, [Installation](installation.md) shows how to do it once and for all.

## No virtual environment on Ubuntu

`python3 -m venv` fails with "ensurepip is not available" until the `python3-venv` package is
installed:

```bash
sudo apt install python3-venv
```

## CasADi does not import

pip installs CasADi from a prebuilt wheel. If pip starts building CasADi from source, or
`import casadi` fails, there is no wheel for your Python version or platform. Create the
environment with a Python version listed among the files on CasADi's
[PyPI page](https://pypi.org/project/casadi/#files).

## Compiling takes long

Compiling a controller takes a fraction of a second for the robots in this library, and a
control step about 20 µs. A large model compiles slower. Compile once, and change gains and
goals while running with `controller.set`, which needs no new compile. Only Params that are not
live (geometry, masses, attachment points) need a new `compile` after a change; see
[parameters](tutorials/parameters.md).

## Figures and LaTeX

`viz.use_style()` sets text with LaTeX when `latex` and `dvipng` are installed. If LaTeX then
fails on a missing package, install the fonts and packages matplotlib needs, or use the style
without LaTeX, which keeps the Computer Modern look:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng
```

```{code-cell} python
from virtualmodelcontrol import viz

viz.use_style(usetex=False)
```

matplotlib 3.11 drops minus signs from figures saved as PDF with LaTeX; use matplotlib 3.10 for
those until it is fixed.

## Saving videos

`viz.animate` writes MP4 with the ffmpeg that pip installs with `imageio-ffmpeg`. On a platform
without that wheel, install ffmpeg with your package manager, or save a `.gif` or `.webp`
instead.
