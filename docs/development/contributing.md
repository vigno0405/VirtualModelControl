# Contributing

## Set up

Linux or macOS, from a clone of the repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
```

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`); a conda environment works too. If ROS 2 is sourced in your
shell, prefix every command on this page with `env -u PYTHONPATH`.

## Check your change

Run these from the repository root before opening a pull request:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/lint-imports
.venv/bin/sphinx-build -W -b html docs docs/_build/html
```

`pre-commit run --all-files` runs the formatters and file checks in one go. It is not
installed as a git hook.

## Rules

- **SI units** inside the library. Unit conversions belong in the hardware layers.
- **No numbers in model code.** Geometry, gains, masses and calibration are `Param`s; their
  defaults live in robot data.
- **CasADi is the only symbolic source.** No second copy of a model in numpy or sympy.
- **Short docstrings:** numpydoc, a 1–3 line summary, units on physical parameters. Theory goes
  in the docs.
- **Ported code** comes with a golden regression test: a small `.npz` fixture in `tests/data/`
  generated from the original implementation, plus the property tests that apply.
- **Behaviour changes are opt-in.** Projects pin library versions; deprecate for one minor
  version before removing anything.
- Add a line to `CHANGELOG.md` under "Unreleased".

## Release

Releases follow semantic versioning (0.x for now). The version comes from git tags: pushing a
tag `vX.Y.Z` builds the package and creates a GitHub release.
