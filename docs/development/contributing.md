# Contributing

## Set up

From a clone of the repository, on Linux or macOS:

```bash
cd VirtualModelControl
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
```

With ROS 2 sourced in the shell, prefix every command on this page with `env -u PYTHONPATH`.

## Check a change

These are the checks CI runs; all of them must pass:

```bash
cd VirtualModelControl
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/lint-imports
.venv/bin/sphinx-build -W -b html docs docs/_build/html
```

The site then opens from `docs/_build/html/index.html`.

## Code

- SI units inside the library; degrees only in the hardware profiles.
- No numbers in model code: every geometric or physical number is a `Param`, with its default
  in the robot's template, and templates take their geometry as arguments.
- Models are written once, with CasADi operations; no second copy in numpy or sympy.
- Docstrings are short (one to three lines, numpydoc, units on physical quantities); theory goes
  in the documentation.
- New behaviour is opt-in: projects pin a version, and a finished experiment must run the same
  after an update.
- Every change adds a line to `CHANGELOG.md` under "Unreleased".

## Tests

`tests/` mirrors `src/`. Derivatives are checked against finite differences and energies against
the power balance. Code ported from earlier implementations is checked against its recorded
results: the small `.npz` fixtures in `tests/data/`. `tests/test_readme.py` runs the README's
examples and `tests/test_docs.py` the documentation's house rules. Timing benchmarks run on
request:

```bash
cd VirtualModelControl
.venv/bin/python -m pytest -m bench
```

## Documentation

Pages are MyST notebooks: every code cell runs at each build, and an error fails it.

- Open a tutorial or example with one sentence on what we build; then, for each step, a task
  heading, a few short sentences, the code, the result.
- Keep pages essential: state each fact once and link to it elsewhere.
- Visible code lines are at most 76 characters.
- Numbers in the text come from the page's own computation, inserted with `glue`.
- Figures use `vmc.viz` (the lab style, set by `docs/docs_setup.py`); schematics are drawn from
  the robots' Params by the modules in `docs/schematics/`.
- A page that simulates shows its run with `viz.animate` and the `video` directive.

## Releases

Versions follow semantic versioning and come from git tags. To release:

1. In `CHANGELOG.md`, rename "Unreleased" to the new version and date.
2. Push `main`, then tag and push the tag:

   ```bash
   cd VirtualModelControl
   git tag -a v0.2.0 -m "virtualmodelcontrol 0.2.0"
   git push origin v0.2.0
   ```

The release workflow builds the package, publishes it to PyPI and creates the GitHub release.
The documentation is rebuilt and published whenever a push to `main` changes it or the code.
