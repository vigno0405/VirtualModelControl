# Architecture

## Vocabulary

A **mechanism** is a set of coordinates plus components acting on them. Every component is one
of four kinds:

- **storage:** has an energy `V(y)`; springs, gravity, elastic structure
- **dissipation:** dampers and friction
- **inertance:** masses, inertias, inerters
- **source:** metered forces and motions

This holds for the physical robot and for the virtual mechanism that controls it. A closed loop
is the union of both mechanisms, so the same objects simulate, optimize and control.

## Layers

A module imports only from layers below it. `import-linter` checks this in CI
(`lint-imports`), using the contracts in `pyproject.toml`.

| Layer | Packages | Role |
|---|---|---|
| L7 research | `codesign`, `learning` | optional extras |
| L6 decide | `optim`, `adaptation` | optimization, online tuning |
| L5 analyse | `estimation`, `identification`, `passivity` | |
| L4 run | `sim` | plants, model simulator, run loop |
| L3 control | `system`, `compiler`, `dynamics`, `control` | compiled controllers and robot dynamics |
| L2 describe | `models` | kinematics, continuum, rigid, actuation |
| L1 vocabulary | `mechanisms` | coordinates and components |
| L0 core | `core`, `io` | parameters, spaces, symbolic helpers, signals, registry |
| edges | `viz`, `hardware`, `ros`, `robots` | never imported by L0–L7 |

Packages appear as their release lands; the table is the target.

## Seams

Protocols exist only where implementations are swapped: model ↔ compiler, runtime ↔ hardware,
problem ↔ solver. Everything else is concrete classes.

## Symbolic core

All models are written once with CasADi symbols. The dynamics have one canonical form, the
implicit residual `M(q) a + h(q, v) − B(q) u − Σ Jᵀ f = 0`, so simulation, collocation,
estimation and identification share it.
