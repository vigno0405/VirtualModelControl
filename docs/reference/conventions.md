# Conventions

## Units

SI everywhere: m, rad, s, N, N·m, kg. Degrees, grams, encoder ticks and motor current appear
only at the hardware boundary (the `hardware` and `ros` layers).

## Coordinates and forces

- The configuration `q` lives on a {term}`space <Space>`; the velocity `v` lives in its
  tangent space, so `nq` may differ from `nv` (a floating base, for example).
- `q_m` are motor angles, `τ` is a generalized force, `u` is an actuator command.
- Delivered torque = η · commanded torque. Calibration files state which of the two their
  parameters refer to.

## Components

- **Storage:** deflection `y = x − x_ref`, energy `V(y) ≥ 0`, force `f = −∂V/∂y`.
- **Dissipation:** `f · ẏ ≤ 0`.
- **Sources** are metered: their power enters the energy accounting.
- Forces act on the robot through the coordinate's Jacobian: `τ = Jᵀ f`.

## Tendons

`q_m > 0` pulls the tendon (it shortens). Each robot declares the sign of its encoders against
this convention in its hardware profile.

## Continuum robots

- The arc parameter `s ∈ [0, 1]` is global and uniform in arc length (0 at the base, 1 at the
  tip).
- Piecewise constant curvature (PCC) uses `Δ = [Dx, Dy, Dl]` per segment, base to tip.
- The base frame's z axis runs along the straight body.
- Gravity is a vector in the base frame.

## Parameters

Every continuous number is a `Param` with a unit, a default and bounds: gains, references,
attachment points, all geometry, masses, transmissions, motor constants. Structural integers
(numbers of segments, tendons or joints) are constructor arguments. Defaults live in robot
data and calibration files, never in model code.

A parameter's *scope* is the time scale on which it may change:

`fixed`
: never changes (a physical constant).

`design`
: changes with the hardware design (geometry, masses).

`episode`
: changes between runs (attachment points, element sets).

`stage`
: may change at every control step (gains, references).

The compiled controller takes `stage` parameters as live inputs and folds the others in as
constants; changing one of those means compiling again.

## Arrays

Inputs accept anything array-like; outputs are 1-D numpy arrays. CasADi objects stay inside
unless asked for.

## Code

- `snake_case` functions and variables, `CapWords` classes. Robot names appear only in robot
  data modules.
- Docstrings follow numpydoc, with a 1–3 line summary and units on every physical parameter.
  Derivations belong in the theory pages.

```{glossary}
Space
  The manifold a configuration lives on, with its tangent space for velocities.
```
