# Conventions and glossary

## Units and signs

- SI everywhere: m, rad, s, N, N·m, kg. Degrees stay at the boundary with the hardware, in the
  robot's hardware profile. A driver's raw units (encoder ticks, motor currents) never enter
  the library.
- A positive motor angle pulls its tendon. Each template states the sign of its robot's
  encoders against this convention (`helyx.ENCODER_SIGN`, `bimanual.ENCODER_SIGN`,
  `turtle.MOTOR_SIGNS`); multiply raw readings and commands by it.
- Controllers send their torques as computed, never divided by an efficiency. A robot's
  efficiency maps the commanded motor torques to the delivered ones. It is 1 by default,
  because the templates' stiffness and damping were identified from the commanded torques
  (see [Transmission efficiency](efficiency.md)).

## Coordinates and forces

- The configuration `q` lives on a space, and the velocity `v` in its tangent space. So `q` and
  `v` may differ in size (a floating body, for example).
- Motor angles are `θ`, a generalized force is `τ`, actuator commands are `u`.
- A spring's deflection is `y = x − x_ref`, so it pulls `x` towards `x_ref`. Storage gives
  `f = −∂V/∂y`, a damper `f = −D ẏ`, and every force reaches the robot as `τ = Jᵀ f`.
- Arrays in, arrays out: inputs accept anything array-like; outputs are numpy arrays.

## Continuum robots

- The arc parameter `s ∈ [0, 1]` runs from the base to the tip, uniformly in arc length.
- A segment's configuration is `Δ = (Dx, Dy, Dl)` [m]. The base frame's `z` axis runs along
  the straight body. See [soft-arm kinematics](pcc.md).

## Parameters

Every continuous number is a `Param`: gains, goals, attachment points, every geometric number,
masses, transmissions. Integers that fix the structure (numbers of segments, joints or tendons)
are constructor arguments. A Param's scope is the time scale on which it may change: `fixed`
(never), `design` (with the hardware), `episode` (between runs), `stage` (at any control step).
`compile` keeps `stage` Params live and folds the others in; see [parameters](../tutorials/parameters.md).

## Glossary

```{glossary}
mechanism
  A set of coordinates and the components acting on them. The robot is one mechanism, its
  controller another.

coordinate
  A smooth function of the configuration, the virtual states, the Params and time: a point, a
  joint angle, a distance, a projection.

component
  An element acting on a coordinate: storage (springs, gravity), dissipation (dampers),
  inertance (masses) or source (force sources, gravity compensation).

virtual state
  A degree of freedom of the controller itself, such as the turtle's flywheel; it needs an
  inertance.

site
  A named point of a kinematic model, such as `"tip"`; continuum models also accept an arc
  parameter `s`.

plant
  What the controller drives: a simulator such as `ModelPlant`, or the real robot.

output stage
  An optional correction of the motor commands for real hardware: friction compensation,
  pretension, an offset or a torque limit.

template
  A function that builds a ready-made robot, such as `helyx.arm`, with every geometric number
  as an argument and a Param.
```
