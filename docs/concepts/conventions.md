# Conventions and glossary

The rules that every part of the library follows, and the words it uses. Read it once, and come back when a sign or a unit surprises you.

## Units and signs

- SI everywhere: m, rad, s, N, N·m, kg. Degrees stay at the boundary with the hardware, in the
  robot's hardware profile. A driver's raw units (encoder ticks, motor currents) never enter
  the library.
- A positive motor angle pulls its tendon. Each template states the sign of its robot's
  encoders against this convention (`helyx.ENCODER_SIGN`, one sign for each geometry,
  `bimanual.ENCODER_SIGN`, one sign, and `turtle.MOTOR_SIGNS`, one sign per motor). The robot's
  hardware profile applies it for you; if you convert by hand, multiply raw readings and commands
  by it.
- Controllers send their torques as computed, never divided by an efficiency. A robot's
  efficiency maps the commanded motor torques to the delivered ones. It is 1 by default,
  because the templates' stiffness and damping were identified from the commanded torques
  (see [Transmission efficiency](efficiency.md)).

## Coordinates and forces

- The configuration `q` lives on a space, and the velocity `v` in its tangent space. So `q` and
  `v` may differ in size (a floating body, for example).
- A generalized force is `τ`, actuator commands are `u`. Symbols are local to a page, and each page
  says what it uses. The one to watch is `θ`: a motor angle in the robot's signals
  (`motor_position`); a joint angle in the finger and hand pages, where the motors are `q`; the
  bend of a segment in the PCC page.
- A spring's deflection is `y = x − x_ref`, so it pulls `x` towards `x_ref`. Storage gives
  `f = −∂V/∂y`, a damper `f = −D ẏ`, and every force reaches the robot as `τ = Jᵀ f`.
- Arrays in, arrays out: inputs accept anything array-like; outputs are numpy arrays (torch
  tensors from a model exported with `backend="torch"`).

## Continuum robots

- The arc parameter `s ∈ [0, 1]` runs from the base to the tip, uniformly in rest length (the arc
  length while no segment stretches).
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

inertance
  A mass or an inertia on a coordinate, and the component that gives it. A virtual state needs
  one; a robot has its own.

live Param
  A Param that may change while the controller runs (scope `stage`), for example with
  `controller.set`; the others are folded in when the controller is compiled
  ([parameters](../tutorials/parameters.md)).

law
  A rule that changes a controller's live Params at every step, such as `ForceTracking`.

guard
  The check of a run that sends zero torque when a reading is missing, not a number, or too old.

site
  A named point of a kinematic model, such as `"tip"`; continuum models also accept an arc
  parameter `s`.

plant
  What the controller drives: a simulator such as `ModelPlant`, or the real robot. In a plan,
  `Problem(system, plant=robot)` plans the motion of a larger `robot` than the controller's own
  (see [Optimize](../tutorials/optimize.md)).

tank
  An energy budget for changing a running controller: a change is applied as far as the tank
  pays for the energy it gives the controller, and what the controller's dampers take refills it.

output stage
  An optional correction of the motor commands for real hardware: friction compensation,
  pretension, an offset or a torque limit.

template
  A function that builds a ready-made robot, such as `helyx.arm`, with every geometric number
  as an argument and a Param.
```
