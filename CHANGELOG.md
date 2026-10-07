# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may change between minor versions).

## [1.0.0] - 2026-10-07

The first stable release. It also carries the work meant for 0.6.0 (planning with virtual states,
unmeasured coordinates, custom dynamics, the planner on a larger plant, MPC and underactuation) and
for 0.7.0 (models as numpy and PyTorch code), which were never released on their own, and it adds
the checking: the whole suite on the oldest and the newest supported Python and numpy, on Linux,
macOS and Windows, every page of the documentation run from a clean checkout, and a review of the
public API (what `__all__` lists in each module is the API) and of every page of the docs at desktop
and phone width. The rules for what changes between versions are on the page "Support policy".

### Added

- `optimization.Shooting(q0, horizon, nodes, v0=, steps=, initial=, transition=, scales=,
  integrator=, substeps=)`: multiple shooting, a second transcription next to `Collocation`
  with the same terms. The unknowns are q and v at the nodes; over each interval the closed
  loop is simulated as `vmc.sim.rollout` does (the command held over each of `substeps` control
  steps, the robot advanced by the linearly implicit step or by `"rk4"`), and constraints join
  the end of an interval to the next node. A plan is the closed loop of a simulation at that
  control step (a test reproduces `rollout` to rounding error), and agrees with `Collocation` as
  the control step shrinks (first order). The start is the pair of Params `shooting.q0` and
  `shooting.v0`, which a problem can declare as parameters. `steps` names live Params that take
  one value in every interval, between their bounds: trajectories of stiffness and references,
  returned in `Result.steps` (`apply(target, interval=k)` sets the values of interval k). The
  model does not carry the force that balances the robot at q0 under `initial`, as `Collocation`
  does.
- `optimization.TankBudget(level, refill=True)`: the planned changes of the Params that step stay
  within the energy of a `vmc.control.Tank`. The plan keeps the tank's level from the exact jump
  of the controller's energy at each change, and from the energy the controller's dampers take
  back, and keeps it at least zero; a test runs the plan through a real `Tank` and finds its level
  to rounding error. The tank's capacity is not planned.
- `optimization.MPC(problem, shift=1, rti=False)`: the plan as a receding-horizon controller. The
  program is built once, with the measured state, the controller's Params now, the tank's level
  and the references as parameters of each solve. `step(controller, q, v, references, latency=)`
  solves from the measured state, starting from the previous plan moved up by `shift` intervals,
  and applies the plan's Params through `controller.set` (so a `Tank` bounds them), returning
  the energy jump as the adaptation laws do. A plan that took `latency` seconds (the measured
  solve time by default) is applied as of that time. With `rti`, one SQP iteration per step.
  `start` and `poll` plan in a thread and apply the plan once it is ready, as of the time since
  the start (CasADi lets other threads run during a solve; the solver is created with the `MPC`,
  as it is not safe to create one in a second thread). `Problem.warm_up()` builds the program and
  the solver now, and `Problem.solve` attaches the iteration callback only when it is given a
  `progress`.
- Planning with virtual states: `Collocation`, `Shooting` and `Equilibrium` accept controllers
  with virtual states (a flywheel, a mass the robot is tied to). The states are unknowns of the
  plan at every node, with the controller's own dynamics: collocated with the same scheme as the
  robot (`Collocation`), or advanced as `VMCController.step` and `vmc.sim.rollout` advance them
  (`Shooting`, which then reproduces `rollout` to rounding error, the states included). They
  start from the state the controller is compiled with, or from `z0` (positions, then
  velocities); a periodic motion repeats them; `Equilibrium` rests them. `Result.z` holds them,
  `warm_start` takes them, `TankBudget` reads the energy at them, and the controller in place
  (`initial`) may have states too, as a system (it starts with the plan) or as a running
  controller (its state is where it is now). `MPC` plans a running controller from
  `controller.z` when the `Shooting` has a `z0`, which is then a parameter of the program
  (`shooting.z0`). The docstring of `Shooting` now says what the code does: the controller is
  called at every one of an interval's `substeps` control steps.
- Sensors of combinations of the coordinates: `Measurement(..., matrix=)` takes an (m, n) matrix
  and reads `matrix @ q` and `matrix @ v` (instead of `observed`, a selection of coordinates).
  `KalmanFilter.encoder` of a robot with fewer motors than coordinates (`Underactuated`) no longer
  reads the motors as if they gave `q`, with the passive coordinates frozen: it returns the
  measurement of the motors themselves, with the Jacobian of the motor angles at the estimate as
  the matrix, so the passive coordinates are left to the model (`Rq` and `Rv` are then the
  covariances of the motors' readings). A robot with a motor on every coordinate keeps its encoder
  as it was. The underactuated tutorial runs the naive controller on such an estimate, and shows
  what it takes: a process noise that admits the model's error (with the default one the gate
  throws out every reading after the first swing and the filter breaks) and a controller that
  does not outrun the filter.
- Equations of motion from a function: `models.Equations(residual, energy=None)`, given to
  `FunctionModel(..., equations=)`, for a black-box, learned or external model whose dynamics
  are not built from parts. The residual `r(q, v, a, tau, f, p) = M(q) a + h(q, v) - tau - f` is
  written with CasADi operations and affine in `a`; `tau` is the generalized force of the motors
  and `f` that of the robot's own components, which still add springs, dampers and contact (the
  masses and the weight are in the equations, so there is no `Inertance` or `Gravity`).
  Everything that runs on a robot runs on it: the simulator, `rollout`, the planners, the
  Kalman filter, the momentum observer. With an `energy(q, v, p)` returning (T, V) it has the
  robot's energy and power; without one `Dynamics.energy` and `power` are `None` and what needs
  them (`ModelPlant.energy`, the `Passivation` stage) raises a `ValueError` that says so
  (`dynamics.needs_energy`). `testing.check_model` checks such a robot's mass matrix
  (symmetric, positive) and, with an energy, that a simulation never gains any. A test builds
  the same two-link arm from parts and from textbook equations and finds the same dynamics,
  energy, simulation, Kalman prediction and plan.
- Plan a controller on some coordinates of a larger robot: `Problem(system, plant=robot)` takes
  the dynamics from `plant` (every term and `q0` in its coordinates) while the controller stays
  written for its own robot, and reads the plant's motors as `VMCController` does in a
  simulation, through its transmission. The law, the energy of a `TankBudget` and the dampers'
  power of a shooting all go through it. A shooting of the turtle's flywheel controller on
  `turtle.crawler()` equals the simulation step for step, the flywheel's states included (a
  test), and the tutorial "Crawl with a flywheel" plans a stride with it. `Shooting(running=)`
  says whether a controller started from `z0` is already running (`z0` alone still means so)
  or starts with the plan, as one reset there does. `Space.on_manifold(q)` gives the equations
  that a configuration obeys (the unit norm of a quaternion; none for a flat space), and
  `Shooting` and `Equilibrium` constrain their nodes with them: before, a plan of a floating body
  could drift off unit quaternions and plan another body's motion.
- The models as numpy and PyTorch code: `core.backends.source`, `translate` and `export` write
  a CasADi function as straight-line Python, one statement per operation of its expression
  graph, for numpy (the text imports numpy only, so it runs without CasADi) or PyTorch
  (differentiable, on any dtype and device). Arguments and results have CasADi's shapes, with any
  leading batch dimensions. `Kinematics.functions(at, backend="numpy")`, `Dynamics.export` and
  `Compiled.export` give a model's functions that way, each with its `source`. numpy gives
  CasADi's numbers to rounding error and PyTorch to 1e-12 on the kinematics with their Jacobians
  and Hessians, including the derivatives by autograd; a model with a poorly conditioned mass
  matrix (the soft arm's dynamics) differs by a few parts in 1e11, as PyTorch's `sin` and `cos`
  differ in the last place. The extra `virtualmodelcontrol[torch]` and the tutorial "Use a model
  outside CasADi".
- `ForceTracking`, `StiffnessTracking`, `PositionRegulation`, `HoldingGoals`, `ContactForce` and
  `TaskStiffness` refuse a `StateController` (also inside a `Tank`) with a `ValueError` that says
  so, instead of failing on the shapes of its inputs: they read a controller in the layout of the
  motors. A robot with fewer motors than coordinates uses the plain frozen controller (it reads the
  motors), or `control.underactuated.DirectionalForce` for a force.
- Solver presets: `Problem(system, solver=...)` takes `"ipopt"` (the default), `"ipopt-exact"`
  (exact Hessians: 40 iterations instead of 165 on the tutorial's shooting), `"sqp"` (CasADi's
  `sqpmethod` with `qrqp`, the Hessian's negative eigenvalues clipped; other QP solvers by the
  `qpsol` option), `"rti"` and `"fatrop"` (as a general problem, without stages).
  `optimization.PRESETS` lists them and `optimization.solver.available(name)` tells whether the
  CasADi build has the plugin. `Result.status` words FATROP's endings as IPOPT's.
- Docs: tutorial "Model predictive control".
- Docs: the page "Support policy": which versions of Python and numpy are supported, what
  semantic versioning means here (a deprecation lasts at least one minor version), what the public
  API is, the schema of the run logs, where to report an issue, and the safety checks before the
  first run on a robot (also in "Real time and the robot").

### Changed

- `__all__` in `math`, `testing`, `compiler`, `dynamics` and `system`: what they list is the public
  API, as in every other module; `from virtualmodelcontrol.x import *` gives only that.
- The configuration and hardware profile files are read and written as UTF-8 on every platform,
  so that a file written on Linux reads on Windows.
- The docs use American spelling, the figure legends are 18 pt (readable on a phone), and wide
  tables wrap their code on a narrow screen.

## [0.5.0] - 2026-10-06

This release also carries the work meant for 0.4.0 (optimization of virtual mechanisms, the energy
tank, tuning and the calibrations), which was never released on its own.

### Added

- Underactuated Virtual Model Control, for robots with fewer motors than coordinates.
  `models.Underactuated(B)` is the actuation with an input matrix B (`Underactuated.joints` for
  motors on some joints): motor angles Bᵀq, torques B u, the allocation B⁺ τ, the projector E and
  the `defect` E τ it cannot realize. Its motors give the frozen configuration (the motors' part
  of q, the rest at `q_rest`), so a `VMCController` on it is the frozen controller.
  `control.StateController` reads (q, v) from the measurement (`meas["q"]`, `meas["v"]`, or its
  `read`) instead of the motors, which is the naive controller and also how a controller acts
  through an estimate of unmeasured coordinates. `control.underactuated` has `controller(compiled,
  base="naive"|"frozen", correction=None|"passive"|"tank", gravity=)`, the `Passivation` output
  stage (the passive and the tank correction: the motors never inject more power than the
  robot's own dampers take, or more than the tank holds), `Frozen` (the frozen configuration,
  also balancing the robot's gravity), `DirectionalForce` (a force along a direction by one
  scalar of the stiffness, closed form, positive definite), `projector`, `defect` and `feasible`
  (the wrenches the motors realize in full). An output stage with a `reset` is reset with its
  controller. They agree with the lab's own code on its three robots (every flag, the frozen
  point, the force law, and the closed loops).
- `robots.planar`: `arm("three-link")` with one passive joint, `arm("five-link")` with three, and
  `continuum()`, a three-section continuum arm driven by two tendons; `add_dynamics` and
  `add_continuum_dynamics` give them gravity and their springs and dampers.
- Docs: tutorial "Robots with fewer motors than joints".
- `vmc.optimization`: plan the motion of a controller on a robot and optimize its Params.
  `Problem(system)` is built once and solved many times. `free` chooses the Params to optimize
  and `parameter` the ones set at every solve (references, say), by name or glob; bounds and
  scale come from the Param. `Collocation(q0, horizon, nodes, initial=, transition=, scales=)` is
  trapezoidal collocation of the closed loop on the robot's full dynamics, without inverting the
  mass matrix. With an `initial` controller, its torques fade out as the new ones fade in with the
  blend of `SwapController`, so the plan is what the swap executes, and the model starts at rest
  where the robot is whatever it gets wrong there. `Effort`, `Cost` (a library coordinate brought
  to zero from a time on) and `Bound` (a coordinate kept within limits) are the terms, `Term`
  the base to write your own. `Result` holds the motion, the Params found, the cost by term and
  the solver's status, and `apply` puts the values into a running controller (returning the exact
  energy jump) or into a system. `search_references` searches a reference on a grid, one at a
  time, keeping the best converged plan. `Problem.build` returns the `NLP` for any other solver.
  `Collocation(scheme="hermite-simpson")` integrates to fourth order instead of second: it adds the
  acceleration at the middle of every interval to the unknowns, and needs far fewer nodes for the
  same agreement with a simulation.
  The program agrees with the lab's own optimization code to rounding error, and its solved plans
  to solver tolerance. Controllers with virtual states and non-flat configuration spaces are not
  supported yet.
- `optimization.Equilibrium(q0)`: the closed loop at rest, as a static problem. It takes the place of
  `Collocation`: the unknown is a configuration where the controller's torques balance the
  robot's own forces, found with the free Params, and the terms see it as their one node.
  `Cost` and `Effort` count it once, and a `Bound` holds there. The tutorial finds where the
  soft arm rests, in milliseconds.
- `Collocation(periodic=True)` plans an orbit that repeats: the last node equals the first (q and
  v), the first node is free (`q0` and `v0` are only the solver's starting guess), and `Bound`
  reaches it. `Collocation(free_time=(lower, upper))` makes the horizon an unknown within these
  bounds, starting at `horizon`: the node spacing, the controller's time and the coordinates of
  `vmc.Time()` are expressions of it, and `Effort` and `Cost` integrate with the horizon found.
  The windows of the terms still read the nodes' times at the starting horizon. `Result.horizon`
  is the horizon found (the given one for a fixed horizon, 0 for an `Equilibrium`) and
  `Result.t` uses it; `warm_start` takes a `horizon`. `Trajectory.dt` is a number or an
  expression of the free horizon, and `Trajectory.times` and `Trajectory.horizon` are the same,
  for terms of your own. `optimization.Period(param)` holds the horizon equal to a Param, so that
  a controller whose reference is a function of `vmc.Time()` and that Param (a period, free or a
  parameter) repeats with the orbit. Neither option goes with an `initial` controller. Both
  schemes work.
- Docs: tutorial "Optimizing a virtual mechanism".
- Docs: the optimization tutorial finds the period of a mass on a spring, and the least-effort
  period of a pulled one with a controller that keeps time, against their closed forms.
- Docs: the optimization tutorial plans the soft arm around a sphere (`Bound` on `SphereDistance`),
  checks the clearance in simulation, and shows that two sets of scales give the same plan.
- `vmc.sim.rollout(system, q0, T, dt)`: the closed loop simulated in one compiled call
  (`mapaccum`), differentiable with respect to the controller's live Params, with the robot
  integrated by `"implicit"` (the default, as `ModelPlant`), `"rk4"` or `"cvodes"`;
  it reproduces `vmc.sim.run` to rounding error. `vmc.sim.ode(system)` is the closed loop in
  continuous time, f(t, x) for SciPy's `solve_ivp`.
- `vmc.control.Tank(controller, level, capacity)`: an energy budget for changing a running
  controller. `tank.set(values)` applies a change of live Params as far as the tank can pay for
  the exact jump it gives the controller's energy (the largest fraction of the step that fits);
  a step that releases energy is applied whole, even with an empty tank, and refills it.
  `result.apply(tank)` passes an optimizer's result through it. `VMCController.jump(values)`
  gives the jump `set` would give, without applying it.
  Run in place of the controller (`vmc.sim.run(plant, tank, ...)`), the tank also refills itself
  with what the controller's own dampers take at every step.
- `identification.fit_stiffness_damping` takes a run log (`RunLog`) as it is, or a mapping whose motor
  torques are `motor_torque` or `u`.
- `identification.Steps(baseline, pulls, held_out, hold, rest)`: the step experiment as a controller
  that runs on any plant. Every motor gets the baseline torque, and each pull adds a torque to some
  motors for `hold` seconds, with `rest` seconds at the baseline between them. Its log marks the
  training steps (`train`), which the fit uses as they are; the held-out ones are for validation.
- `fit_stiffness_damping(..., friction=True)` adds a static friction torque per motor, F sign(θ̇),
  to the fit and returns it after K and D (`K, D, F`). Friction that is not in the fit passes for
  damping. A motor that stands still has none in the model, so a run that stops at the end of every
  step hardly shows it: use one that keeps the motors moving.
- `identification.validate(model, run)`: simulates a robot with its stiffness and damping under the
  logged torques of the held-out steps of a run, from the state the run is in at the first of them,
  and returns the root-mean-square error and the variance accounted for of each coordinate.
  `vmc.sim.rollout` takes `u`, motor torques with a row per step, to simulate open loop.
- Docs: the hanging arm's page identifies its stiffness and damping with `Steps`, checks them on
  held-out steps and shows the friction option.
- `identification.fit_transmission(motor, joint)`: the joint angle per motor angle of a cable drive,
  from a sweep of the joint, by least squares through the origin. It reproduces the finger's and
  the hand's transmission constants from their recorded sweeps. The finger's page shows it.
- `vmc.adaptation`: laws that change a running controller's live Params, a step at a time. Every
  `step` returns the jump of the controller's energy [J] that was applied, and a `Tank` passed in
  place of the controller applies only the part it pays for. A new value stays within the Param's
  bounds (a stiffness stops at zero), and a square matrix stays symmetric and positive
  semidefinite (`control.project_psd`).
  - `ForceTracking(controller, site, params, normal=None, max_force_step=0.01, max_step=None,
    rate=None)` descends the error of a contact force: `params` are glob patterns of live Params
    (a spring's goal or its stiffness), `normal` keeps the force along it. The gradient is exact,
    by automatic differentiation of the delivered torque, and agrees with the lab's hand-derived
    one to 1e-8 for the goal and the stiffness, at the tip and mid-arm, along a normal and in 3D,
    also where the point's Jacobian has rank below 3 (the two-motor finger). A step is the largest
    that changes the force by `max_force_step` [N] (the lab finds it by a finite-difference probe
    and agrees to 1e-5), or `rate`, a fixed learning rate as in the finger's experiments, where
    it agrees with the lab's joint-space stiffness and reference descent at the lab's rates.
    `direction` gives the descent direction without applying it.
  - `ForceRatio(controller, params, max_change=0.05)` and `Stiffening(controller, params, low,
    high, alpha)`: the laws of the lab's hand and finger that need no model. The first scales
    stiffnesses by 1 plus or minus `max_change` times the relative error of the force (the hand's
    multiplicative law); the second sets k(F) = low + (high - low)(1 - exp(-alpha F)) from the
    measured force (the finger's law). Both equal the lab's. K(d) is the `SigmoidSpring` and the
    `PolynomialSpring`.
  - `StiffnessTracking(controller, site, params, normal=None, robot=None, fraction=1.0)`: direct
    stiffness tracking. The stiffness at the site is affine in the springs' stiffness, so `target`
    solves a linear system for the Params that give a wanted matrix, the solution of smallest
    norm as in the lab's inversion, and `step` goes `fraction` of the way. Where no spring pulls
    it agrees with the lab's inversion to 1e-6; where they do, it counts the geometric terms
    that the lab's leaves out.
  - `PositionRegulation(controller, sites, gain=0.05)`: integral pose regulation. A step adds
    `gain` times the error of a point of the robot to the goal of the spring that pulls it.
  - `HoldingGoals(controller, sites, robot=None)`: the open-loop counterpart, the lab's position
    feedforward. It puts the goals at wanted positions (default: where the points are) plus the
    smallest offset that lets the springs carry the robot's own stiffness and weight at the pose
    now, one Newton step on the static balance. It agrees with the lab's to 1e-9; with as many
    springs as motors the arm then stays where it is.
- `vmc.estimation`: quantities of a running robot that no sensor measures. Both take the
  controller, a site (a name, an arc parameter or `(part, s)`) and a `normal`, and a `robot`:
  the model that holds the arm at rest, without the surroundings the plant may have. They use the
  controller's law before the output stages, the system's transmission and efficiency, and leave
  velocities out.
  - `ContactForce(controller, site, normal=None, robot=None)`: the force a robot exerts at a site
    at rest. What the delivered motor torques and the robot's own stiffness and weight do not
    balance goes through the site's Jacobian by the pseudo-inverse. It agrees with the lab's
    contact force from the virtual springs and the structural stiffness (the efficiency on the
    springs' term only) to 1e-9, with its torque-based variant, and with the finger's. A fingertip
    at rest on a simulated table reads the table's force to 2 %.
  - `TaskStiffness(controller, site, normal=None, robot=None)`: the stiffness [N/m] at the site, a
    3 by 3 matrix. It is the stiffness of the static balance, by automatic differentiation, with
    the geometric terms (the Hessians of the site and of the springs) counted, and agrees with
    the lab's exact apparent stiffness to 1e-8. The lab's first-order version leaves the
    geometric terms out and is not provided.
  - `KalmanFilter(system, dt, Q=None, P0=None, gate=40.0, robot=None)` and `Measurement(q, v, Rq,
    Rv, observed, name)`: a Kalman filter of the robot's state (q, v). Its process model is the
    robot's own dynamics, linearised at the estimate at rest and held over the step (the lab's
    frozen mass matrix, in general form), with the command going through the system's
    transmission. It fuses encoders (`kf.encoder`) and any sensor that sees part of the state,
    gates each measurement on its innovation, and reports the sensors it rejected (`rejected`,
    `rejected_total`) and the expected ones that offered nothing (`missing`). Over a run with
    dropouts and an outlier of each sensor it agrees with the lab's filter in its state to 1e-9,
    its covariance, its innovation distances and the sensors it rejects.
  - `Inversion(robot, at, damping, tol, max_iter)`: the configuration that puts the robot's points
    `at` where a position sensor saw them (motion-capture markers), by damped Gauss-Newton from the
    last result, for any robot. It agrees with the lab's closed form for the three-section arm
    to 1e-8, and finds the soft arm from its neutral configuration up to 0.6 rad of bend in each
    section (give `q0` beyond). `VelocityFilter(dt, alpha=0.3)`: the low-passed difference of
    consecutive samples, the lab's velocity of the markers, with `dt` the sensor's own period.
  - `ImuFilter(arm, mounts, kp, acc_tol, gravity)`: the curvature (Dx, Dy) of each section of a
    soft arm and its rate from an IMU on its base and on each section end, by a complementary
    filter of each section's relative rotation (the gyros turn it, the gravity both accelerometers
    see corrects it, and it is kept on the rotations the arm can make). `calibrate` takes the
    gyro bias and the starting pose from the arm held still. Its `observed` coordinates go to
    `Measurement`. It agrees with the lab's filter to 1e-10 over a run with accelerometers that
    cannot be trusted.
- `VMCController.inputs()`: the vector the law reads, with the live Params as they are now.
  A Param that acts only through a virtual state does not change the force at once, and the
  laws leave it.
- Docs: the tutorials "Track a contact force" (the laws, the estimate without a sensor, the tank)
  and "Shape the stiffness of the tip"; the finger's page presses between two force levels; the
  two arms' page holds an object with a chosen force, with estimates alone, with a sensor on the
  object, and through a tank (the object feels the wanted force to 0.2 % from the estimates).
- `vmc.Gated(component, gate)`: an element whose force and energy are multiplied by a live Param
  `gate` between 0 and 1. `optimization.Sparsity(weight, *patterns)` adds the sum of the free Params
  that match (each at least 0) to the cost. With the gates free, the optimizer keeps the elements
  that earn their place: structure optimization, tested on a mass with three springs and on the
  soft arm with five repulsive fields, and shown in the optimization tutorial ("Which fields to
  keep").
- `vmc.models.FunctionModel(frame, space, params, sites, q_unit)`: a kinematic model from a function
  `frame(q, at, p)` that returns the rotation and the position of a site, as CasADi values or as
  lists and arrays. It works in `Kinematics`, mechanisms and `check_model`; a function cannot be
  written to a file, so it has no `to_dict`.
- `vmc.testing.check_model(model)`: the model contract in one call, for a kinematic model or a
  robot's mechanism: everything finite (also with each Param a tenth of its default and ten times
  it), orthonormal rotations, Jacobians, angular Jacobians and Hessians against finite
  differences, no jump along the arc parameter `s`, `to_dict` and `from_dict` giving the model
  back, and, for a robot with masses and no dampers, an energy drift that falls with the step. It
  raises an `AssertionError` that lists the failed checks and returns the worst error of each. The
  tutorials "Build your own robot" and "Extend the library" use it.
- `vmc.BoxDistance`, `CapsuleDistance` and `CylinderDistance`: signed distances from a point to a
  box with its sides along the axes, a capsule and a cylinder, for contacts and obstacles, with the
  same Params as `SphereDistance` (and the kinds `box_distance`, `capsule_distance` and
  `cylinder_distance` in configuration files). The contact tutorial uses the cylinder.
- `vmc.ContactFriction(distance, stiffness, friction)`: Coulomb friction at a contact, smoothed. It
  pushes a point against its sliding speed along the surface, with the size of the friction
  coefficient times the contact spring's force (give it the spring's stiffness `Param`), and falls
  linearly below the small speed `speed`, so a point at rest creeps. It works on every signed
  distance, which now share the base class `SurfaceDistance` with a `unit_normal`; the type is
  `contact_friction` in configuration files. The contact tutorial drags a fingertip along a table.
- `vmc.sim.run(..., record="robot")` logs what a simulated robot feels, component by component:
  `robot/<name>/y`, `ydot`, `force` and `torque` of each spring, damper and contact, from
  `ModelPlant.elements()` (and `Dynamics.elements`). A real robot reports no forces, so recording
  it raises an error.
- `vmc.estimation.Encoders`, `Markers`, `Imus` and `LoadCell`: simulated sensors, the true state in
  and a noisy reading out, as the estimators take it. The encoders read the motors through the
  transmission, with noise and a constant slack per motor; the markers, the positions of points of
  the arm; the IMUs, the angular velocity and gravity in each frame's own axes, with a constant gyro
  bias; the load cell, a force with a bias and noise. Every one has a seed. The estimation tutorial
  uses them in place of the lines it wrote by hand.
- `vmc.adaptation.DitherSeeking(controller, params, amplitude, frequency, gain, window)`: extremum
  seeking from the cost alone, the two-tone law of the turtle paper. Each scalar live Param is held
  at an estimate plus a sinusoidal dither at its own frequency (2π n / `window`, a whole n each, so
  that the tones are orthogonal); the cost's correlation with a tone over the window is the slope
  with respect to that Param, and the estimate moves downhill at `gain` times it, at most
  `max_rate`, within the Param's bounds. On a `Tank`, the changes are paid from its energy, which is
  the paper's passivity cap. The crawl tutorial lets the crawler find its best `peak` this way.
- `turtle.controller(robot, depth=, peak=, steer=, limit=)`: with any of them the cranks' springs
  are `PhaseSpring`s (the paper's phase-modulated, saturating potential); without, they are the
  plain ones of the lab's controller, as before.
- `vmc.optimization.MovingHorizon(system, dt, window=10, Q=, P=)`: moving-horizon estimation of a
  robot's state (q, v). Every step finds the states of the last `window` steps that best agree
  with the sensors' readings (the same `Measurement`s as the Kalman filter's), with the robot's own
  dynamics up to the process noise `Q`, and with the state before the window, which goes one step of
  an extended Kalman filter whenever the window slides. It is solved by Gauss-Newton with a trust
  region. With a linear robot it is the Kalman filter, to rounding error, however short the window;
  it also gives `cost` and the covariance `P`. The estimation tutorial runs it on the soft arm.
- `vmc.identification.fit_params(robot, names, runs)`: any Params of a robot, by name or glob,
  fitted to logged runs by least squares on the residual of its own dynamics, within the Params'
  bounds. A residual linear in the Params (masses, stiffnesses, dampings, efficiencies) is solved
  in a step, and any other, such as the saturating damper's, by Gauss-Newton. It returns the values
  in each Param's shape, their standard errors and the residual's root mean square. A run with
  an `a` uses it, and otherwise the acceleration is the smoothed velocity's difference against the
  torques averaged over the step they were held. It agrees with ordinary least squares, values and
  covariance, on a mass, spring and damper. The tutorial "Fit Params to a run" uses it.
- `KalmanFilter(..., unscented=True, substeps=10)`: the prediction sends sigma points of the
  estimate (the scaled unscented transform, a root of three standard deviations out) through the
  robot's own integrator, in `substeps` steps, instead of linearising at the estimate; the sensors
  stay linear, so the update is the same. On a double pendulum it follows the mean of a Monte
  Carlo to 0.003 where the linearised prediction is off by 2; on a linear robot the two agree.
- `vmc.estimation.MomentumObserver(system, dt, gain, robot=None)`: the external force on a robot
  from its motion, without accelerations. From the momentum and what the model says would change
  it, it gives the generalized force the model does not explain, as a low-pass filter of bandwidth
  `gain` of the true one, and `force(site, normal)` carries it to a site, as `ContactForce` does.
  It works while the robot moves, where `ContactForce` assumes rest. The force tutorial shows a
  finger landing on a table. Its velocity must be the rate of its configuration (no floating
  bodies yet).
- `vmc.estimation.object_compliance(position, force, baseline_position, baseline_force)`: the
  compliance in m/N of an object a tip presses, the distance over the force that stiffer settings
  of the controller add to a gentle one, from the median of each setting's samples. It is the
  lab's probe of the hand, and it agrees with the lab's code to rounding error. The force tutorial
  shows it on three objects, with the force from `ContactForce`.
- Docs: the contact tutorial also shows two points of a robot touching (a self-contact, the two
  arms), friction between two points and a soft object between two tips, from the pieces above.
- `vmc.control.project_psd(K)`: the symmetric positive semidefinite matrix nearest to a stiffness,
  for an update that an adaptation law proposes, before it goes through `set` or a tank.
- `vmc.sim.energy_balance(log)`: the controller's energy over a run recorded with
  `record=["energy"]`: its energy, the work given through its port, what its dampers took and its
  sources gave, `injected` (what changes of live Params put in, plus the error of the steps) and
  the `margin` it can still give, which stays above 0 while the controller is passive.
- `vmc.optimization.Grid`, `Random`, `CMAES`, `ExtremumSeeking` and `Bayes`: gradient-free tuning by ask and tell
  (`ask()` gives candidates, `tell(candidates, costs)` takes their costs), `tune` to run it, and
  `bounds_of` for the bounds and values of named Params. `Bayes` fits a Gaussian process to the
  costs seen so far and asks for the candidate with the highest expected improvement: it suits
  slow episodes, tens of them. The episodes are yours, in simulation or on a robot in your own loop.
- `compile_dynamics(robot, runtime, actuation)` takes the actuation of a robot that has none, so a
  system's efficiency Params (`robot.efficiency.c1`) are the ones the dynamics read.
- `vmc.math`: rotations and rigid transforms that run on numpy arrays and CasADi symbols alike
  (`rot_x`, `rot_y`, `rot_z`, `rot`, `skew`, `vee`, `exp_so3`, `log_so3`, `transform`, `invert`,
  `exp_se3`, `log_se3`, `adjoint`), for writing the `frame` of a `FunctionModel`. Every function
  is smooth at the identity, also in its second derivatives (the exact forms lose them there).
  They replace the three rotation helpers of `core.symbolic`.
- `SerialChain` joints: `"helical"` (a turn that slides by a pitch, given as `("helical", pitch)`),
  `"spherical"` (three coordinates, a rotation vector about the joint's point) and `"free"` (six:
  a translation and a rotation vector, a floating base). A chain on a free base, a brick of point
  masses thrown with a spin and its momenta and energy are tested; a branch of a tree is a part
  mounted on a site of another part by `Assembly`.
- `SerialChain` joint `("rail", waypoints)`: a slide along the natural cubic spline through the
  waypoints, which are a `design` Param (an array of rows of three) in the frame of the joint
  before it. Its coordinate is the spline parameter, 0 at the first waypoint and 1 at the last. It
  agrees with SciPy's natural spline, and a bead on a circular wire swings as a pendulum.
- `SerialChain` joint `"floating"`: a floating base whose rotation is a unit quaternion, with
  seven coordinates and six velocities (the translation's and the body's angular velocity in its
  own axes). `vmc.Quaternion` is its space, and `Space.coadjoint` gives the term that
  velocities that do not commute add to the dynamics (the Euler-Poincare term), which
  `compile_dynamics` now includes. A body tumbles about its intermediate axis through several
  full turns by Euler's equations, and a base with an arm keeps its momenta and energy.
- `vmc.FramePoint(model, site, q=coordinate)`: a point of a virtual model, at the configuration
  that `coordinate` gives: a virtual state (a cart on a rail with a mass of its own), a
  reference, or a function of time. `vmc.Time()` is the time of the run, to build functions of
  time with `Custom`; a damper on the difference to such a goal feels its velocity.
- More coordinates: `vmc.Sum` (and the plus sign: `tip + [0, 0, 0.05]`), `vmc.FrameRotation` (a
  frame's rotation matrix, row by row), `vmc.OrientationError(model, site, goal=)` (the rotation
  vector from a goal orientation to a frame's, in the goal's axes, with `goal` a live Param; a
  spring on it is a rotational spring whose torque is exact at any error, where the one from the
  angular Jacobian is 8 % off at 0.4 rad), `vmc.InFrame` and `vmc.FromFrame` (a vector along the
  axes of a frame, and back: a stiffness along the tool's axes turns with the tool). They take a
  site or an arc parameter, and a virtual model through `q=`. Configuration files build them as
  `sum`, `rotation`, `orientation_error`, `in_frame` and `from_frame`.
- More components: `vmc.RotationalInertia(rotation, inertia)`, the inertia of a rigid body about
  its frame (a matrix, or three principal moments) on the frame's rotation matrix, next to a
  `PointMass` for its mass: it follows Euler's equations, moves like the same body made of point
  masses to rounding error, and a rotational spring on it oscillates at the square root of K over
  I. `vmc.DiodeDamper(y, D, sign, smoothing)` damps one direction of motion only.
  `ForceSource(y, f, max_force=, max_power=)` bounds the force smoothly, and the power it
  delivers. `Mechanism.add_mass_along(name, mass, s0, s1, n)` spreads a mass along a model as
  point masses at the nodes of Gauss-Legendre quadrature. An `Inertance` on a difference is an
  inerter, tested.
- Docs: "A spring on the tool's orientation" and "A spring along the tool's axes" in "Kinematics on
  the UR5"; "A damper that works one way" in "Coordinates and components"; a rigid body and the
  mass along an arm in "Build your own robot".
- Docs: the tutorial "Estimate the state of a soft arm" (encoders with slack, markers and IMUs
  fused by the filter, an outlier and lost frames); "A rail along a path" and the floating joint
  in "Build your own robot"; "A cart on a rail" in "Coordinates and components".
- `vmc.PhaseSpring(coord, stiffness, depth, peak, steer, side, limit)`: a spring on a deflection e
  whose stiffness follows a phase, on the coordinate `Stack(e, phase)`: K (1 + side steer) (1 +
  depth cos(phase - peak)). Its force is minus the gradient of its energy, so the phase feels the
  reaction that keeps a flywheel loop passive. With `limit` the potential saturates, and the torque
  never goes above K (1 + |steer|) times the limit. With no depth, no steer and no limit it is a
  `LinearSpring`. The type is `phase_spring` in configuration files.
- `vmc.models.Passive(nv)`: an actuation with no motors, for a part that nothing drives, such as a
  floating body. Stacked with the other parts' motors (`Assembly.stacked_actuation`), the motor
  vector holds only theirs, so a simulated robot can have more coordinates than motors. The motors
  do not give a passive part's configuration, so a controller is compiled against a robot without
  it, and the simulator runs the one with it.
- `robots.turtle.crawler(**CRAWLER)`: a simple crawler for the simulator, with placeholder
  constants, not the lab's turtle: a floating body on the ground with gravity, two cranks about its
  lateral axis with a foot each, and a contact spring, damper and friction at each foot and at the
  four corners of the underside. Its motors are the two cranks, in the gait convention of
  `turtle.robot()`, which is what the controller is compiled against. Under the flywheel controller
  it crawls forward, further the faster the flywheel is driven, and does not advance without
  friction; a positive steer of `PhaseSpring` turns it left. Docs: the tutorial "Crawl with a
  flywheel".

### Removed

- `vmc.ros` (`RosPlant`, `control`, `serve`, `LiveParams`, `JointIO`, `TOPICS`) and the direct
  Dynamixel path (`hardware.DynamixelPlant`, `home`, `present_ticks`, `scan`, `latency_timer`,
  `Bus`, `SdkBus`, `FakeBus`), deprecated in 0.3.0. The library is communication-agnostic: it
  turns measurements into torques, and how they reach the robot stays outside it. The Dynamixel
  SDK and pyserial are no longer dependencies, and CI no longer runs the ROS tests. Hardware
  profiles (`HardwareProfile`, `Motor`, `KT`, `MODES` and each robot's `hardware` template) stay.
  A project that still uses a removed name keeps `virtualmodelcontrol==0.3.0`.
- `vmc.interactive` (the window, the keyboard, the joystick, the session, the mailbox, the wrapper
  and the recorder), the `joystick` extra and `SimClock`'s `speed`, `now` and `sleep`: the library
  is for control, not for interfaces. A running controller still changes through
  `controller.set`, schedules and swaps. A project that uses them keeps `virtualmodelcontrol==0.3.0`.
- From the hardware profiles, the raw Dynamixel units (encoder ticks, velocity units, goal
  currents, motor constants `kt` and the `KT` table, baud rate, serial port, velocity filter, home
  ticks, per-motor torque limits and operating modes). A `Motor` is now `Motor(id, sign, hold)`
  and a `HardwareProfile` `(motors, rate, bus_order)`, with the conversions between a driver's
  degrees, bus order and signs and the library's SI units.

## [0.3.0] - 2026-10-05

### Added

- `vmc.config`: experiments in YAML files. `vmc.config.load` reads the robot (a template and its
  arguments), named coordinates, the controller's elements (a registered component `type`, the
  `coordinate` it acts on, its gains), controllers to swap to, and how to run them: a simulation
  (with the robot's dynamics and simulated elements such as contacts), the rate, the duration,
  output stages, schedules and the initial virtual state.
  `Experiment.run` runs it, `save` writes it back with the Params' current values. Robot
  templates, output stages and coordinate kinds are registry entries, so a project adds its own.
  An experiment from a file gives the same run as the same experiment written in Python.
- `vmc.control.Schedule` and `ScheduledController`: live Params that follow values over time
  (in straight lines or steps) and swaps at given times; a new run starts afresh. The lab's
  linear ramps are two-point schedules.
- `vmc.sim.run(..., z0=...)` also takes a function of the first reading, such as
  `turtle.initial_state`.
- Docs: tutorial "Experiments in files".
- Run logs. `vmc.sim.RunLog` saves a run whole or not at all (`save`: a compressed `.npz` with a
  schema version and what the run was, in `meta`: the library's version, the start time, the
  Params at the start, the plant's hardware profile and, from a configuration, the
  configuration; written through a temporary file, and a name already taken is refused unless
  `overwrite=True`), loads it back (`load`) and exports CSV (`to_csv`, one column per entry).
  `step` adds one step and keeps every signal at one row per step, with NaN for what a step
  did not compute.
- `vmc.sim.replay` sends a log's motor torques again to a simulated robot, each held for as long
  as it was in the run, from the log's `q` and `v` or, for a real robot's log, from its first
  motor reading (`ModelPlant.state_from_motors`); `vmc.sim.compare` gives the largest and the
  root-mean-square difference of each signal of two runs. A replay on the true model gives the
  run back.
- A run's log holds `law_torque`, the controller's torque before the output stages.
  `vmc.sim.run(..., record=...)` adds the live Params (`"params"`), each element's coordinate,
  rate, force and share of the motor torques (`"elements"`), and the controller's energies and
  powers (`"energy"`), from `VMCController.elements()` and `balance()`.
- Configurations: `experiment.run` settings (`name`, `folder`, `overwrite`, `record`) save each
  run beside the file and refuse a name already taken before the run starts.
- Docs: tutorial "Run logs".
- `vmc.interactive`: change a running controller by hand. `Controls` is a thread-safe mailbox
  for new values of live Params and for swaps. `Interactive` wraps any controller and applies
  what the mailbox holds just before each step, in a simulation or inside a robot's own node;
  the changes act exactly as `controller.set` before the step, and it counts the energy they give
  the running controller (`injected`). `Recorder` turns what was applied into `schedule` entries
  with `step` interpolation and saves them into a configuration, which repeats the session
  exactly. `Window` (matplotlib, no new dependency) draws the robot live, with goals to drag,
  sliders, buttons to swap controllers, record and pause; `Keyboard` and `Joystick` (pygame, the
  extra `virtualmodelcontrol[joystick]`) move a goal; `Session` runs a simulation in real time
  in a worker thread with its window.
- `vmc.sim.SimClock(dt, speed=...)` paces a simulation to the computer's clock: with `speed`,
  `run(T=None)` runs until stopped and Ctrl-C ends it with the log so far. Other simulations are
  unchanged. `VMCController.live_params()` and `Experiment.z0()`.
- Docs: tutorial "Interactive control".
- Docs: tutorials "Real-time runs" (the loop on the computer's clock: measured steps, late steps
  never caught up in a burst, old and missing readings, Ctrl-C, and a checklist for the first run
  on a robot) and "Swaps and schedules" (the quintic blend and its continuity, a swap on the soft
  arm, the schedules' interpolation).
- `vmc.ros.control`: the controller node. It runs a controller at the profile's rate on the
  robot behind the driver's topics, with its live Params as ROS parameters, for a duration or
  until Ctrl-C, and ends with zero torque.
- On real time (`WallClock`), `vmc.sim.run(..., T=None)` runs until Ctrl-C, and Ctrl-C ends
  a real-time run with the log so far.
- CI runs the ROS tests in ROS 2 Humble and Jazzy containers, on a private domain.
- `vmc.control.SwapController` and `blend_weight`: one set of virtual elements replaces another
  with a quintic blend of the two controllers' torques (continuous torque and first two
  derivatives), the blend the lab's controller uses; a swap asked for during a blend waits for
  it. `vmc.ros.control(..., swaps={name: controller})` swaps on a `String` message.
- `vmc.Efficiency`: the motor torque a transmission delivers, as a polynomial of the commanded
  one, τ = c₁u + c₂u² + … + cₙuⁿ, each coefficient one value for every motor or one per motor.
  The coefficients are `design` Params, to change, tune, keep live in a simulator or identify.
  `Direct` and `TendonTransmission` take one, or a number for its linear coefficient.
- `vmc.identification`: `plateaus`, the settled end of every hold of a command in a log, and
  `fit_efficiency`, a least-squares fit through the origin, motor by motor or shared, from
  delivered torques or from one measured quantity with a weight per motor (a fingertip force,
  say). Both agree with the lab's calibration code, and the fit gives the finger's
  `adapt.MOTOR_EFFICIENCY` back from its recorded data.
- Docs: concepts pages "Transmission efficiency", "Virtual Model Control" (coordinates, forces,
  the torques that realize them, virtual states), "Passivity" (the energy balances, stability,
  what weakens it) and "Finger and hand kinematics" (motors to joints, joints to the
  fingertip, the hand's couplings).
- The constrained elements as components of their own: `ConstrainedLinearSpring`,
  `ConstrainedTanhSpring`, `ConstrainedGaussianSpring`, `ConstrainedLinearDamper` and
  `ConstrainedTanhDamper` act along one direction (`normal`) only, leaving the plane normal to
  it free. They give the forces of the lab's constrained elements, checked against them; the
  tanh ones saturate along the normal rather than axis by axis.
- `vmc.identification.fit_stiffness_damping`: a robot's diagonal stiffness and damping from
  logged runs under known motor torques, by least squares with K, D ≥ 0 in motor torques,
  relative to a resting baseline, with the robot's masses and gravity known; each run can carry
  its own gravity. It recovers the stiffness and damping of a simulated arm within a few percent.
- Docs: two single-arm examples, each about its own arm and its own experiments. The soft arm on
  its side (`145-145-145`) reaches past an obstacle with two repulsive fields, over a sweep of
  their strength, and a tanh spring limits the force on a string tied to the tip. The hanging
  arm (`145-290-290`) is identified from simulated step responses with
  `fit_stiffness_damping` and reaches around an obstacle with a tanh spring and two fields.
  Both use the same stiffness and damping. The tutorials, the home page and the README use the
  arm on its side; the Helyx geometries and the parametric template are on the soft-arm
  kinematics page.

### Changed

- Every robot's efficiency is 1 by default: models identified from the commanded torques
  already include their transmission. The soft arm and the two arms take each torque as the
  controller sends it again, with `helyx.SIM_STIFFNESS`, `helyx.SIM_DAMPING`,
  `bimanual.STIFFNESS` and `bimanual.DAMPING` referred to the commanded torque (the identified
  values divided by the 0.12 they were fitted with), as in 0.1.0. 0.2.0 applied 0.12 to every
  torque with physical stiffness and damping: the same equilibria without gravity, but natural
  frequencies about 2.9 times lower and a gravity compensation that cancelled only 12 % of the
  weight. `helyx.EFFICIENCY`, `bimanual.EFFICIENCY`, `adapt.MOTOR_EFFICIENCY` and
  `adapt.HAND_MOTOR_EFFICIENCY` stay as calibrations, never as defaults.
- A transmission's efficiency Param is now `efficiency.c1` (and `c2`, … for a polynomial);
  transmissions saved by 0.2.0 still load.
- `import virtualmodelcontrol` no longer loads SciPy, which took about three quarters of its
  import time; `identification.fit_stiffness_damping` loads it when called.
- Docs: the opening page has no left index (the pages of the five sections keep theirs) and
  leads to the concepts and the reference. Installation: the conda recipe takes Python from
  conda-forge, because conda's default channels stop a new installation until Anaconda's terms of
  service are accepted, and shows how to keep ROS's `PYTHONPATH` out of a conda environment, where
  pip otherwise ends with an error about ROS's packages and skips the ones ROS also ships.

### Deprecated

- `vmc.ros` (`RosPlant`, `control`, `serve`, `LiveParams`, `JointIO`, `TOPICS`) and the direct
  Dynamixel path (`hardware.DynamixelPlant`, `home`, `present_ticks`, `scan`, `latency_timer`,
  `Bus`, `SdkBus`, `FakeBus`) warn when used, and leave in 0.4.0 together with the Dynamixel SDK
  and pyserial dependencies and the ROS CI job. The library is communication-agnostic: it turns
  measurements into torques, and how they reach the robot stays outside it. Hardware profiles
  (`HardwareProfile`, `Motor`, `KT`, `MODES` and each robot's `hardware` template) stay. Python
  shows a `DeprecationWarning` only for code run as a script or in a notebook; run a node with
  `python -W default` to see them there.

### Fixed

- `vmc.sim.run` needs only the plant's documented methods: when its guard trips, it sends zero
  torque without reading the plant's last command `u`, which a plant of one's own may not have.
  The zero command has one torque per motor rate, also on real time.
- `vmc.ros` is available after `import virtualmodelcontrol as vmc`, like `vmc.viz` and
  `vmc.hardware`; it imports ROS only when a ROS class is first used.
- The hand example grasps the ball with the pads of the thumb, index and middle fingers: the
  ball sits where the thumb opposes the fingers, and the grasp takes the thumb's tip at the end
  of its last phalanx. Before, the thumb pressed on the ball with its side. The template's
  kinematics are unchanged.
- Figure labels written as math where the figure font has no glyph (N·m on the efficiency
  page); a test keeps figure text to characters the font draws.

## [0.2.0] - 2026-10-04

### Added

- Every dependency installs with the package: numpy, SciPy, CasADi, matplotlib, PyYAML, the
  Dynamixel SDK and pyserial. CI runs on Ubuntu 22.04 and 24.04, with a clean-install job.
- Robot templates: `robots.bimanual` (two Helyx arms on one frame, with their identified
  stiffness and damping and the transmission efficiency 0.12), `robots.adapt.hand` (five digits,
  13 motors, spread coupling, masses and joint limits), `robots.turtle` (two cranks and the
  virtual-flywheel controller), `robots.ur5` (from DH parameters) and `ur5.with_hand()`.
- Every template is parametric in its geometry, with today's values as defaults:
  `helyx.arm(lengths=..., section_radius=..., spool_radius=..., tendon_angles=..., masses=...)`
  for any number of segments; `bimanual.arms(lengths=..., tendon_angles=...,
  base_positions=..., efficiency=...)`, each one for both arms or by arm;
  `adapt.finger(link_lengths=..., link_masses=..., link_cogs=..., joint_axes=...,
  motor_radius=..., pulley_radius=..., pip_transmission=...)` and `adapt.finger_coupling`;
  `adapt.hand(...)` and `ur5.with_hand(...)`, which replace any entry of the hand's geometry
  tables by key (`tip_offsets={"thumb": ...}`), with the hand's mounting position on the
  flange; `ur5.arm(d=..., a=..., alpha=...)`; joint-limit springs with custom ranges; output
  stages and `add_dynamics` sized to the arm.
- `vmc.Kinematics`: positions, rotations, Jacobians, angular Jacobians and Hessians of any site,
  from configuration or from motor angles.
- `models.JointSpace` for robots described by their joints only; `SerialChain.from_dh` and site
  rotations; `Assembly` parts mounted on another part's frame.
- `TendonTransmission(efficiency=...)` and `Direct(efficiency=...)` (one value or one per
  motor): the robot receives the commanded torque times the efficiency. `adapt.finger`,
  `adapt.hand` and `ur5.with_hand` take `efficiency=...` and stay lossless by default: the
  measured `adapt.MOTOR_EFFICIENCY` and `adapt.HAND_MOTOR_EFFICIENCY` are static calibrations for
  force estimates, and unequal efficiencies on coupled motors make virtual elements
  non-conservative in simulation.
- `SpeedRegulator`: drives a virtual state at a commanded speed, with a ramp.
- `control.output`, opt-in output stages for real hardware: `FrictionCompensation`,
  `Pretension`, `TorqueOffset`, `TorqueLimit`; each template's own stage
  (`helyx.output_stage()`, `bimanual.output_stage()`, `adapt.output_stage()`,
  `adapt.hand_output_stage()`).
- `vmc.viz`: the lab figure style (`use_style`, `save` as PDF and SVG) and drawing helpers
  (`draw_robot`, `draw_spring`, `draw_damper`, `draw_frame`, `draw_force`, `draw_goal`,
  `label_axes`). `vmc.viz` is available after `import virtualmodelcontrol as vmc` and loads
  matplotlib only when first used.
- `vmc.viz.animate`: an animation of a run (the robot, springs to fixed points, the path of a
  site, anything else through a callback), saved as MP4, GIF or animated WebP. `imageio-ffmpeg`
  is now a dependency, so MP4 works on every system.
- `vmc.sim.run(..., z0=...)` starts the controller from a given virtual state, and the run log
  records the virtual state `z` when the controller has one.
- Contact as stiff one-sided springs: signed distances `PlaneDistance` and `SphereDistance`,
  `ContactSpring` (optional smoothing for optimization) and `ContactDamper` (damps only in
  contact); `adapt.add_dynamics` gives the finger and the hand their gravity for simulation.
- Documentation rebuilt for students: the Material layout (sections as tabs, the pages of each
  section in the left sidebar), large type, the library's logo and navy colours;
  installation and troubleshooting; tutorials (how it works, a first controller, coordinates
  and components, parameters, energy, kinematics, tuning, contact, building a robot, extending
  the library); an example per robot (soft arm, two arms, finger, hand and the UR5, turtle), each
  opening with its robot's schematic and parameters; concepts (the library's structure, soft-arm
  kinematics, conventions and glossary). Every figure, schematic, table value and animation is
  computed when the documentation builds.
- `vmc.hardware.HardwareProfile`: the motors of a robot (bus ID, Dynamixel model, sign, mode,
  motor constant, optional torque limit) in the library's motor order, with the bus settings;
  conversions both ways between encoder ticks, raw velocities, goal currents and the published
  degrees of a ROS driver and SI units, with commands clamped before the integer conversion so
  they saturate instead of overflowing; saved and loaded as YAML. Each template has its
  profile: `helyx.hardware()`, `bimanual.hardware()`, `adapt.finger_hardware()`,
  `adapt.hand_hardware()`, `turtle.hardware()`.
- `vmc.hardware.DynamixelPlant`: a robot's Dynamixel motors as a plant through the Dynamixel
  SDK, without ROS. It sets a safe goal before each motor's torque goes on, takes the start
  positions as zero, reads angles and smoothed rates, writes torques as clamped goal currents,
  sends zero torque when commands stop (a watchdog), and on leaving (exceptions and Ctrl-C
  included) sends zero torque and switches the torque motors off; held motors keep their
  position. `hardware.home` drives the motors slowly to recorded home positions and refuses a
  motor that lost a turn; `hardware.scan` finds the motors and their baud rate;
  `hardware.FakeBus` stands in for the bus in tests and dry runs.
- `vmc.sim.WallClock`: `vmc.sim.run` on real time, the loop of simulation unchanged. Each step
  runs on the computer's clock with its measured time; a reading older than `stale` sends
  zero torque; a late step starts the next one from now instead of catching up; the log
  records each step's `dt`, and `log.info` the rate, the slowest step, the overruns and the
  stale readings, with a warning when steps overran.
- `virtualmodelcontrol.ros` (ROS 2, any distribution; rclpy and std_msgs, loaded only when
  used): `RosPlant`, the robot behind the lab driver's topics (degrees, the bus order, raw
  signs, torques kept within the range of the driver's goal current); `serve`, a digital twin
  that publishes a simulated robot on the same topics, with a lockstep mode in which a run over
  ROS repeats the simulator's; `LiveParams`, a controller's live Params as ROS parameters,
  applied between steps.

### Changed

- The soft arm models the efficiency of its tendons: it receives η = 0.12 times the commanded
  torque (`helyx.EFFICIENCY`), and `helyx.SIM_STIFFNESS` and `helyx.SIM_DAMPING` are the
  identified, physical values. 0.1.0 divided them by 0.12 and delivered the full torque: the
  same equilibria without gravity, but natural frequencies about 2.9 times too high and a sag
  under gravity about 8 times too small. Controllers are unchanged and never divide their
  torques by η, so the simulated arm receives η of every command, gravity compensation
  included.
- The figure style without LaTeX uses matplotlib's own Computer Modern font (`cmr10`) instead
  of DejaVu Serif, with tick labels set as math. Figures made with LaTeX are unchanged.

### Fixed

- `RunLog` copies each row, so a signal updated in place (a controller's virtual state) is
  recorded correctly at every step.
- `bimanual.add_dynamics` takes the stiffness and damping of one arm (a dict by arm, arrays
  included) and keeps the other arm's identified values.
- `viz.draw_robot`: no joint markers along continuum arms mounted in an assembly; robots without
  geometry (joint-space models) draw nothing instead of failing; serial chains end at their
  `tool` site (the UR5 flange) as well as at a `tip`.

## [0.1.0] - 2026-10-04

First public release.

### Added

- Package scaffold: `pyproject.toml` (hatchling + hatch-vcs), tests, docs, CI workflows.
- `core`: `Param` and `ParamSet` with scopes (fixed, design, episode, stage) and a `Binding` that
  keeps live Params symbolic and folds the rest in; spaces `Euclidean`, `SO2`, `Product`;
  symbolic helpers; `Signals`; a registry with plugin entry points; unit strings.
- `mechanisms`: coordinates (`FramePoint`, `Joint`, `State`, `Ref`, `Difference`, `Slice`,
  `Stack`, `Projection`, `Norm`, `Custom`), components (linear, tanh, Gaussian, sigmoid,
  polynomial and limit springs; linear and tanh dampers; `PointMass`, `Inertance`; `ForceSource`,
  `GravityCompensation`) and `Mechanism`. A tanh spring on a projection saturates at its maximum
  force in every direction.
- `models`: parametric `PCC` with n segments, `TendonTransmission` (θ > 0 pulls), `Direct`
  drive, and `SerialChain` (product of exponentials).
- `robots.helyx`: the three Helyx arm geometries with their default parameters.
- `VirtualMechanismSystem`, `compile()` and `VMCController`: about 20 µs per control step for
  three springs and three dampers on a three-segment arm (`benchmarks/tick.py`).
- `compile_dynamics()`: robot dynamics assembled from the robot's components (canonical
  residual, mass matrix, energy, power, linearly implicit Euler step); `Gravity` component.
- `sim`: `Plant` and `SimPlant` protocols, `ModelPlant`, `run()` with a `Guard` (zero torque on
  missing or non-finite readings) and a `RunLog`.
- `models.Actuation` protocol; `helyx.add_dynamics` with the simulated arm's stiffness and
  damping.
- Docs: closed-loop simulation tutorial and an "Extend the library" how-to.
- `models.LinearCoupling` (one motor driving several joints), `models.Assembly` (several models
  mounted on one base) with `StackedActuation`; `FramePoint` accepts a part and an arc parameter.
- `robots.adapt`: the ADAPT finger (two motors, coupled distal joints), joint-angle coordinate
  and joint-limit spring.

[Unreleased]: https://github.com/vigno0405/VirtualModelControl/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/vigno0405/VirtualModelControl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vigno0405/VirtualModelControl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vigno0405/VirtualModelControl/releases/tag/v0.1.0
