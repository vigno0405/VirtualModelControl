# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may change between minor versions).

## [Unreleased]

### Added

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
- Docs: tutorial "Optimizing a virtual mechanism".
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
- `vmc.Gated(component, gate)`: an element whose force and energy are multiplied by a live Param
  `gate` between 0 and 1. `optimization.Sparsity(weight, *patterns)` adds the sum of the free Params
  that match (each at least 0) to the cost. With the gates free, the optimizer keeps the elements
  that earn their place: structure optimization, tested on a mass with three springs and on the
  soft arm with five repulsive fields, and shown in the optimization tutorial ("Which fields to
  keep").
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
