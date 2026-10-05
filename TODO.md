# TODO

The plan of record for `virtualmodelcontrol`: everything still to build, in the order to build it.
Work from top to bottom. Each item says **what** to build, **why** it is needed, **how** (when the
design is settled) and **done when** (the checks that close it). Ticked items are done.

## How to work on this list

- **Order.** Finish a release before starting the next one, unless an item blocks a running
  experiment.
- **Port before inventing.** Many items port code that already runs in the lab (identification,
  Kalman filter, energy tank, offline optimization, adaptation laws). Read the original first,
  write down the differences, and add a regression test against its results (small fixtures in
  `tests/data/`).
- **One symbolic source.** Every model is written once, with CasADi operations. The numpy,
  sympy, C, JAX and PyTorch versions are generated from that graph (release 0.8.0), never
  maintained by hand, so they can never disagree.
- **Everything is a parameter.** Gains and references, and also all geometry (lengths, radii,
  tendon angles, joint axes, mounting poses), masses, transmissions, motor constants and
  efficiencies: each is a `Param` with a unit, a default and bounds. Defaults live in `robots/`
  data, never in model code.
- **Jacobians and Hessians.** Every model and coordinate gives both, by automatic
  differentiation, tested against finite differences.
- **SI units inside.** Degrees, encoder ticks, grams and motor currents exist only at the
  hardware boundary.
- **Communication-agnostic.** The library turns measurements into commands; how they reach the
  robot (the lab's ROS Dynamixel driver, or any other platform) stays outside it. A plant is
  anything with `read` and `write`.
- **Everything has a release.** No item waits in a backlog: each one has its release, and 1.0.0
  holds them all.
- **Opt-in.** New behaviour never changes how finished experiments run; projects pin a version.
- **Modular.** Each subpackage works without the ones above it. Someone with only a kinematic
  model builds and runs a controller without importing simulation, optimization or learning code.
- **Safety.** Nothing touches a real robot without the owner's go-ahead. ROS tests (until 0.4.0)
  use a private `ROS_DOMAIN_ID` between 70 and 79, never 0. No torque limits unless asked for
  (they are opt-in output stages).
- **Light on the CPU.** At most two or three processes; optimizers on one core.
- **Double-check everything, as thoroughly as possible.** Every change is verified once while it
  is made and again in a separate review from a clean state (fresh clone, fresh environment,
  clean docs build). Numbers in the docs come from the code, every documented command is run,
  every figure and animation is looked at, and anything not verified is said so.
- **Done means** tests (derivatives checked against finite differences where relevant), a
  documentation page or section with at least one figure, a `CHANGELOG.md` line, green CI, and
  the box ticked here.

## Where things stand (5 October 2026)

- **0.2.0 is on PyPI** (see `CHANGELOG.md`): the bimanual, hand, turtle and UR5 templates;
  Jacobians and Hessians of every site; mounting parts on frames; opt-in output stages; virtual
  states; figures and animations; contact; the rebuilt documentation; and the first parts of
  0.3.0 (hardware profiles, the Dynamixel plant, the real-time run loop, the ROS 2 bridge).
- **On `main`, for 0.3.0** (see `CHANGELOG.md`, "Unreleased"): the repository review and its
  fixes; experiments in YAML files (`vmc.config`); run logs that save, load, export, replay and
  compare; control by hand that records itself as a schedule (`vmc.interactive`); the ROS
  controller node and live element swaps; every robot's efficiency is 1 by default, and
  `vmc.Efficiency` gives a polynomial one per motor; `vmc.identification` fits efficiencies and
  stiffness and damping from data; the constrained springs and dampers are components of their
  own; the two single-arm examples run their own arm's experiments; concepts pages on Virtual
  Model Control, passivity, finger and hand kinematics and efficiency.
- **Decided on 5 October 2026:** the library is communication-agnostic, so `vmc.ros` and the
  direct Dynamixel path (the motor plant, homing, the bus check) are deprecated in 0.3.0 and
  removed in 0.4.0. After 0.3.0 the releases follow the research: optimization of virtual
  mechanisms, then passivity, adaptation and estimation, MPC and underactuation, co-design,
  learning. Every item has its release, up to 1.0.0.
- **To update an installed copy,** see "Update" on the documentation's Installation page:
  `pip install --upgrade virtualmodelcontrol`, or the newest `main` from GitHub with
  `pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"`.

## The plan, step by step

Everything still to do, in order. Each step is done when its items below are ticked.

1. **Review the repository** (done, 5 October 2026).
2. **Finish 0.3.0:** configurations (done); run logs; interactive tools; the deprecations; the
   documentation of the real-time loop, a safety checklist, element swaps and schedules; then
   the release, checked in simulation, with the owner's go.
3. **0.4.0, optimization of virtual mechanisms:** problems, integrators and compiled rollouts;
   the lab's offline optimization; structure optimization; gradient-free tuning; the energy
   tank for online updates; the step experiment and calibration; 3D views; the deprecated parts
   removed.
4. **0.5.0, passivity, adaptation, estimation and locomotion:** adaptation laws and passivity
   filters; grasp-force tracking on two arms and the hand's fingertip laws; Kalman filters and
   force, stiffness and shape estimates; trees, joints, floating bases, more coordinates and
   components, more contact; the turtle crawling; parity with VMRobotControl.jl.
5. **0.6.0, MPC and underactuation:** MPC; underactuated VMC; continuum models beyond PCC;
   bring your own dynamics; URDF import and export, serialization, a MuJoCo plant; realism
   wrappers.
6. **0.7.0, co-design:** actuator models; co-design structures and optimization across time
   scales; grasp transitions with a variable-stiffness actuator.
7. **0.8.0, learning:** every model in PyTorch, JAX, numpy, sympy and C; datasets; parameter
   networks; imitation learning with diffusion models; learned residual dynamics.
8. **1.0.0:** API review and freeze, with everything above in.

---

## 0.2.0: finish what is started, and documentation for everyone

The library already does a lot; this release makes it learnable. A student who has never seen
Virtual Model Control must be able to control a robot from the documentation alone.

### Done

- [x] Every dependency installs with the package (numpy, SciPy, CasADi, matplotlib, PyYAML,
  Dynamixel SDK, pyserial)
- [x] CI on Ubuntu 22.04 and 24.04 (Python 3.10 to 3.14, numpy 1.26 and 2.x), plus a clean install
- [x] Robot templates: Helyx soft arm (three geometries), bimanual Helyx, ADAPT finger, ADAPT
  hand, turtle, UR5, and the hand on the UR5 flange
- [x] Every template parametric in its geometry (lengths, radii, tendon angles, mounts, masses,
  transmissions, DH tables), today's values as defaults
- [x] Jacobians and Hessians of every site (`vmc.Kinematics`)
- [x] Mount a part on another part's frame; orientations of serial-chain sites
- [x] Speed regulator for virtual states
- [x] Opt-in output stages: friction compensation, pretension, torque offset, torque limit
- [x] Figures in the lab style, saved as PDF and SVG (`vmc.viz`)
- [x] Contact as stiff one-sided springs: `PlaneDistance`, `SphereDistance`, `ContactSpring`,
  `ContactDamper`, `adapt.add_dynamics`; guide page "Contact"
- [x] README with install, update and development instructions

### Documentation overhaul (the priority)

Decided (4 October 2026): the site uses the Material layout (sphinx-immaterial): five sections
as tabs in the top bar (Getting started, Tutorials, Examples, Concepts, Reference), the pages of
each section in the left sidebar, large type and the plain style of the best library docs
(VMRobotControl.jl, pykoopman). Pages are complete but essential:
each fact once, linked elsewhere. Schematics and figures are drawn by code from the robots'
Params. Robots get no pages of their own and no videos of the real hardware: each example opens
with its robot.

**Schematics.** Every concept gets a picture before any code, drawn by code (`docs/schematics/`)
in the lab style, readable on a phone:

- [x] Library map: the layers (core, mechanisms, models, compiler and dynamics, control, sim,
  then estimation, optimization and learning) with the edges (hardware, ROS, robots, viz), and
  which ones a controller-only user needs.
- [x] "What a mechanism is": a robot drawing with its coordinates (points, joints, distances)
  and components (springs, dampers, masses, sources) attached; the same drawing for a controller.
- [x] The control loop: plant (simulated or real) and controller exchanging measurements and
  torques (read, step, write, advance), with the guard, output stage, logger and ROS bridge in
  place.
- [x] Energy flow: storage, dissipation, sources, and the power balance the library checks.
- [x] One schematic per robot: frames and sites, joints, motors in their order, tendons and their
  angles, sign conventions, units.

**Figures.** Every guide shows results as plots (trajectories, forces, energies), made when the
docs build. Nothing is pasted as an image if it can be computed.

**Animations.**

- [x] `vmc.viz.animate(robot, log, ...)`: animation of a run (2D projections now, 3D with
  meshes after 0.4.0), saved as MP4 (H.264) or animated WebP/GIF under 2 MB.
- [x] Every page that simulates embeds the animation of its own run, made by the docs build.

**Pages to write or rewrite.**

- [x] Getting started: install (pip, venv, conda; Ubuntu 22.04 and 24.04; with ROS sourced),
  update and uninstall; first controller with a figure, simulated, with plots and an animation
  (one tutorial).
- [x] Tutorials: how it works; coordinates and components; parameters and live changes; energy
  and passivity; kinematics (frames, sites, Jacobians, Hessians, task-space stiffness); tuning
  (choosing stiffness and damping, the loop-delay limit on damping, reading energies); contact;
  build a robot (from DH or product-of-exponentials data, a continuum model, joint space only,
  actuation and couplings, masses, a template function, the tests); extend the library (new
  component, coordinate, model, plant; plugins through entry points). From simulation to the
  real robot comes with 0.3.0, using a model outside CasADi with 0.8.0, a new solver with 0.4.0.
- [x] Examples: soft arm, two arms, finger, hand (and on the UR5), turtle; each opens with its
  robot: a schematic, a parameter table (name, value, unit, meaning), calibration constants, the
  output stage, the signs, and how to build it with other numbers.
- [x] Concepts: the library's structure (layers and contracts), soft-arm kinematics,
  conventions and glossary.
- [x] Reference: API with short names and no cut text; changelog; this list.
- [x] Developer notes: contributing, with tests, documentation rules and releases.
- [x] Troubleshooting: ROS on the `PYTHONPATH`, missing `python3-venv`, CasADi import errors,
  first compile time, matplotlib and LaTeX.
- [x] Create a logo for the library: in the site header, as the favicon and at the top of the
  README.

**Quality bar.**

- [x] The build passes with `-W` (warnings fail) and every code block executes.
- [x] Every page checked at desktop and phone width (screenshots) for cut letters, overlaps and
  unreadable figures.
- [x] No paper references and no project repository names anywhere.
- [x] Plain-list references are named `ref` (`point - [x, y, z]` becomes `ctrl.<name>.ref`): say
  so where `controller.set` is explained, and use `vmc.Ref("goal", ...)` in examples that change
  goals live.

Done when every page above exists, each page that simulates has figures and an animation, each
example opens with its robot's schematic, and the build and the screenshots are clean.

### Release

- [x] **Transmission efficiency in the templates**, as in the lab's own models: the robot
  receives η times the commanded torque, and its own stiffness and damping are the physical,
  identified values (the soft arm kept η = 1 and divided its stiffness and damping by 0.12: the
  same equilibria without gravity, but natural frequencies about 2.9 times too high). Soft arm
  and two arms 0.12 in the tendon transmission; the finger and the hand stay lossless by default,
  their measured per-motor efficiencies opt-in (`Direct(efficiency=...)`), because unequal
  efficiencies on coupled motors make virtual elements non-conservative; controllers never
  divide by η. Done when its two failing tests are updated, every page that simulates these
  robots is re-run and its text checked against the new numbers, and the CHANGELOG says what
  changed. Changed back in 0.3.0: every robot's efficiency is 1 by default (see below).
- [x] README: show
  `pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"` in the
  "Install" section too, right under `pip install virtualmodelcontrol`, as the way to get the
  current `main` before a release (today it appears only under "Update"). Run it end to end in a
  scratch environment: install at one commit, push a newer one, run the command again, and check
  that `vmc.__version__` changed.
- [x] Fill `CHANGELOG.md`, tag `v0.2.0`, check the PyPI upload, then run
  `pip install --upgrade virtualmodelcontrol` in clean environments on Ubuntu 22.04 and 24.04.

---

## 0.3.0: experiments in files, run logs, interactive tools

Everything done by hand around an experiment becomes a file or one library call: describing it,
running it in simulation or in real time on any plant, recording it, changing it while it runs.
How the commands reach the robot stays outside the library.

- [x] **Review the repository:** read every package (core, mechanisms, models, compiler,
  dynamics, control, sim, identification, hardware, ros, robots, viz), the tests, every
  documentation page, the README, the packaging and CI, looking for bugs, mistakes, unclear
  text and improvements. Done (5 October 2026): six findings, all fixed. Four have a test that
  fails without the fix: the run loop's guard read the plant's `u`, which the plant contract
  does not require; the constrained elements named their `normal` Param `direction`; `vmc.ros`
  was not reachable as documented; SciPy made up three quarters of the import time. Two were
  texts: a docstring and the README's SciPy row.
- [x] **Hardware profiles** (`hardware/profile.py`): motor IDs and order, encoder signs, motor
  constants, operating modes, limits; YAML; one per robot template; the conversions (ticks and
  radians, degrees and radians, current and torque) tested both ways. They stay when the direct
  Dynamixel path leaves: any transport needs them.
- [x] **Transmission efficiency:** 1 by default for every robot, because models identified
  from the commanded torques already include their transmission (the soft arm and the two arms
  back to stiffness and damping referred to the commanded torque); `vmc.Efficiency`, the
  delivered torque as a polynomial of the commanded one, motor by motor, to tune and identify;
  `identification.plateaus` and `fit_efficiency`, checked against the lab's calibration code
  and the finger's recorded data; a concepts page.
- [x] **Configurations:** robots, controllers and experiments in YAML through the registry,
  round-trip tested. Done: `vmc.config` (robot templates, named coordinates, elements, swaps,
  simulation plants, output stages, schedules), files for every robot in
  `tests/data/configs/`, the same runs as in Python, the tutorial "Experiments in files".
- [x] **Run logs:** a run's log saved and loaded whole or not at all (`.npz` with a schema
  version and what the run was: library version, start time, the Params at the start, the
  hardware profile, the configuration); CSV export; at every step the law's torque before the
  output stages and, on request, the live Params, each element's coordinate, rate, force and
  share of the motor torques, and the controller's energies; replay of a run's commands on a
  simulated robot, and comparison of runs; a configuration's `run` settings (name, folder,
  what to record) save each run and refuse a name already taken before the run starts. Done:
  `vmc.sim.RunLog`, `replay` and `compare`, `record=` on `vmc.sim.run`, and the `run` settings of
  a configuration, tested (a replay on the true model gives the run back to rounding error),
  and the tutorial "Run logs".
- [x] **Interactive tools:** a matplotlib window (no new dependency) to drag goals, set gains and
  swap elements of a running controller in the same process (a simulation, or inside the
  robot's own node); keyboard teleoperation of a goal; joystick teleoperation (pygame, an
  opt-in extra); what was set by hand recorded and repeated as a schedule in a configuration.
  Done: `vmc.interactive` (a mailbox and a wrapper controller that work in a simulation and
  inside a node, the energy the changes give the controller, the recorder, the window, keys, the
  joystick, the session) and a real-time factor on `SimClock`, tested offscreen (a recorded
  session runs again exactly), and the tutorial "Interactive control" on the soft arm.
- [x] **Deprecations:** `vmc.ros` and the direct Dynamixel path (`DynamixelPlant`, `home`,
  `scan`, `latency_timer`, the buses) warn when used, and leave in 0.4.0 with dynamixel-sdk and
  pyserial: the library is communication-agnostic, and the lab drives its robots through its
  own ROS Dynamixel driver. Done: the names warn when used (a `DeprecationWarning` at the
  user's line), tested, and the CHANGELOG lists them.
- [ ] **Wall-clock run loop:** done in code (`vmc.sim.WallClock`, measured steps, stale
  readings, rate statistics, overrun warnings); its documentation remains (see Docs).
- [ ] **Smooth element swaps and schedules:** done in code (`vmc.control.SwapController`,
  `blend_weight`, `Schedule`, `ScheduledController`); the planner uses the same blend in 0.4.0;
  the documentation remains (see Docs).
- [ ] **Docs:** the real-time loop (`vmc.sim.run` with a `WallClock` on any plant: measured
  steps, stale readings, the guard, Ctrl-C, rate statistics); a safety checklist for a first
  run on a robot (signs, limits, a watchdog and zero torque on exit in the robot's node, stale
  data); element swaps and schedules.
- [ ] **Release 0.3.0:** a clean check from a fresh clone (in simulation), the CHANGELOG, then
  the tag with the owner's go.

---

## 0.4.0: optimization of virtual mechanisms

Virtual mechanisms designed by optimization instead of by hand: their stiffness, references
and structure, offline with the robot's dynamics and online within an energy budget, first on
the soft arm among obstacles.

- [ ] **Remove the deprecated parts:** `vmc.ros`, the direct Dynamixel path, dynamixel-sdk and
  pyserial, and the ROS CI job.
- [ ] **Problems:** built from Params by scope (design, episode, stage) and blocks, without a
  new language: trajectories (trapezoidal and Hermite-Simpson collocation, multiple shooting,
  free final time, periodic), equilibria, data fits (identification, moving-horizon
  estimation); automatic variable scaling; solver presets that work on these problems (IPOPT
  with L-BFGS and `expand=True`, FATROP, SQP and QP solvers; acados optional).
- [ ] **Integrators and rollouts:** RK4, the linearly implicit step, CVODES and IDAS through
  CasADi, an `ode()` that returns f(t, x) for SciPy's `solve_ivp`; closed-loop rollouts
  compiled with `mapaccum`, fast and differentiable.
- [ ] **The lab's offline optimization, ported:** stiffness, reference and combined
  optimization of the soft arm's virtual mechanism with its dynamics; task terms (reaching,
  obstacle avoidance, effort); grid search over references; planning of element swaps with
  the controller's own blend; the results applied to the controller and checked in
  simulation; regression tests against the lab's results.
- [ ] **Structure optimization:** element gates g between 0 and 1 with a sparsity cost (which
  springs to keep and where to attach them).
- [ ] **Gradient-free tuning:** an ask-and-tell interface (grid, CMA-ES, Bayesian
  optimization, extremum seeking) for episodes run in simulation or on the robot by its own
  node.
- [ ] **Energy tank:** exact bounds on parameter steps, including steps that release energy
  when the tank is empty; online updates of a running controller, such as an optimizer's
  result applied through `controller.set`, pass through it.
- [ ] **Step experiment:** the experiment as a library tool; the stiffness and damping fit with
  a friction column, validated by simulating held-out steps; masses stay fixed. The arm mounted
  on its side needs it: its closed-loop runs cannot identify it.
- [ ] **Calibration:** transmission ratios, motor constants, base transforms between arms,
  Stribeck friction for the compensation stage.
- [ ] **3D views:** 3D drawings with meshes; sketches of joints, frames, coordinates and
  components (coils for springs, dashpots for dampers); frame labels; 3D animations to MP4 or
  GIF; an interactive viewer in the browser that updates during a run; a distinct colour for
  the virtual mechanism.
- [ ] **Docs:** optimizing a virtual mechanism (the soft arm among obstacles); the energy
  tank; tuning without gradients; the step experiment; 3D views.

---

## 0.5.0: passivity, adaptation, estimation and locomotion

Controllers that change while they run without losing passivity, the estimates they need, and
the first floating robot: the two arms tracking a grasp force, the hand's fingertips, the
turtle crawling.

- [ ] **Energy accounting and passivity checks** from logs.
- [ ] **Adaptation laws:** stiffness modulation, reference modulation, direct stiffness
  tracking, integral pose regulation, stiffness schedules K(F) and K(d), force tracking by
  reference or by stiffness gradient descent; every stiffness update symmetrized and projected
  onto positive semidefinite matrices.
- [ ] **Passivity filters:** every online update can pass through the tank or a projection,
  opt-in.
- [ ] **Grasp-force tracking on the two arms:** the lab's tank-based algorithm, open and
  closed loop, ported with a regression test.
- [ ] **The hand's fingertip laws:** fingertip force and stiffness optimization (stiffness-
  and reference-based gradient descent, the heuristic laws), toward grasp stability with
  fingertip sensing.
- [ ] **Estimation:** measurement models shared by simulated sensors and estimators (encoders,
  motion-capture markers, IMU relative rotations, load cells; models only, no sensor readers);
  Kalman filters (EKF and UKF) fusing encoders with motion capture or IMUs, with per-sensor
  gating, health flags and staleness; a soft arm's shape from IMUs or motion capture by
  kinematic inversion, with velocities at each sensor's own rate; contact force from the
  virtual springs (virtual work) and from motor torques; task-space stiffness (congruence
  transformation, Hessian terms included); object compliance by probing; a momentum observer
  for external forces; linear-in-parameters regression from the dynamics residual, and
  nonlinear least squares.
- [ ] **Bring your own kinematics:** `FunctionModel`, a user function `frame(q, at, p)` written
  with CasADi operations or with `vmc.math` (a small set of functions that run on numpy arrays
  and CasADi symbols alike); kinematic trees and fixed joints; joint types (revolute,
  prismatic, helical, spherical, a rail along a spline path, joints driven by a reference or
  by a function of time); floating bases (SO(3) and SE(3) with their Lie helpers);
  `vmc.testing.check_model(model)`, the model contract in one call (derivatives against finite
  differences, orthonormal rotations, continuity across links and segments, energy
  conservation without damping, no NaN at the defaults, the bounds and singular poses,
  serialization round trip).
- [ ] **More coordinates and components:** the sum of coordinates, an orientation error, the
  angular velocity of a frame, a vector expressed in a frame and back; one-sided springs and
  dampers, a diode damper (damps one direction of motion only), a linear inerter, rigid-body
  inertias with an angular-velocity coordinate, rotational springs on an orientation error,
  mass distributed along a continuum by quadrature, force- and power-limited sources, a helper
  that adds deadzone springs at joint limits.
- [ ] **Contact, continued:** tangential friction (smooth Coulomb); contact forces reported
  per component by the simulator; boxes, cylinders and capsules; contact between two robot
  points (self-contact, two arms); soft objects.
- [ ] **The turtle crawling:** its floating body on the ground in the library's own simulator;
  locomotion elements (phase-modulated stiffness, saturating potentials, steering, a
  series-VSA potential); extremum seeking of gaits and gains under a passivity cap.
- [ ] **Parity with VMRobotControl.jl** (the Julia library that shares this library's
  vocabulary of coordinates and components): virtual mechanisms with their own kinematics (a
  virtual cart on a rail along a path, a virtual tool); a table that maps its features to this
  library's; examples mirroring it (reaching with obstacle avoidance, compliant path
  following, a pendulum on a spline rail; the seven-joint arm from its URDF comes with 0.6.0).
- [ ] **Docs:** adaptation and passivity filters; estimation; the turtle crawling; bring your
  own kinematics (a function, DH or product-of-exponentials data), each ending in a working
  controller without simulation or optimization imports.

---

## 0.6.0: MPC, underactuation, URDF and MuJoCo

Controllers that look ahead and robots with passive joints, continuum models beyond piecewise
constant curvature, robots from URDF files, and a second simulator to check the first.

- [ ] **MPC:** initial state and references as parameters, shifted warm start, a horizon of
  stiffness and reference trajectories under passivity (tank) constraints; runs asynchronously
  and applies its result through `controller.set` with the measured latency; a
  real-time-iteration option.
- [ ] **Underactuated VMC:** actuation projector, torque defect, feasible force set; naive and
  frozen-base controllers; passive and tank corrections; direction-constrained force tracking;
  each a flag, so the old and new behaviour compare; an underactuated allocation that reports
  the part it cannot realize.
- [ ] **Robots where some coordinates are not measured:** controllers that act through an
  estimate of them.
- [ ] **Underactuated templates:** a planar three-link arm with one passive joint, a five-link
  arm with three passive joints, a two-tendon continuum arm.
- [ ] **Continuum models beyond PCC:** affine and polynomial curvature, piecewise-constant
  strain (Cosserat), each tested against PCC where they coincide; the elongation offset as a
  Param; a PyElastica plant to validate PCC and the strain models.
- [ ] **Bring your own dynamics:** a custom residual r(q, v, a, u, f_ext, p) = 0 with an
  optional energy, for black-box, learned or external models; inverse dynamics and
  operational-space quantities (task-space inertia, operational-space force).
- [ ] **URDF and serialization:** URDF import (joints, axes, origins, limits, masses, centres
  of mass, inertia tensors, visual and collision meshes; every number a Param with the URDF
  value as its default; checked numerically against pinocchio); export to URDF and MJCF, with
  a rigid-link approximation of continuum segments; whole systems (robot and controller) to
  YAML or JSON and back.
- [ ] **MuJoCo plant** from URDF or MJCF: friction contacts, meshes, offscreen rendering to
  video; cross-fidelity tests (`ModelPlant` against MuJoCo on the UR5 carrying the hand; PCC
  against PyElastica).
- [ ] **Realism wrappers:** delays, sensor rates, noise, quantization, spikes; actuator
  friction, efficiency and torque limits.
- [ ] **Example:** impedance control of a seven-joint arm loaded from its URDF.
- [ ] **Docs:** MPC; underactuation; URDF and MuJoCo; bring your own dynamics.

---

## 0.7.0: co-design with variable-stiffness actuators

Mechanical design and control as one problem: the hardware's stiffness and rest state, the
settings of variable-stiffness actuators (VSAs) and the virtual stiffness, across their time
scales.

- [ ] **Actuator models:** motor inertia; elastic transmissions (series and parallel elastic
  elements, each grounded where the hardware grounds it); VSA potentials with slow motors;
  allocation with positive tendon tensions.
- [ ] **Co-design structures:** a series elastic transmission with a parallel elastic element
  on the motor; slow, non-backdrivable motors that set stiffness and rest angle; power
  reported for slow actuators.
- [ ] **Co-design across time scales:** design variables (geometry, VSA settings, hardware
  stiffness), episode variables and stage variables in one problem, through the Param scopes;
  scenario sets that share design variables; bilevel problems.
- [ ] **Example:** precision-to-power grasp transitions with a VSA.
- [ ] **Docs:** co-design.

---

## 0.8.0: learning

- [ ] **Every model in PyTorch, JAX, numpy, sympy and C:** one translator from a CasADi
  function's instruction list to numpy source (runs without CasADi installed), sympy
  expressions (symbolic checks and LaTeX), JAX and PyTorch (differentiable); C from CasADi's
  own code generator; one way to ask for any of them, such as
  `vmc.Kinematics(robot).function("tip", backend="numpy")`, `compiled.export("sympy")`,
  `dynamics.export("torch")`; every backend agrees with CasADi to 1e-12 on random inputs,
  Jacobians and Hessians included; a CasADi and PyTorch autograd bridge for training through
  compiled functions; docs "Use a model outside CasADi".
- [ ] **Datasets** from run logs, data-glove recordings and kinesthetic demonstrations.
- [ ] **Parameter networks:** networks that output VMC parameters, always passed through a
  passivity-consistent projection; small networks exported back to CasADi for the control
  loop.
- [ ] **Imitation learning with diffusion models:** policies trained on demonstrations that
  output VMC parameters (stiffness, references), through the same projection.
- [ ] **Learned residual dynamics** as a custom-residual model.
- [ ] **Examples:** grasp stiffness learned from demonstrations; a diffusion policy that
  reproduces demonstrated stiffness and reference changes in simulation.

---

## 1.0.0

- [ ] API review and freeze; deprecated names removed; a support policy. Every item above is
  done by then.

---

## Research the library must carry

The library exists to serve research on Virtual Model Control of compliant manipulators. Check
new designs against these themes, so nothing they need is made hard:

- **Distributed, interpretable compliance:** joint- and task-space springs on fingers and along
  continuum bodies, directional springs, force-dependent stiffness, stiffness distributions across
  the fingers of a hand, contact at many points of a chain. (Mostly done; contact friction in
  0.5.0.)
- **Model-based sensing:** contact force from spring deflection, object compliance by probing,
  efficiency calibration. (0.5.0; the efficiency fit is done.)
- **Force and stiffness regulation:** open and closed loop; gradient laws on references and
  stiffness; grasp-force tracking on two arms with an energy tank. (0.5.0)
- **Optimizing virtual mechanisms:** offline (collocation, reference search, element swaps),
  online (within an energy budget) and structure (which elements, where) in 0.4.0; MPC in
  0.6.0.
- **Underactuated and under-sensed compliant robots.** (0.6.0)
- **Co-design across time scales with VSAs:** hardware rest state and stiffness, VSA settings and
  virtual stiffness in one problem; precision-to-power grasp transitions; impact absorption with
  a wrist VSA. (0.7.0)
- **Locomotion with VMC instead of central pattern generators:** virtual flywheel, ground contact,
  gait tuning (turtle). (Flywheel done; crawling in 0.5.0.)
- **Learning VMC parameters from demonstrations, with passivity guarantees.** (0.8.0)
- **IMU proprioception of soft arms** as an estimator. (0.5.0)

## Everything done by hand in lab code, and where it lands

| Done by hand today | In the library | Release |
| --- | --- | --- |
| PCC and serial kinematics, Jacobians, Hessians | models, `vmc.Kinematics` | done |
| virtual springs and dampers, summed into torques | mechanisms, `vmc.compile`, `VMCController` | done |
| gravity compensation | `GravityCompensation` | done |
| friction compensation, pretension, torque clip | opt-in output stages | done |
| plant nodes simulating the arm | `ModelPlant`, `vmc.sim.run` | done |
| experiment configs (a Python module per experiment) | `vmc.config` YAML files | done |
| degree, sign and current conversions in every node | hardware profiles | done |
| swapping springs live, blended | element swaps with a quintic blend, schedules | done |
| efficiency calibration against a load cell | `identification.fit_efficiency` | done |
| the driver's topics, homing and bus checks in every project | the projects' own nodes and tools (communication stays outside) | none |
| controller GUIs, live gain changes, teleoperation | interactive tools, `controller.set` | 0.3.0 |
| logging runs to npz and CSV | run logs | 0.3.0 |
| step experiments and K, D fits | step experiment, `identification.fit_stiffness_damping` | 0.4.0 |
| offline optimization and reference search | optimization | 0.4.0 |
| Kalman filter fusing encoders and motion capture | estimation | 0.5.0 |
| contact-force and task-stiffness estimates | estimation | 0.5.0 |
| tank-based grasp-force tracking | passivity and adaptation | 0.5.0 |
| stiffness and reference gradient descent | adaptation | 0.5.0 |

## Open points to check with the robot owners

- `controller.set()` changes the controller's own copy of the live parameters, not the shared
  `Param` objects. Confirm this is the wanted behaviour, then document it.
- Gravity compensation of the soft arms is about 20% stronger than before the arc-formula fix:
  check it on the next real-arm run.
- The hand's thumb tip: the template keeps the point the hand's controllers attach to,
  (0, 0, 0.0175) in the last joint frame, which lies beside the thumb's last phalanx (it runs
  along −x); the end of the phalanx is about (−0.025, 0, 0), which the grasp example uses.
  Measure it, then decide which one the controllers should use.
- matplotlib 3.11 drops minus signs from LaTeX-rendered PDFs; render paper figures with 3.10 until
  it is fixed.
- The arm mounted on its side (`145-145-145`) uses the stiffness and damping identified on
  another Helyx arm: its closed-loop runs cannot identify it, since its tendons go slack under
  small torques. Identify it with the step experiment (0.6.0).
- The hanging arm's bus (`helyx.hardware("145-290-290")`): IDs 1 to 9 at 4 Mbaud with the
  motor constant 0.001783, from its start script; its documentation says IDs 11 to 19. Confirm
  on the arm.
- The finger's distal phalanx weighs 0.025 kg in the finger's parameters but 0.0025 kg in the
  hand's: weigh it.
