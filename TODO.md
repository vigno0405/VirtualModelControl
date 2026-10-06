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
  PyTorch versions are generated from that graph (release 0.7.0), never maintained by hand, so
  they can never disagree.
- **Everything is a parameter.** Gains and references, and also all geometry (lengths, radii,
  tendon angles, joint axes, mounting poses), masses, transmissions, motor constants and
  efficiencies: each is a `Param` with a unit, a default and bounds. Defaults live in `robots/`
  data, never in model code.
- **Jacobians and Hessians.** Every model and coordinate gives both, by automatic
  differentiation, tested against finite differences.
- **SI units inside.** Degrees exist only in the hardware profiles; a driver's raw units
  (encoder ticks, motor currents) never enter the library.
- **Communication-agnostic.** The library turns measurements into commands; how they reach the
  robot (the lab's ROS Dynamixel driver, or any other platform) stays outside it. A plant is
  anything with `read` and `write`.
- **Everything has a release.** No item waits in a backlog: each one has its release, and 1.0.0
  holds them all.
- **Opt-in.** New behaviour never changes how finished experiments run; projects pin a version.
- **Modular.** Each subpackage works without the ones above it. Someone with only a kinematic
  model builds and runs a controller without importing simulation, optimization or learning code.
- **Safety.** Nothing touches a real robot without the owner's go-ahead. No torque limits unless
  asked for (they are opt-in output stages).
- **Light on the CPU.** At most two or three processes; optimizers on one core.
- **Double-check everything, as thoroughly as possible.** Every change is verified once while it
  is made and again in a separate review from a clean state (fresh clone, fresh environment,
  clean docs build). Numbers in the docs come from the code, every documented command is run,
  every figure and animation is looked at, and anything not verified is said so.
- **Done means** tests (derivatives checked against finite differences where relevant), a
  documentation page or section with at least one figure, a `CHANGELOG.md` line, green CI, and
  the box ticked here.

## Where things stand (5 October 2026)

- **0.3.0 is on PyPI** (see `CHANGELOG.md`): the bimanual, hand, turtle and UR5 templates;
  Jacobians and Hessians of every site; experiments in YAML files (`vmc.config`); run logs that
  save, load, export, replay and compare; the real-time loop and swaps with their
  documentation; hardware profiles;
  every robot's efficiency is 1 by default, and `vmc.Efficiency` gives a polynomial one per
  motor; `vmc.identification` fits efficiencies and stiffness and damping from data; the
  constrained springs and dampers are components of their own; contact; figures and animations;
  and the rebuilt documentation.
- **Decided on 5 October 2026:** the library is communication-agnostic, so `vmc.ros` and the
  direct Dynamixel path (the motor plant, homing, the bus check) are deprecated in 0.3.0 and
  removed in 0.4.0. After 0.3.0 the releases follow the research: optimization of virtual
  mechanisms, then passivity, adaptation and estimation, MPC and underactuation, and models in
  PyTorch and numpy. Every item has its release, up to 1.0.0.
- **The library is for control only (5 October 2026):** the interface tools of 0.3.0 (the window, the
  keyboard, the joystick, the session and their mailbox) and the real-time simulation clock are
  removed, and the 3D views item is dropped. Changing a running controller stays possible through
  `controller.set`, schedules and swaps.
- **Rescoped on 5 October 2026:** after 0.5.0 the plan is short: MPC and underactuation (0.6.0),
  models in PyTorch and numpy (0.7.0), then a full check and the 1.0.0 release. Co-design with
  variable-stiffness actuators, URDF and MuJoCo, continuum models beyond PCC, inverse dynamics,
  realism wrappers, learning from demonstrations and the other model backends are not planned.
- **To update an installed copy,** see "Update" on the documentation's Installation page:
  `pip install --upgrade virtualmodelcontrol`, or the newest `main` from GitHub with
  `pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"`.

## The plan, step by step

Everything still to do, in order. Each step is done when its items below are ticked.

1. **Review the repository** (done, 5 October 2026).
2. **Finish 0.3.0** (done, 5 October 2026): configurations, run logs, the
   deprecations, the documentation of the real-time loop, a safety checklist, element swaps and
   schedules, and the release.
3. **0.4.0, optimization of virtual mechanisms:** problems, integrators and compiled rollouts;
   the lab's offline optimization; structure optimization; gradient-free tuning; the energy
   tank for online updates; the step experiment and calibration; the deprecated parts
   removed.
4. **0.5.0, passivity, adaptation, estimation and locomotion:** adaptation laws and passivity
   filters; grasp-force tracking on two arms and the hand's fingertip laws; Kalman filters and
   force, stiffness and shape estimates; trees, joints, floating bases, more coordinates and
   components, more contact; the turtle crawling; parity with VMRobotControl.jl.
5. **0.6.0, MPC and underactuation:** MPC; underactuated VMC and its templates; robots with
   unmeasured coordinates; custom dynamics for a `FunctionModel`.
6. **0.7.0, models in PyTorch and numpy:** every model as a numpy function and a PyTorch
   function, generated from the CasADi graph.
7. **1.0.0:** check everything, from a fresh clone, and publish.

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

- [x] `vmc.viz.animate(robot, log, ...)`: animation of a run (2D projections), saved as MP4
  (H.264) or animated WebP/GIF under 2 MB.
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
  real robot comes with 0.3.0, using a model outside CasADi with 0.7.0, a new solver with 0.4.0.
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

## 0.3.0: experiments in files and run logs

Everything done by hand around an experiment becomes a file or one library call: describing it,
running it in simulation or in real time on any plant, and recording it.
How the commands reach the robot stays outside the library.

- [x] **Review the repository:** read every package (core, mechanisms, models, compiler,
  dynamics, control, sim, identification, hardware, ros, robots, viz), the tests, every
  documentation page, the README, the packaging and CI, looking for bugs, mistakes, unclear
  text and improvements. Done (5 October 2026): six findings, all fixed. Four have a test that
  fails without the fix: the run loop's guard read the plant's `u`, which the plant contract
  does not require; the constrained elements named their `normal` Param `direction`; `vmc.ros`
  was not reachable as documented; SciPy made up three quarters of the import time. Two were
  texts: a docstring and the README's SciPy row.
- [x] **Hardware profiles** (`hardware/profile.py`): motor IDs and order, signs, held motors and
  the driver's bus order; YAML; one per robot template; the conversions (degrees and radians,
  torque) tested both ways. They stay when the direct Dynamixel path leaves: any transport needs
  them. The raw motor units (ticks, currents, motor constants, baud rate) were removed in 0.4.0.
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
- [x] **Deprecations:** `vmc.ros` and the direct Dynamixel path (`DynamixelPlant`, `home`,
  `scan`, `latency_timer`, the buses) warn when used, and leave in 0.4.0 with dynamixel-sdk and
  pyserial: the library is communication-agnostic, and the lab drives its robots through its
  own ROS Dynamixel driver. Done: the names warn when used (a `DeprecationWarning` at the
  user's line), tested, and the CHANGELOG lists them.
- [x] **Wall-clock run loop:** `vmc.sim.WallClock` (measured steps, stale readings, rate
  statistics, overrun warnings), documented in "Real-time runs" with a checklist for the first
  run on a robot.
- [x] **Smooth element swaps and schedules:** `vmc.control.SwapController`, `blend_weight`,
  `Schedule`, `ScheduledController`, documented in "Swaps and schedules"; the planner uses the
  same blend in 0.4.0.
- [x] **Docs:** the real-time loop, the safety checklist, element swaps and schedules.
- [x] **Release 0.3.0:** checked from a fresh clone in simulation, tagged and on PyPI (5 October
  2026).

---

## 0.4.0: optimization of virtual mechanisms

Virtual mechanisms designed by optimization instead of by hand: their stiffness, references
and structure, offline with the robot's dynamics and online within an energy budget, first on
the soft arm among obstacles.

- [x] **Remove the deprecated parts:** `vmc.ros`, the direct Dynamixel path, dynamixel-sdk and
  pyserial, and the ROS CI job. Done (5 October 2026): the ROS package, the Dynamixel plant,
  homing and bus modules, their tests and the CI job are gone, the two dependencies leave the
  install, and the real-time test runs on a plain plant. Hardware profiles stay.
- [x] **Problems:** built from Params by scope (design, episode, stage) and blocks, without a
  new language. Done (6 October 2026): trajectories by trapezoidal and Hermite-Simpson
  collocation, and equilibria (`Equilibrium`), in `vmc.optimization`, with IPOPT (L-BFGS and
  `expand=True`) as the preset. Multiple shooting, free final time, periodic problems and the
  other solver presets (FATROP, SQP, QP solvers, acados) wait for what needs them: periodic
  problems and free final time for the turtle's gait (0.5.0), multiple shooting and the solver
  presets for MPC (0.6.0), data fits for estimation (0.5.0).
- [x] **Automatic scales:** the trajectory problem could derive the typical sizes of q, v and a
  itself. Decided (6 October 2026): no simple derivation holds up, so `(1, 1, 1)` stays the
  default and `Collocation(scales=...)` stays an option. Measured on the soft arm (20 nodes): on
  the reaching problems every choice reaches the same plan in 14 to 34 iterations. On the
  avoidance problem with free placement, the lab's hand-tuned scales `(0.05, 0.3, 20)` take 29
  iterations and stop at a poor local optimum (the spring near the base); no scaling takes 123
  iterations and finds another optimum with less than a quarter of the cost; the motion's typical
  sizes `(0.003, 0.02, 12)` take 92. The scales decide which optimum is found, so a derivation
  would have to be judged on many problems and robots, and the tutorial says to try two scalings
  and compare.
- [x] **Integrators and rollouts:** RK4, the linearly implicit step and CVODES through
  CasADi, an `ode()` that returns f(t, x) for SciPy's `solve_ivp`; closed-loop rollouts
  compiled with `mapaccum`, fast and differentiable. Done (6 October 2026): `vmc.sim.rollout`
  reproduces `vmc.sim.run` to rounding error; RK4 needs a robot that is not stiff, and CVODES is
  slow on the soft arm. IDAS was dropped.
- [x] **The lab's offline optimization, ported:** stiffness, reference and combined
  optimization of the soft arm's virtual mechanism with its dynamics; task terms (reaching,
  obstacle avoidance, effort); grid search over references; planning of element swaps with
  the controller's own blend; the results applied to the controller and checked in
  simulation; regression tests against the lab's results. Done (5 October 2026):
  `vmc.optimization` (`Problem`, `Collocation`, `Effort`, `Cost`, `Bound`, `Result.apply`,
  `search_references`) plans the closed loop on the robot's full dynamics, and agrees with the
  lab's optimizer to rounding error in its cost and constraints and to solver tolerance in its
  solved plans and its search; a plan applied to a controller and swapped to on the simulator
  gives the planned motion; tutorial "Optimizing a virtual mechanism".
- [x] **Structure optimization:** element gates g between 0 and 1 with a sparsity cost (which
  springs to keep and where to attach them). Done (6 October 2026): `vmc.Gated` puts a live gate
  on any element, `optimization.Sparsity` adds the sum of the free gates to the cost; candidate
  elements at several places keep only the ones that pay. The result depends on the weight and,
  on the arm, on the start and the scales; the tutorial says so.
- [x] **Gradient-free tuning:** an ask-and-tell interface (grid, CMA-ES, Bayesian
  optimization, extremum seeking) for episodes run in simulation or on the robot by its own
  node. Done (6 October 2026): `Grid`, `Random`, `CMAES`, `ExtremumSeeking` and `Bayes` with
  `tune`; the tuning tutorial runs three of them on the fingertip's damping.
- [x] **Energy tank:** exact bounds on parameter steps, including steps that release energy
  when the tank is empty; online updates of a running controller, such as an optimizer's
  result applied through `controller.set`, pass through it. Done (6 October 2026):
  `vmc.control.Tank` applies a change as far as the tank pays for its exact energy jump, a
  release is applied whole and refills the tank, and `result.apply(tank)` passes an optimizer's
  result through it; `controller.jump` gives the jump without applying it; section "A budget
  for changes" of the energy tutorial.
- [x] **Step experiment:** the experiment as a library tool; the stiffness and damping fit with
  a friction column, validated by simulating held-out steps; masses stay fixed. The arm mounted
  on its side needs it: its closed-loop runs cannot identify it. Done (6 October 2026):
  `identification.Steps` runs the torque steps on any plant and marks the training ones, the
  fit takes its log as it is, `friction=True` adds a static friction torque per motor, and
  `validate` simulates the held-out steps with `vmc.sim.rollout`; the hanging arm's page shows
  it. The arm on its side keeps another arm's values until someone runs the experiment on it.
- [x] **Calibration:** transmission ratios and base transforms between arms. Done (6 October
  2026): the templates carry the constants of the lab's models (the transmissions fitted from the
  finger's and the hand's sweeps, the arm bases of the bimanual kinematics, placed by hand, not
  measured), and `identification.fit_transmission` fits a new drive's ratio from a sweep and
  reproduces them. No calibration protocol for friction: the compensation stage keeps its
  Stribeck defaults.
- [x] **Docs:** optimizing a virtual mechanism (the tutorial ends with the soft arm planned around a
  sphere), the energy tank, the step experiment (the hanging arm's page), rollouts (Real-time
  runs) and tuning with a search (Tuning stiffness and damping). Done (6 October 2026); each
  item below that is still open brings its own section.

---

## 0.5.0: passivity, adaptation, estimation and locomotion

Controllers that change while they run without losing passivity, the estimates they need, and
the first floating robot: the two arms tracking a grasp force, the hand's fingertips, the
turtle crawling.

- [x] **Energy accounting and passivity checks** from logs. Done (6 October 2026):
  `vmc.sim.energy_balance(log)` gives a run's energy, the work given through the port, what the
  dampers took and the sources gave, the energy injected by changes of live Params and the
  margin that stays above 0 while the controller is passive; section "The balance from a log"
  of the energy tutorial.
- [x] **Adaptation laws:** stiffness modulation, reference modulation, direct stiffness
  tracking, integral pose regulation, stiffness schedules K(F) and K(d), force tracking by
  reference or by stiffness gradient descent; every stiffness update symmetrized and projected
  onto positive semidefinite matrices.
  Done (6 October 2026): `vmc.adaptation.ForceTracking`, gradient descent of a contact
  force error on any live Params, a spring's goal or its stiffness, the step bounded by the
  force it may change or a fixed rate, by the Param's bounds, by the positive semidefinite cone
  for a matrix, and by the tank; `ForceRatio` and `Stiffening`, the lab's laws that need no
  model (K(d) is the `SigmoidSpring` and the `PolynomialSpring`); the tutorial "Track a contact
  force" uses them; `StiffnessTracking` (direct stiffness tracking) and `PositionRegulation`
  (integral pose regulation), in the tutorial "Shape the stiffness of the tip"; `HoldingGoals`,
  the open-loop counterpart of the pose regulation (the lab's position feedforward): the goals
  that hold points of the arm at wanted positions, in the same tutorial.
- [x] **Passivity filters:** every online update can pass through the tank or a projection,
  opt-in. Done (6 October 2026): the tank (`vmc.control.Tank`) takes in what the controller's
  dampers take when it runs in place of the controller, and `vmc.control.project_psd` gives the
  nearest symmetric positive semidefinite stiffness; the energy tutorial shows both.
- [x] **Grasp-force tracking on the two arms:** the lab's tank-based algorithm, open and
  closed loop, ported with a regression test. Done (6 October 2026): one `ForceTracking` and one
  `ContactForce` per arm, told its own estimate (open loop) or a sensor on the object (closed
  loop), alone or through a `Tank`. The laws and the estimate agree with the lab's (golden tests);
  the tank is the library's, whose exact energy jump replaces the lab's first-order bound and its
  safety factor, so a run through it is tested for what it must do (it stalls when empty and
  reaches the force when funded), not against the lab's numbers. The "Two arms" example holds an
  object with a chosen force.
- [x] **The hand's fingertip laws:** fingertip force and stiffness optimization (stiffness-
  and reference-based gradient descent, the heuristic laws), toward grasp stability with
  fingertip sensing. Done (6 October 2026): the two gradient laws, `ForceTracking(..., rate=)` on
  the joint-space stiffness and on the joint reference at the lab's learning rates, agree with
  the finger's and run on the hand's fingertips; the heuristic laws are `ForceRatio` (the hand's
  multiplicative law) and `Stiffening` (the finger's K(F)), equal to the lab's; the finger's page
  tracks two force levels, and the force tutorial runs the heuristic laws.
- [ ] **Estimation:** measurement models shared by simulated sensors and estimators (encoders,
  motion-capture markers, IMU relative rotations, load cells; models only, no sensor readers);
  Kalman filters (EKF and UKF) fusing encoders with motion capture or IMUs, with per-sensor
  gating, health flags and staleness; a soft arm's shape from IMUs or motion capture by
  kinematic inversion, with velocities at each sensor's own rate; contact force from the
  virtual springs (virtual work) and from motor torques; task-space stiffness (congruence
  transformation, Hessian terms included); object compliance by probing; a momentum observer
  for external forces; linear-in-parameters regression from the dynamics residual, and
  nonlinear least squares, and moving-horizon estimation as a problem of `vmc.optimization`.
  Done so far (6 October 2026): `vmc.estimation.ContactForce`, the contact force from the
  controller's command and the robot's model, which gives the open-loop force of the adaptation
  laws; `TaskStiffness`, the task-space stiffness with the Hessian terms (the exact one of the
  lab; the first-order one is not provided); the tutorials "Track a contact force" and "Shape
  the stiffness of the tip" use them.
  `KalmanFilter` and `Measurement` (6 October 2026): the robot's own dynamics linearised at the
  estimate, per-sensor gating with `rejected`, `rejected_total` and `missing`, partial sensors,
  encoders through the transmission; its state, covariance, gate and health agree with the lab's
  filter over a run with dropouts and outliers. The shape from motion capture: `Inversion`
  (damped Gauss-Newton on the robot's kinematics, any robot) and `VelocityFilter`. The shape from
  IMUs: `ImuFilter`, the complementary filter of the sections' relative rotations of the soft
  arm. The tutorial "Estimate the state of a soft arm" fuses all three sensors.
  Object compliance by probing (6 October 2026): `object_compliance`, the lab's probe of the hand
  (the distance over the force that stiffer settings add to a gentle one, medians of the samples),
  equal to the lab's code; the force tutorial probes three objects with it.
  `MomentumObserver` (6 October 2026): the external force from the momentum, with no acceleration,
  as a first-order low-pass of the true force; the force tutorial shows it on a landing finger.
  Open: the UKF variant of the filter, regression and nonlinear least squares, and
  moving-horizon estimation.
- [x] **Bring your own kinematics:** `FunctionModel`, a user function `frame(q, at, p)` written
  with CasADi operations or with `vmc.math` (a small set of functions that run on numpy arrays
  and CasADi symbols alike); kinematic trees and fixed joints; joint types (revolute,
  prismatic, helical, spherical, a rail along a spline path, joints driven by a reference or
  by a function of time); floating bases (SO(3) and SE(3) with their Lie helpers);
  `vmc.testing.check_model(model)`, the model contract in one call (derivatives against finite
  differences, orthonormal rotations, continuity across links and segments, energy
  conservation without damping, no NaN at the defaults, the bounds and singular poses,
  serialization round trip).
  Done so far (6 October 2026): `vmc.models.FunctionModel` (a model from a function) and
  `vmc.testing.check_model(model)`, the model contract in one call for a model or a robot's
  mechanism (finite, rotations, derivatives against finite differences, continuity along `s`,
  serialization round trip, energy).
  Also done (6 October 2026): `vmc.math` (rotations and rigid transforms on numpy and CasADi
  alike), the joint types `helical`, `spherical` and `free` of `SerialChain` (a floating base,
  with its rotation as a rotation vector), trees as parts mounted on sites by `Assembly`.
  Done (6 October 2026): the joint `("rail", waypoints)`, a slide along the natural cubic spline
  through the waypoints (it agrees with SciPy's, and a bead on a circular wire swings as a
  pendulum); the joint `"floating"`, a floating base on a unit quaternion (`vmc.Quaternion`, with
  the Euler-Poincare term in the dynamics), which turns through several full turns by Euler's
  equations; and joints driven by a reference or by a function of time, as a virtual model at the
  value of any coordinate (`FramePoint(model, site, q=coordinate)`, with `vmc.Time()` and
  `Custom`), which a controller uses as a cart on a rail. A robot's own joint is driven the same
  way, by a stiff spring to a reference or to a function of time.

- [x] **More coordinates and components:** the sum of coordinates, an orientation error, the
  angular velocity of a frame, a vector expressed in a frame and back; one-sided springs and
  dampers, a diode damper (damps one direction of motion only), a linear inerter, rigid-body
  inertias with an angular-velocity coordinate, rotational springs on an orientation error,
  mass distributed along a continuum by quadrature, force- and power-limited sources, a helper
  that adds deadzone springs at joint limits.
  Done (6 October 2026): `Sum` and the plus sign, `FrameRotation`, `OrientationError` (a rotational
  spring is a spring on it; a damper on it damps the frame's angular velocity relative to its goal,
  and there is no coordinate of the angular velocity itself, which is not the derivative of any
  function of q), `InFrame` and `FromFrame`, `RotationalInertia` (a rigid body's inertia on the
  frame's rotation matrix, with a `PointMass`, in place of an angular-velocity coordinate),
  `DiodeDamper`, a `ForceSource` bounded in force and in power, `Mechanism.add_mass_along`. The
  one-sided springs and dampers are the contact ones, the linear inerter is an `Inertance` on a
  difference, and soft joint limits are a `LimitSpring` on a slice of the joints.
- [x] **Contact, continued:** tangential friction (smooth Coulomb); contact forces reported
  per component by the simulator; contact between two robot points (self-contact, two arms);
  soft objects. Done (6 October 2026): boxes, cylinders and capsules (`BoxDistance`,
  `CylinderDistance`, `CapsuleDistance`, also in configuration files); `ContactFriction`, the
  smooth Coulomb friction on any of these surfaces and on the plane and the sphere, with the
  contact spring's own normal force; the force, rate and torque of each component of the robot,
  from the simulator (`ModelPlant.elements()`) and in a run's log (`record="robot"`), which gives
  the contact forces. Contact between two robot points (a self-contact, the two arms) is
  `ContactSpring(Norm(b - a) - width, k)`, and its friction `ContactFriction` on
  `SphereDistance(b - a, 0, width)`; a soft object is a part with coordinates and springs of its
  own, in series with the contacts: no new pieces, tested (`tests/mechanisms/test_pair_contact.py`)
  and shown in the contact tutorial.
- [ ] **The turtle crawling:** its floating body on the ground in the library's own simulator;
  locomotion elements (phase-modulated stiffness, saturating potentials, steering, a
  series-VSA potential); extremum seeking of gaits and gains under a passivity cap.
  Its gait needs periodic trajectory problems (the orbit repeats, with references that change in
  time) and free final time (the period is an unknown), both built on `vmc.optimization`. The
  force laws and the estimators work in motor coordinates and need as many velocity coordinates as
  configuration coordinates; a floating body has fewer, so they are to be extended for it.
- [ ] **Parity with VMRobotControl.jl** (the Julia library that shares this library's
  vocabulary of coordinates and components): virtual mechanisms with their own kinematics (a
  virtual cart on a rail along a path, a virtual tool); a table that maps its features to this
  library's; examples mirroring it (reaching with obstacle avoidance, compliant path
  following, a pendulum on a spline rail).
- [ ] **Docs:** adaptation and passivity filters; estimation; the turtle crawling; bring your
  own kinematics (a function, DH or product-of-exponentials data), each ending in a working
  controller without simulation or optimization imports.

---

## 0.6.0: MPC and underactuation

Controllers that look ahead, and robots with passive joints or unmeasured coordinates.

- [ ] **MPC:** initial state and references as parameters, shifted warm start, a horizon of
  stiffness and reference trajectories under passivity (tank) constraints; runs asynchronously
  and applies its result through `controller.set` with the measured latency; a
  real-time-iteration option.
  It brings multiple shooting (one rollout per interval, as in `vmc.sim.rollout`) and the solver
  presets that suit it (FATROP, SQP with a QP solver, acados optional).
- [ ] **Underactuated VMC:** actuation projector, torque defect, feasible force set; naive and
  frozen-base controllers; passive and tank corrections; direction-constrained force tracking;
  each a flag, so the old and new behaviour compare; an underactuated allocation that reports
  the part it cannot realize.
- [ ] **Robots where some coordinates are not measured:** controllers that act through an
  estimate of them.
- [ ] **Underactuated templates:** a planar three-link arm with one passive joint, a five-link
  arm with three passive joints, a two-tendon continuum arm.
- [ ] **Custom dynamics:** a residual r(q, v, a, u, f_ext, p) = 0 with an optional energy,
  attached to a `FunctionModel`, for black-box or external models. The passivity tools need the
  energy and refuse without it.
- [ ] **Docs:** MPC; underactuation; custom dynamics.

---

## 0.7.0: models in PyTorch and numpy

- [ ] **Every model in PyTorch and numpy:** one translator from a CasADi function's instruction
  list to numpy source (it runs without CasADi installed) and to PyTorch (differentiable); one
  way to ask for either, such as `vmc.Kinematics(robot).function("tip", backend="numpy")` and
  `dynamics.export("torch")`; both agree with CasADi to 1e-12 on random inputs, Jacobians and
  Hessians included.
- [ ] **Docs:** "Use a model outside CasADi".

---

## 1.0.0

- [ ] **Check everything and publish:** every page, example and documented command run from a
  fresh clone on each supported Python and numpy; API review and freeze; deprecated names
  removed; a support policy; the release.
- [ ] **Review all the documentation** page by page, so that the final version is ready: text,
  numbers, figures, animations and links, at desktop and phone width.

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
- **Locomotion with VMC instead of central pattern generators:** virtual flywheel, ground contact,
  gait tuning (turtle). (Flywheel done; crawling in 0.5.0.)
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
| live gain changes | `controller.set`, schedules and swaps | 0.3.0 |
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
  another Helyx arm, an assumption: its closed-loop runs cannot identify it, since its tendons go
  slack under small torques. The step experiment (`identification.Steps`) identifies it on the arm
  when someone runs it.
- The hanging arm's bus (`helyx.hardware("145-290-290")`): IDs 1 to 9 at 4 Mbaud with the
  motor constant 0.001783, from its start script; its documentation says IDs 11 to 19. Confirm
  on the arm.
- The finger's distal phalanx weighs 0.025 kg in the finger's parameters but 0.0025 kg in the
  hand's: weigh it.
