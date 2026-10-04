# TODO

The plan of record for `virtualmodelcontrol`: everything still to build, in the order to build it.
Work from top to bottom. Each item says **what** to build, **why** it is needed, **how** (when the
design is settled) and **done when** (the checks that close it). Ticked items are done.

## How to work on this list

- **Order.** Finish a release before starting the next one, unless an item blocks a running
  experiment.
- **Port before inventing.** Many items port code that already runs in the lab (identification,
  Kalman filter, energy tank, offline optimization, hardware scripts). Read the original first,
  write down the differences, and add a regression test against its results (small fixtures in
  `tests/data/`).
- **One symbolic source.** Every model is written once, with CasADi operations. The numpy,
  sympy, C, JAX and PyTorch versions are generated from that graph (release 0.4.0), never
  maintained by hand, so they can never disagree.
- **Everything is a parameter.** Gains and references, and also all geometry (lengths, radii,
  tendon angles, joint axes, mounting poses), masses, transmissions, motor constants and
  efficiencies: each is a `Param` with a unit, a default and bounds. Defaults live in `robots/`
  data, never in model code.
- **Jacobians and Hessians.** Every model and coordinate gives both, by automatic
  differentiation, tested against finite differences.
- **SI units inside.** Degrees, encoder ticks, grams and motor currents exist only at the
  hardware boundary.
- **Opt-in.** New behaviour never changes how finished experiments run; projects pin a version.
- **Modular.** Each subpackage works without the ones above it. Someone with only a kinematic
  model builds and runs a controller without importing simulation, optimization or learning code.
- **Safety.** Nothing touches a real robot without the owner's go-ahead. ROS tests use a private
  `ROS_DOMAIN_ID` between 70 and 79, never 0. No torque limits unless asked for (they are opt-in
  output stages).
- **Light on the CPU.** At most two or three processes; optimizers on one core.
- **Double-check everything, as thoroughly as possible.** Every change is verified once while it
  is made and again in a separate review from a clean state (fresh clone, fresh environment,
  clean docs build). Numbers in the docs come from the code, every documented command is run,
  every figure and animation is looked at, and anything not verified is said so.
- **Done means** tests (derivatives checked against finite differences where relevant), a
  documentation page or section with at least one figure, a `CHANGELOG.md` line, green CI, and
  the box ticked here.

## Where things stand (4 October 2026)

- **0.2.0 is on PyPI** (see `CHANGELOG.md`): on top of 0.1.0, the bimanual, hand, turtle and UR5
  templates; Jacobians and Hessians of every site; transmission efficiency; mounting parts on
  frames; opt-in output stages; speed regulator for virtual states; figures and animations;
  contact; the rebuilt documentation; and the first parts of 0.3.0: hardware profiles, the
  Dynamixel plant, the real-time run loop and the ROS 2 bridge.
- **To update an installed copy,** see "Update" on the documentation's Installation page:
  `pip install --upgrade virtualmodelcontrol`, or the newest `main` from GitHub with
  `pip install --upgrade "git+https://github.com/vigno0405/VirtualModelControl.git"`.

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
  real robot comes with 0.3.0, using a model outside CasADi with 0.4.0, a new solver with 0.8.0.
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

## 0.3.0: talk to the robots (hardware, ROS 2, logs)

Everything done by hand around an experiment must become one library call: homing, starting the
bus, running the controller at rate, recording, and stopping safely. The same controller object
runs in simulation and on the robot.

- [ ] **Hardware profiles** (`hardware/profile.py`): motor IDs and order, encoder signs, zero
  offsets, motor constant per motor model, baud rate, control rate, operating mode per motor
  (torque, position, velocity), limits; loaded from YAML; one per robot template. Done when the
  conversions (ticks and radians, degrees and radians, current and torque) are tested both ways.
  Progress: code and tests done (`vmc.hardware.HardwareProfile`, a profile per template);
  the documentation section remains.
- [ ] **Dynamixel plant** (through the Dynamixel SDK, no ROS): synchronous read and write; units
  from the profile; homing (go to a known pose, detect lost turns, set the zero); command
  watchdog (zero torque when commands stop); torque clamp before the integer conversion, so a
  large command can never overflow or flip sign; zero torque on exit, on exceptions and on
  Ctrl-C; a bus check (scan IDs and baud rates; USB latency timer on Linux). Done when tested
  against a fake bus in CI, then one supervised run on a real robot.
  Progress: code and fake-bus tests done (`vmc.hardware.DynamixelPlant`, `home`, `scan`);
  the supervised run on a real robot and the documentation remain.
- [ ] **Wall-clock run loop:** the simulated loop's API on real time, with the measured time step,
  a staleness guard (zero torque when readings are old), rate statistics and overrun warnings.
  Progress: done (`vmc.sim.WallClock`); the documentation remains.
- [ ] **ROS 2, any distribution** (`ros/`, rclpy and std_msgs only, imported lazily): joint I/O
  on the topics of the existing driver (positions and velocities in degrees, goal torques and
  positions, motor order, signs); `RosPlant`; `serve(sim_plant)`, which publishes the same topics
  as the driver from a simulated robot (digital twin); a controller node whose ROS parameters map
  to `controller.set`; a recorder node; messages that swap or retune virtual elements live; the
  "robot is simulated" flag. Done when an in-process run and the same run over ROS give the same
  log, and CI runs the ROS tests in Humble and Jazzy containers on a private domain.
  Progress: joint I/O, `RosPlant`, the digital twin `serve` (with a lockstep mode that
  repeats in-process runs), live parameters and the controller node `vmc.ros.control` done,
  tested in CI in Humble and Jazzy containers on a private domain, and swap messages; the
  recorder node (with the run logs) and the documentation remain.
- [ ] **Smooth element swaps:** replace one set of virtual elements by another with a quintic
  blend, in the controller and later in the planner (the same blend in both).
  Progress: done in the controller (`vmc.control.SwapController`, `blend_weight`) and over ROS
  (swap messages); the planner uses `blend_weight` in 0.8.0; the documentation remains.
- [x] **Transmission efficiency:** 1 by default for every robot, because models identified
  from the commanded torques already include their transmission (the soft arm and the two arms
  back to stiffness and damping referred to the commanded torque); `vmc.Efficiency`, the
  delivered torque as a polynomial of the commanded one, motor by motor, to tune and identify;
  `identification.plateaus` and `fit_efficiency`, checked against the lab's calibration code
  and the finger's recorded data; a concepts page.
- [ ] **Configurations:** robots, controllers and experiments in YAML through the registry,
  round-trip tested.
- [ ] **Interactive tools:** a small GUI (matplotlib widgets or Qt) to drag goals, change gains
  and swap elements while running; keyboard or joystick teleoperation; teleoperate and repeat.
- [ ] **Run logs:** atomic save and load (`.npz` with a schema version and metadata: library
  version, git hash, parameters, hardware profile), CSV export, replay of a log in simulation,
  comparison of runs.
- [ ] **Docs:** "Run on the real robot" for each robot, a ROS topics reference, a safety
  checklist.

---

## 0.4.0: open modeling (any kinematics, any dynamics, any backend, URDF and meshes)

### Bring your own kinematics

- [ ] `FunctionModel`: wrap a user function `frame(q, at, p) -> (R, p)` written with CasADi
  operations, or with `vmc.math` (a small set of functions that run on numpy arrays and on CasADi
  symbols alike), plus its named sites and Params.
- [ ] Kinematic trees (branching chains) and fixed joints; joint types: revolute, prismatic,
  helical, spherical, a rail along a spline path, and joints driven by a reference or by a
  function of time.
- [ ] Floating bases: SO(3) and SE(3) spaces with their Lie helpers.
- [ ] `vmc.testing.check_model(model)`: the model contract in one call (derivatives against finite
  differences, orthonormal rotations, continuity across links and segments, energy conservation
  without damping, no NaN at the defaults, the bounds and singular poses, serialization round
  trip).
- [ ] Docs, "Bring your own kinematics": three ways (a function, DH or product-of-exponentials
  data, a URDF), each ending in a working controller without simulation or optimization imports.

### Bring your own dynamics

- [ ] Custom residual `r(q, v, a, u, f_ext, p) = 0`, with an optional energy, for black-box,
  learned or external models.
- [ ] Rigid-body inertias (3 × 3) with an angular-velocity coordinate; inerters; rotational
  springs with an orientation-error coordinate; mass distributed along a continuum by quadrature.
- [ ] Actuator models: motor inertia; elastic transmissions (series and parallel elastic elements,
  each grounded where the hardware grounds it); VSA potentials with slow motors; allocation with
  positive tendon tensions; underactuated actuation that reports the part it cannot realize.
- [ ] Continuum models beyond PCC: affine and polynomial curvature, piecewise-constant strain
  (Cosserat), each tested against PCC where they coincide; elongation offset as a Param.
- [ ] Inverse dynamics and operational-space quantities (task-space inertia, operational-space
  force).

### Every model in CasADi, numpy and sympy (and C, JAX, PyTorch)

- [ ] One translator from a CasADi function's instruction list to: numpy source (runs without
  CasADi installed), sympy expressions (for papers, symbolic checks and LaTeX), JAX and PyTorch
  (differentiable, for learning). C comes from CasADi's own code generator (embedded boards,
  fastest loops).
- [ ] One way to ask for any of them, for example
  `vmc.Kinematics(robot).function("tip", backend="numpy")`, `compiled.export("sympy")`,
  `dynamics.export("torch")`; generated numpy code can be saved as a `.py` file.
- [ ] Tests: every backend agrees with CasADi to 1e-12 on random inputs, including Jacobians and
  Hessians.
- [ ] Docs, "Use a model outside CasADi".

### URDF, meshes, and everything VMRobotControl.jl offers

VMRobotControl.jl (the Julia library that shares this library's vocabulary of coordinates and
components) is the reference for completeness. Each of its features gets an equivalent here, and
the docs keep a table that maps one to the other.

- [ ] **URDF import:** joints, axes, origins, limits, masses, centres of mass, inertia tensors,
  visual and collision meshes (STL, OBJ, DAE). Every number becomes a Param whose default is the
  URDF value. Checked numerically against pinocchio in the tests.
- [ ] **Export** to URDF and MJCF (for MuJoCo, Gazebo, Isaac), including a rigid-link
  approximation of continuum segments.
- [ ] **Serialization** of whole systems (robot and controller) to YAML or JSON, round trip.
- [ ] **Visualization:** 3D drawings with meshes; sketches of joints, frames, coordinates and
  components (coils for springs, dashpots for dampers); frame labels; animations of runs to MP4 or
  GIF; an interactive 3D viewer in the browser that updates during a run; a distinct colour for
  the virtual mechanism.
- [ ] **Coordinates still missing:** sum; orientation error; angular velocity of a frame; a vector
  expressed in a frame, and back to the base frame.
- [ ] **Components still missing:** one-sided (rectified) springs and dampers; a diode damper
  (damps one direction of motion only); linear inerter; rigid inertia; force- and power-limited
  sources; a helper that adds deadzone springs to joint limits.
- [ ] **Virtual mechanisms with their own kinematics:** a controller whose model is driven by the
  virtual state (a virtual cart on a rail along a path, a virtual tool).
- [ ] **Examples** mirroring it: impedance control of a seven-joint arm loaded from its URDF;
  reaching with obstacle avoidance; compliant path following; a pendulum on a spline rail.

---

## 0.5.0: simulators

- [ ] Integrators for `ModelPlant`: RK4, the current linearly implicit step, CVODES and IDAS
  through CasADi; an `ode()` that returns f(t, x) for SciPy's `solve_ivp`.
- [ ] `MuJoCoPlant`, from a URDF or MJCF: contact with friction, meshes, offscreen rendering to
  video.
- [ ] `ElasticaPlant`: the soft arm in PyElastica, to validate PCC and the strain models.
- [ ] Other engines as plugins: PyBullet; Gazebo through ROS 2 (the ROS plant already speaks its
  topics); Isaac.
- [ ] Realism wrappers: delays, sensor rates, noise, quantization, spikes; actuator models
  (friction, efficiency, torque limits).
- [ ] Contact, continued: tangential friction (smooth Coulomb); contact forces reported per
  component by the simulator; more shapes (box, cylinder, capsule); contact between two robot
  points (self-contact, two arms); soft objects.
- [ ] Cross-fidelity tests: `ModelPlant` against MuJoCo on the UR5 carrying the hand; PCC against
  PyElastica.
- [ ] Closed-loop rollouts compiled with CasADi (`mapaccum`): fast and differentiable, for
  optimization and learning.

---

## 0.6.0: identification, calibration, estimation

- [ ] Stiffness and damping from step responses: linear least squares with K, D ≥ 0, smoothing,
  baseline subtraction, a friction column, validation by simulating held-out steps. Masses stay
  fixed (they cannot be identified from slow data).
- [ ] Generic linear-in-parameters regression from the derivative of the dynamics residual with
  respect to the parameters; nonlinear least squares.
- [ ] Calibration: transmission ratios, motor constants, base transforms between arms, Stribeck
  friction for the compensation stage (the efficiency fit is done, in 0.3.0).
- [ ] Measurement models shared by simulated sensors and estimators: encoders, motion-capture
  markers, IMU relative rotations, load cells.
- [ ] Shape from motion capture and from IMUs (kinematic inversion), with velocities computed at
  each sensor's real rate.
- [ ] Kalman filters (EKF and UKF) fusing encoders with motion capture or IMUs: per-sensor gating,
  health flags, staleness.
- [ ] Contact force from the virtual springs (virtual work) and from motor currents; task-space
  stiffness (congruence transformation, Hessian terms included); object compliance by probing;
  a momentum observer for external forces.

---

## 0.7.0: passivity, energy tanks, adaptation, underactuation

- [ ] Energy accounting from logs; passivity checks.
- [ ] Energy tank with exact bounds on parameter steps, including steps that release energy when
  the tank is empty.
- [ ] Adaptation laws: stiffness modulation, reference modulation, direct stiffness tracking,
  integral pose regulation, stiffness schedules K(F) and K(d), force tracking by reference or by
  stiffness gradient descent; every stiffness update symmetrized and projected onto positive
  semidefinite matrices.
- [ ] Every online update can pass through a passivity filter (tank or projection), opt-in.
- [ ] Underactuated VMC: actuation projector, torque defect, feasible force set; naive and frozen
  base controllers; passive and tank corrections; direction-constrained force tracking; each one a
  flag, so the old and new behaviour can be compared. Templates: a planar three-link arm with one
  passive joint, a five-link arm with three passive joints, a two-tendon continuum arm.
- [ ] Controllers for robots where some coordinates are not measured.
- [ ] Locomotion: phase-modulated stiffness, saturating potentials, steering, a series-VSA
  potential; extremum seeking of gaits and gains under a passivity cap (turtle).
- [ ] Co-design structures: series elastic transmission with a parallel elastic element on the
  motor; slow, non-backdrivable motors that set stiffness and rest angle; power reported for slow
  actuators.

---

## 0.8.0: optimization and MPC

- [ ] Problems built from scoped variables and blocks (no new language): trajectory blocks
  (trapezoidal and Hermite-Simpson collocation, multiple shooting, free final time, periodic),
  equilibrium, data fit (identification, moving-horizon estimation), energy tank; automatic
  variable scaling; solver presets that work on these problems (IPOPT with L-BFGS and
  `expand=True`, FATROP, SQP and QP solvers; acados optional).
- [ ] Port the offline optimization of virtual mechanisms on the soft arm: stiffness, reference
  and combined optimization with the arm's dynamics; task terms (reaching, obstacle avoidance,
  effort); grid search over references; planning of element swaps with the controller's own
  blend; the results applied to the controller and checked in simulation.
- [ ] MPC: initial state and references as parameters, shifted warm start, a horizon of stiffness
  and reference trajectories under passivity (tank) constraints; runs asynchronously and applies
  its result through `controller.set()` with the measured latency; a real-time-iteration option.
- [ ] Structure optimization: element gates g between 0 and 1 with a sparsity cost (which springs
  to keep and where to attach them).
- [ ] Co-design across time scales: design variables (geometry, VSA settings, hardware stiffness),
  episode variables and stage variables in one problem, through the Param scopes; scenario sets
  that share design variables; bilevel problems.
- [ ] Gradient-free and hardware-in-the-loop tuning with an ask-and-tell interface (grid, CMA-ES,
  Bayesian optimization, extremum seeking); episodes reset by homing on hardware.

---

## 0.9.0: learning

- [ ] PyTorch and JAX versions of every model and controller (from the 0.4.0 translator), and a
  CasADi and PyTorch autograd bridge for training through compiled functions.
- [ ] A gymnasium environment over any simulated plant and controller: actions are VMC parameters
  (stiffness, references, gates), observations configurable, rewards supplied by the user.
- [ ] Datasets from run logs, data-glove recordings and kinesthetic demonstrations.
- [ ] Networks that output VMC parameters, always passed through a passivity-consistent
  projection; small networks exported back to CasADi for the control loop.
- [ ] Learned residual dynamics as a custom-residual model.
- [ ] Examples: grasp stiffness learned from demonstrations; reinforcement learning of reference
  trajectories in simulation.

---

## 1.0.0

- [ ] API review and freeze; deprecated names removed; a support policy.

---

## Research the library must carry

The library exists to serve research on Virtual Model Control of compliant manipulators. Check
new designs against these themes, so nothing they need is made hard:

- **Distributed, interpretable compliance:** joint- and task-space springs on fingers and along
  continuum bodies, directional springs, force-dependent stiffness, stiffness distributions across
  the fingers of a hand, contact at many points of a chain. (Mostly done; contact friction in 0.5.0.)
- **Model-based sensing:** contact force from spring deflection, object compliance by probing,
  efficiency calibration. (0.6.0)
- **Force and stiffness regulation:** open and closed loop; gradient laws on references and
  stiffness; grasp-force tracking on two arms with an energy tank. (0.7.0)
- **Optimizing virtual mechanisms:** offline (collocation, reference search, element swaps),
  online (adaptation within an energy budget), MPC, and structure (which elements, where).
  (0.8.0)
- **Underactuated and under-sensed compliant robots.** (0.7.0)
- **Co-design across time scales with VSAs:** hardware rest state and stiffness, VSA settings and
  virtual stiffness in one problem; precision-to-power grasp transitions; impact absorption with
  a wrist VSA. (0.7.0 and 0.8.0)
- **Locomotion with VMC instead of central pattern generators:** virtual flywheel, ground contact,
  gait tuning (turtle). (0.3.0 to 0.7.0)
- **Learning VMC parameters from demonstrations, with passivity guarantees.** (0.9.0)
- **IMU proprioception of soft arms** as an estimator plugin. (0.6.0)

## Everything done by hand in lab code, and where it lands

| Done by hand today | In the library | Release |
| --- | --- | --- |
| PCC and serial kinematics, Jacobians, Hessians | models, `vmc.Kinematics` | done |
| virtual springs and dampers, summed into torques | mechanisms, `vmc.compile`, `VMCController` | done |
| gravity compensation | `GravityCompensation` | done |
| friction compensation, pretension, torque clip | opt-in output stages | done |
| plant nodes simulating the arm | `ModelPlant`, `vmc.sim.run` | done |
| homing scripts, bus checks | `hardware` homing and bus check | 0.3.0 |
| degree, sign and current conversions in every node | hardware profiles, joint I/O | 0.3.0 |
| controller nodes with GUIs, live gain changes | controller node, GUI, `controller.set` | 0.3.0 |
| swapping springs live, blended | element swaps with a quintic blend | 0.3.0 |
| digital twin with simulated sensors | `serve(ModelPlant)` and realism wrappers | 0.3.0, 0.5.0 |
| logging runs to npz and CSV | run logs | 0.3.0 |
| step experiments and K, D fits | identification | 0.6.0 |
| efficiency calibration against a load cell | `identification.fit_efficiency` | done |
| Kalman filter fusing encoders and motion capture | estimation | 0.6.0 |
| contact-force and task-stiffness estimates | estimation | 0.6.0 |
| tank-based grasp-force tracking | passivity and adaptation | 0.7.0 |
| stiffness and reference gradient descent | adaptation | 0.7.0 |
| offline optimization and reference search | optimization | 0.8.0 |

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
