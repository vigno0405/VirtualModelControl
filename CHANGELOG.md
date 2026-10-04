# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions, with one minor version of deprecation before a removal).

## [Unreleased]

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
  section in the left sidebar) and large type;
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

[Unreleased]: https://github.com/vigno0405/VirtualModelControl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vigno0405/VirtualModelControl/releases/tag/v0.1.0
