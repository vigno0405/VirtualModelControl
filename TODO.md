# TODO

Everything planned for `virtualmodelcontrol`, release by release. Ticked items are done. Each
item that ports code already used in the lab comes with a regression test against the original
results, documentation, and a line in `CHANGELOG.md`.

## 0.1.0: the core (released 2026-10-04)

- [x] Parameters with units, bounds and scopes; spaces (Euclidean, SO2, products); registry and
  plugins through entry points
- [x] Mechanisms: coordinates (points, joints, references, differences, projections, norms,
  custom) and components (linear, tanh, Gaussian, sigmoid, polynomial and limit springs; linear
  and tanh dampers; point masses and inertances; force sources; gravity and gravity compensation)
- [x] Models: piecewise constant curvature, tendon transmission, serial chains (product of
  exponentials), linear couplings, assemblies
- [x] Compiler and controller: about 20 µs per control step on a three-segment arm; live gains
  with the exact energy jump
- [x] Robot dynamics from components; model-based simulator; run loop with a guard and a recorder
- [x] Robots: Helyx soft arm (three geometries), ADAPT finger
- [x] Documentation online, CI, release to PyPI

## 0.2.0: every robot, and documentation for everyone

- [ ] Install every dependency with the package (numpy, SciPy, CasADi, matplotlib, PyYAML,
  Dynamixel SDK, pyserial); extras only for heavy optional tools
- [ ] CI on Ubuntu 22.04 and 24.04, including a clean install with the system Python
- [ ] Robot templates
  - [x] Helyx soft arm, three geometries
  - [ ] Bimanual Helyx: two arms on one frame, with each arm's calibration
  - [x] ADAPT finger
  - [ ] ADAPT hand: five digits, spread coupling, wrist
  - [ ] Turtle: two cranks driven through a virtual flywheel, and the VSA angle
  - [ ] UR5, and the hand mounted on its tool flange
- [ ] Hardware profiles: motor IDs and order, encoder signs, motor constants, baud rate, control
  rate, torque limits
- [ ] Mount a part on another part's frame; orientations of serial-chain sites
- [ ] Velocity regulator: drives a virtual state at a commanded speed, with a ramp
- [ ] Figures in the lab style: robots, springs and goals, saved as PDF and SVG
- [ ] Documentation: how the library is organized, how to build a robot, a page per robot, guides
  (soft arm, two arms, finger, hand, turtle), components and coordinates at a glance, glossary,
  troubleshooting
- [ ] A complete README

## 0.3.0: identification, logs and estimation

- [ ] Run logs saved and loaded atomically; configurations in YAML through the registry
- [ ] Stiffness and damping identification from step responses: linear least squares with
  K, D ≥ 0, smoothing, baseline subtraction, a friction term; validation on held-out steps
- [ ] Calibration of actuator efficiency and transmission ratios
- [ ] Measurement models: encoders, motion-capture markers, IMU relative rotations, load cells
- [ ] Shape from motion capture and from IMUs; velocities at the sensors' real rates
- [ ] Kalman filter fusing encoders and motion capture, with per-sensor gating and health flags
- [ ] Contact force from the virtual springs (virtual work), and task-space stiffness
- [ ] Object compliance by probing

## 0.4.0: passivity and adaptation

- [ ] Energy accounting from logs; passivity checks
- [ ] Energy tank with exact bounds on parameter steps, including steps that release energy when
  the tank is empty
- [ ] Adaptation: stiffness and reference modulation, direct stiffness tracking, integral pose
  regulation, stiffness schedules K(F) and K(d), symmetric positive semidefinite projection of
  stiffness updates
- [ ] Smooth replacement of one set of elements by another (quintic blend), in the controller and
  in the planner
- [ ] Output stage, each part opt-in: pretension, efficiency correction, torque clip, friction
  feed-forward
- [ ] Extremum seeking for gait and gain tuning

## 0.5.0: optimization

- [ ] Problems built from scoped variables and blocks: trapezoidal and Hermite–Simpson collocation,
  multiple shooting, variable scaling, solver settings that work for these problems
- [ ] Task terms: reaching, obstacle avoidance, effort
- [ ] Search over reference points; planning of element swaps with the controller's blend
- [ ] Model predictive control with a shifted warm start
- [ ] Gradient-free and hardware-in-the-loop tuning (ask and tell)

## 0.6.0: hardware and ROS 2

- [ ] Dynamixel plant through the SDK: units from the hardware profile, homing, command watchdog,
  torque clamp before the integer conversion, zero torque on exit
- [ ] Serial sensors: load cell, IMU
- [ ] ROS 2 with any distribution (rclpy only): joint I/O, a plant over ROS topics, a simulated
  robot served as a digital twin, a controller node; tested on Humble and Jazzy
- [ ] Wall-clock run loop with a staleness guard

## Later

- [ ] Curvature models beyond PCC (affine and polynomial curvature, Cosserat strains)
- [ ] URDF import; floating bases (SO3, SE3); MuJoCo and PyElastica plants
- [ ] Co-design across time scales; learning with PyTorch; a gym environment
- [ ] API freeze (1.0)
