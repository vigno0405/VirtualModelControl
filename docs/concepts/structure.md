# How the library is organized

Everything is reachable from the top-level package, `import virtualmodelcontrol as vmc`. The
subpackages group the pieces by role:

| Package | What it contains | You use it to |
|---|---|---|
| `vmc.robots` | ready-made robots: `helyx`, `bimanual`, `adapt` (finger, hand), `turtle`, `ur5` | start from a robot that already exists |
| `vmc.models` | kinematic models: `PCC` (continuum), `SerialChain` (rigid chains, also from DH tables), `LinearCoupling`, `Assembly`, `JointSpace`; transmissions `TendonTransmission`, `Direct`; `Kinematics` | describe a new robot; get positions, Jacobians, Hessians |
| `vmc.mechanisms` | `Mechanism`, coordinates (`FramePoint`, `Joint`, `Ref`, `Projection`, ...), components (springs, dampers, masses, sources) | build controllers and the robot's physics |
| `vmc.compile`, `vmc.VirtualMechanismSystem` | the compiler | turn a robot and a controller into fast functions |
| `vmc.control` | `VMCController`, output stages (friction compensation, pretension, torque limits) | run the controller step by step |
| `vmc.dynamics` | `compile_dynamics` | get the robot's equations of motion |
| `vmc.sim` | `ModelPlant`, `run`, `SimClock`, `Guard`, `RunLog` | simulate a closed loop |
| `vmc.viz` | lab-style figures: robots, springs, goals, forces; PDF + SVG | draw what happens |
| `vmc.core` | `Param`, `ParamSet`, spaces, `Signals`, the plugin registry | low-level building blocks |

## Layers

The packages are layered: a module only uses the ones below it, which keeps the core free of
hardware, plotting or ROS code (checked automatically in CI).

```{mermaid}
flowchart TB
    R["robots, viz (edges)"] --> S
    S["sim: plants, run loop"] --> C
    C["control, compile, dynamics"] --> M
    M["models: kinematics, transmissions"] --> K
    K["mechanisms: coordinates, components"] --> P
    P["core: parameters, spaces, signals, registry"]
```

## Where things live, by question

| Question | Answer |
|---|---|
| Where is my robot's geometry? | in its template (`vmc.robots.<name>`), as `Param` defaults you can change |
| How do I get a Jacobian? | `vmc.Kinematics(robot).jacobian(q, "tip")` |
| How do I add a spring? | `ctrl.add("name", vmc.LinearSpring(coordinate, stiffness))` |
| How do I change a gain while running? | `controller.set({"ctrl.name.stiffness": 50.0})` |
| How do I simulate? | `vmc.sim.run(vmc.sim.ModelPlant(robot), controller, vmc.sim.SimClock(dt), T)` |
| How do I add my own element or robot? | [Extend the library](../how-to/extend.md), [Build a robot](../how-to/build-a-robot.md) |
