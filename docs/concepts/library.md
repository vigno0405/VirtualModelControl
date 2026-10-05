---
file_format: mystnb
kernelspec:
  name: python3
---

# How the library is organized

The library is a stack of layers, each of which uses only the layers below it. The ready-made
robots and the figures sit at the edges: they may use every layer, and no layer uses them.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
:tags: [remove-input]
from schematics import library_map
library_map.figure();
```

A controller needs only the layers up to `control`: describe the robot with `models`, place
virtual elements with `mechanisms`, pair them in a `VirtualMechanismSystem`, `compile` it and
run it with `VMCController`. Simulation, figures and the robot templates are optional. The
order is enforced: `lint-imports` checks the contracts written in `pyproject.toml` on every
change, including that nothing in the library imports ROS.

## Where things are

| You want to | Use |
|---|---|
| start from an existing robot | `virtualmodelcontrol.robots`: `helyx`, `bimanual`, `adapt`, `turtle`, `ur5` |
| describe a new robot | `vmc.models`: `PCC`, `SerialChain`, `JointSpace`, `LinearCoupling`, `Assembly`, `TendonTransmission` |
| place virtual elements | `vmc.Mechanism`, coordinates and components such as `vmc.LinearSpring` |
| compile and run a controller | `vmc.VirtualMechanismSystem`, `vmc.compile`, `vmc.VMCController` |
| correct commands for real hardware | `vmc.control.output`: friction compensation, pretension, limits |
| get positions, Jacobians, Hessians | `vmc.Kinematics` |
| get the robot's equations of motion | `vmc.compile_dynamics` |
| simulate a closed loop | `vmc.sim`: `ModelPlant`, `run`, `SimClock` |
| fit stiffness, damping and efficiencies to data | `vmc.identification` |
| plan a motion and optimize a controller's Params | `vmc.optimization`: `Problem`, `Collocation`, `Effort`, `Cost`, `Bound` |
| record, save, replay and compare runs | `vmc.sim`: `RunLog`, `replay`, `compare` |
| change a running controller by hand | `vmc.interactive`: `Session`, `Window`, `Interactive`, `Recorder` |
| describe an experiment in a file | `vmc.config`: `load`, `Experiment` |
| draw and animate | `vmc.viz` |
| parameters, units, scopes | `vmc.Param`, `vmc.ParamSet` |

## One symbolic source

Every model is written once with CasADi operations. Jacobians, Hessians, the control law and the
robot's dynamics all come from that one graph by automatic differentiation, so they can never
disagree with each other. The robot's dynamics take one form for every robot, the residual

$$
\begin{aligned}
&M(q)\,\dot v + h(q, v) \\
&\quad - B(q)\,u - \textstyle\sum_k J_k^\top f_k = 0,
\end{aligned}
$$

with $M$ the mass matrix, $h$ the velocity and gravity terms, $B$ the actuation map, $f_k$ the
forces of the robot's own components and $J_k$ the Jacobian of each one's coordinate with
respect to the velocity.
