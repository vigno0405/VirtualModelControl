---
file_format: mystnb
kernelspec:
  name: python3
---

# Add a model

Any robot can join the library if its model follows this contract, including external,
learned or black-box models.

## Kinematics

- Provide a `Space` (`nq`, `nv`; product spaces for floating bases) and a `ParamSet` holding
  every geometric number, each with a unit, a default and bounds.
- Provide `frame(q, at, p) -> (R, p)` written with CasADi operations. `at` is a named site, or
  an arc parameter `s` for continuous bodies, and may be symbolic.
- Keep the result smooth in `q`, `p` and `s`:
  - no numpy on symbols
  - no Python `if` on symbolic values; use `casadi.if_else` or a smooth blend
  - regularize singular points, as PCC does with `D = √(Dx² + Dy² + ε)`
- Use SI units and list the named sites (bodies, tips, markers, mounting points).
- If a closed form exists, test it against a generic integrator.

## Dynamics

Either way:

- **From components (preferred):** inertances (point masses, rigid inertias, mass distributed
  along `s`), storage (elastic potential, gravity), dissipation and sources. The canonical
  residual, the energy and the power are then assembled automatically.
- **Custom:** a residual `r(q, v, a, u, f_ext, p)` in the canonical form, plus `energy(q, v, p)`
  when one exists. The passivity tools need the energy and refuse to run without it.

## Actuation

Give `B(q)` through actuation components: motors, tendons (length coordinate, Jacobian by
automatic differentiation), transmissions, couplings. Signs, motor constants and efficiencies
are `Param`s or hardware-profile entries.

## Registration

Inside the library, decorate the model class with `register("model", name)`. A project
repository registers through an entry point in the group `virtualmodelcontrol.plugins`.

## Required tests

1. Jacobians (and Hessians, if used) from automatic differentiation match finite differences.
2. Rotations are orthonormal; frames are continuous across segment or link boundaries.
3. With no dissipation and no input, the simulation conserves energy.
4. At rest, the storage gradient equals `Bᵀu` plus the external forces.
5. No NaN or Inf at the defaults, at the parameter bounds, or at singular poses.
6. The model survives a round trip to a dict and back.
7. A cross-check against a reference implementation, when one exists.

## Example: a pendulum

A planar pendulum of length `L` swinging about the base's z axis. The model gives a space,
its Params and one site; everything else (components, compile, controller) works unchanged.

```{code-cell} python
import casadi as ca
import numpy as np
import virtualmodelcontrol as vmc


@vmc.register("model", "pendulum")
class Pendulum:
    """Pendulum about z; q = angle [rad]. Params: ``L`` length [m]."""

    q_unit = "rad"

    def __init__(self, length):
        self.space = vmc.Euclidean(1)
        self.params = vmc.ParamSet([vmc.Param("L", length, unit="m", scope="design")])
        self.sites = ("bob",)

    def frame(self, q, at, p):
        c, s = ca.cos(q[0]), ca.sin(q[0])
        R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
        return R, p["L"] * ca.vertcat(c, s, 0)


robot = vmc.Mechanism("pendulum", model=Pendulum(0.5))
ctrl = vmc.Mechanism("ctrl")
ctrl.add("hold", vmc.LinearSpring(robot.point("bob") - [0.0, 0.5, 0.0], 10.0))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
meas = vmc.Signals(0.0, motor_position=[0.0], motor_velocity=[0.0])
controller.step(0.0, meas)["motor_torque"]   # pulls the bob from angle 0 towards 90°
```

The torque is positive: the spring turns the pendulum towards the goal at 90°.
