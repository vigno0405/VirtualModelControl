---
file_format: mystnb
kernelspec:
  name: python3
---

# Energy and passivity

In this tutorial we follow the energy of the soft arm and of its controller through a run, and
check that the balance the library computes closes.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## Where energy goes

```{code-cell} python
:tags: [remove-input]
from schematics import energy_flow
energy_flow.figure();
```

Each mechanism stores energy in its springs and gravity ($V$) and in its moving masses ($T$),
loses it in its dampers and receives it from its sources. The controller and the robot exchange
energy only through the motors: the power $\tau^\top v$ of the controller's torques leaves the
controller as its port and enters the robot as its input. The arrows point the way energy flows
when a term is positive; dissipation is never positive.

## The balance in code

The compiled controller computes its terms of the figure, and so do the robot's dynamics, which
`ModelPlant` compiles from the robot mechanism and keeps as `plant.dynamics`:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `law.energy` | `q, v, z, p, t` | `V, T` |
| `law.power` | `q, v, z, p, t` | `port, dissipation, source` |
| `plant.dynamics.energy` | `q, v, p, t` | `T, V` |
| `plant.dynamics.power` | `q, v, u, p, t` | `input, dissipation, source` |

The energies come in opposite orders: `(V, T)` for the controller, `(T, V)` for the robot.
`z` holds the controller's virtual states (positions, then velocities), `p` the live Params of
each and `u` the motor torques. The input is the port: the robot receives the torques as the
controller sends them, as long as no output stage changes them and the transmission's
efficiency is 1, the default.

## A run

We take the controller of [your first controller](first-controller.md) and run the loop faster
than on the real arm, so that the logged steps resolve the start, where most of the energy
changes hands.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-145-145"))
goal = np.array([0.08, 0.0, 0.40])  # [m]
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 300.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
controller = vmc.VMCController(law)

plant = vmc.sim.ModelPlant(arm)
clock = vmc.sim.SimClock(dt=1 / 5000)  # [s]
rows = vmc.sim.run(plant, controller, clock, T=0.2).arrays()
```

```{code-cell} python
:tags: [remove-output]
from virtualmodelcontrol import viz

viz.animate(arm, rows, "energy.mp4", springs=[(1.0, goal)], trace=1.0,
            speed=0.1)
```

```{video} energy.mp4
:caption: The run whose energies follow, ten times slower than real time.
```

We evaluate the four functions at every logged step: `map(n)` evaluates a CasADi function on
`n` columns at once.

```{code-cell} python
t = rows["t"].ravel()
q, v = rows["q"].T, rows["v"].T  # a column per step
u = rows["motor_torque"].T
z = np.zeros((0, t.size))  # this controller has no virtual states
p_c, p_r = controller.params, plant.p  # live Params of each


def along(f, *args):
    """f at every step, one array per output."""
    return [x.full().ravel() for x in f.map(t.size)(*args)]


V_c, T_c = along(law.energy, q, v, z, p_c, t)
port, diss_c, src_c = along(law.power, q, v, z, p_c, t)
T_r, V_r = along(plant.dynamics.energy, q, v, p_r, t)
inflow, diss_r, src_r = along(plant.dynamics.power, q, v, u, p_r, t)
np.abs(inflow - port).max()  # [W]
```

At every step the robot receives exactly the power the controller gives. The energy of the
robot plus the energy of the controller then changes by what the dampers take and the sources
give, integrated over time; whatever is left over is the residual:

```{code-cell} python
def integral(power):  # from the start to every step [J]
    steps = (power[1:] + power[:-1]) / 2 * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(steps)])


E_c, E_r = V_c + T_c, T_r + V_r
dampers = integral(diss_c + diss_r)
sources = integral(src_c + src_r)
residual = (E_c - E_c[0]) + (E_r - E_r[0]) - dampers - sources

ms = 1000 * t
fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
top.plot(ms, E_c - E_c[0], label="controller")
top.plot(ms, E_r - E_r[0], label="robot")
top.plot(ms, dampers, label="dampers")
top.plot(ms, sources, label="sources")
top.set_ylabel("energy [J]")
top.legend(loc="center right", fontsize=18)
bottom.plot(ms, 1000 * residual)
bottom.set_xlabel("time [ms]")
bottom.set_ylabel("residual [mJ]");
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

released = (E_c[0] - E_c) / (E_c[0] - E_c[-1])
glue("stored", float(E_c[0]), display=False)
glue("t90", float(ms[np.argmax(released >= 0.9)]), display=False)
glue("kinetic", float(T_r.max()), display=False)
glue("res", 1000 * float(np.abs(residual).max()), display=False)
glue("pct", 100 * float(np.abs(residual).max() / -dampers[-1]), display=False)
assert port.min() > -1e-9, "the arm pushes back: rewrite the text"
glue("work", float(integral(port)[-1]), display=False)
assert abs(integral(src_c)[-1]) < 1e-6, "gravity compensation does work: rewrite"
```

The top panel shows how the energy of each mechanism changed since the start, the energy the
dampers took (negative) and the energy the sources gave. The controller's spring starts with
{glue:text}`stored:.2f` J, and nine tenths of what it releases in this run are gone after
{glue:text}`t90:.0f` ms: some goes into the arm,
whose kinetic energy peaks at {glue:text}`kinetic:.2f` J, and most into the dampers of the arm
and of the controller.

The residual stays within {glue:text}`res:.1f` mJ, {glue:text}`pct:.1f` % of the energy the
dampers took. It is the error of the time steps: the simulator advances in discrete steps and
holds each torque for a control period, and the powers are integrated from the logged steps.
In this run power flows from the controller to the arm at every step; when an arm swings back
against its controller, the port carries energy the other way, and the balance closes all the
same.

## Passivity

From its balance, a controller without sources does on the robot the work

$$
\begin{aligned}
\int_0^t \text{port}\,\mathrm{d}t
&= E(0) - E(t) \\
&\quad + \int_0^t \text{dissipation}\,\mathrm{d}t \\
&\le E(0),
\end{aligned}
$$

since dissipation is never positive and springs and masses never store negative energy.
Whatever the robot does, such a controller can only give back the energy it stored: it is
passive. On a robot that only has masses, springs, gravity and dampers of its own, the energy
of the robot plus the energy of the controller can then only fall.

Gravity compensation is a source, but a tame one: its forces are those that cancel the weight
of the robot's masses, so the energy it supplies is the energy that the robot stores in
gravity, and the arm moves as if it had no weight. This arm moves in a horizontal plane, so
here it supplies nothing. The controller gave {glue:text}`work:.2f` J through its port, of the
{glue:text}`stored:.2f` J its spring stored at the start.

Changing a live parameter while running changes the controller's energy too, by the jump that
`controller.set` returns ([Parameters](parameters.md)).
