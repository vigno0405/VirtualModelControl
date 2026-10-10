---
file_format: mystnb
kernelspec:
  name: python3
---

# Joints and bodies

In this tutorial we use the joints of a `SerialChain` that have more than one coordinate: we toss
a free brick, spin a floating one, give a body its inertia, and slide a bead along a rail.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import SerialChain
```

## Joints with more coordinates

A joint of a `SerialChain` is one of seven kinds. Its coordinates take the next entries of $q$:

| joint | coordinates | motion |
|---|---|---|
| `"revolute"` | 1 | a turn about its axis |
| `"prismatic"` | 1 | a slide along its axis |
| `("helical", pitch)` | 1 | a turn that also slides by `pitch` [m/rad] |
| `"spherical"` | 3 | a rotation vector, about the joint's point |
| `"free"` | 6 | a translation, then a rotation vector: a floating base |
| `"floating"` | 7 | a translation, then a unit quaternion: a floating base that turns any number of times |
| `("rail", waypoints)` | 1 | a slide along the spline through the waypoints |

A spherical, a free, a floating and a rail joint have no axis: give `None`. We toss a brick on a free joint. It
is four point masses at its corners, and it spins as it flies:

```{code-cell} python
corners = {"a": [0.2, 0.0, 0.0], "b": [0.0, 0.1, 0.0],
           "c": [0.0, 0.0, 0.3], "d": [-0.1, -0.1, -0.1]}
mass = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 1.5}  # [kg]
brick = vmc.Mechanism("brick", model=SerialChain(
    ["free"], axes=[None], points=[[0, 0, 0]],
    sites={name: (1, c) for name, c in corners.items()}))
brick.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
for name, m in mass.items():
    brick.add(f"m_{name}", vmc.PointMass(brick.point(name), m))
brick.add("gravity", vmc.Gravity(brick))

v0 = [0.5, 0.0, 2.0, 3.0, -2.0, 1.0]  # the translation rate, the spin
plant = vmc.sim.ModelPlant(brick, v0=v0, max_step=1e-4)
kin = vmc.Kinematics(brick)
times, path, tip = [], [], []
for _ in range(60):
    plant.advance(0.01)
    times.append(plant.t)
    path.append(sum(m * kin.position(plant.q, n)
                    for n, m in mass.items()) / sum(mass.values()))
    tip.append(kin.position(plant.q, "c"))
path, tip, times = np.array(path), np.array(tip), np.array(times)
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

com0 = sum(m * np.array(corners[n]) for n, m in mass.items()) / sum(mass.values())
start = np.array(v0[:3]) + np.cross(v0[3:], com0)  # the center's speed
fall = com0 + np.outer(times, start) + 0.5 * np.outer(times**2, [0, 0, -9.81])
error = np.abs(path - fall).max()
assert error < 1e-3 and np.linalg.norm(plant.q[3:]) > 1.0
glue("error", float(1e3 * error), display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(path[:, 0], path[:, 2], label="center of mass")
ax.plot(tip[:, 0], tip[:, 2], label="corner c")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.legend();
```

The center of mass follows the parabola of a stone to {glue:text}`error:.2f` mm while the corner
winds around it: nothing but gravity acts, so the spin does not change it. The orientation of a
free joint is a rotation vector, which is smooth until the body has turned once. A body that
keeps spinning takes a `"floating"` joint: its orientation is a unit quaternion, so $q$ has
seven entries and the velocity six, the translation's and the body's own angular velocity.
The same brick, thrown with a spin of 15 rad/s, turns more than twice in a second:

```{code-cell} python
spinner = vmc.Mechanism("spinner", model=SerialChain(
    ["floating"], axes=[None], points=[[0, 0, 0]],
    sites={name: (1, c) for name, c in corners.items()}))
for name, m in mass.items():
    spinner.add(f"m_{name}", vmc.PointMass(spinner.point(name), m))

plant = vmc.sim.ModelPlant(spinner, v0=[0, 0, 0, 0, 15.0, 0], max_step=1e-4)
rate = []
for _ in range(50):
    plant.advance(0.02)
    rate.append(np.linalg.norm(plant.v[3:]))
plant.q.size, plant.v.size
```

```{code-cell} python
:tags: [remove-cell]
turns = np.sum(rate) * 0.02 / (2 * np.pi)
assert turns > 2 and abs(np.linalg.norm(plant.q[3:]) - 1) < 1e-12
glue("turns", float(turns), display=False)
```

It turned {glue:text}`turns:.1f` times, and its quaternion is still a unit.

A body does not need its point masses. A `PointMass` at its center gives its mass, and a
`RotationalInertia` on the `FrameRotation` of its frame gives the inertia about that point:
a 3 by 3 matrix in the frame's axes, or its three principal moments. We replace the brick's four
corners by their total mass and the inertia about their center, and give both the same spin:

```{code-cell} python
names = list(corners)
m = np.array([mass[n] for n in names])
r = np.array([corners[n] for n in names])
centre = m @ r / m.sum()  # [m]
inertia = sum(mk * (x @ x * np.eye(3) - np.outer(x, x))
              for mk, x in zip(m, r - centre))  # [kg·m²], about the center

def rigid_brick():
    """A floating brick that turns about its center of mass."""
    sites = {n: (1, c) for n, c in corners.items()}
    sites["center"] = (1, centre)
    chain = SerialChain(["floating"], axes=[None], points=[centre],
                        sites=sites)
    return vmc.Mechanism("brick", model=chain)

corner_masses, rigid = rigid_brick(), rigid_brick()
for n in names:
    point = corner_masses.point(n)
    corner_masses.add(f"m_{n}", vmc.PointMass(point, mass[n]))
rigid.add("mass", vmc.PointMass(rigid.point("center"), m.sum()))
rigid.add("spin", vmc.RotationalInertia(
    vmc.FrameRotation(rigid.model, "center"), inertia))

spins = {}
for name, robot in (("four corners", corner_masses), ("one body", rigid)):
    plant = vmc.sim.ModelPlant(robot, v0=[0, 0, 0, 3.0, 12.0, -2.0],
                               max_step=1e-4)
    spin = []
    for _ in range(40):
        plant.advance(0.025)
        spin.append(plant.v[3:].copy())
    spins[name] = np.array(spin)
```

```{code-cell} python
:tags: [remove-cell]
error = np.abs(spins["four corners"] - spins["one body"]).max()
assert error < 1e-6 and np.ptp(spins["one body"][:, 0]) > 1.0
glue("rigid_error", float(error), display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
time = 0.025 * np.arange(1, 41)
for i, axis_name in enumerate("xyz"):
    line, = ax.plot(time, spins["four corners"][:, i], label=f"$\\omega_{axis_name}$")
    ax.plot(time[::2], spins["one body"][::2, i], "o", color=line.get_color())
ax.set_xlabel("time [s]")
ax.set_ylabel("angular velocity [rad/s]")
ax.legend();
```

The brick of four corners (lines) and the one body (dots) turn alike, to
{glue:text}`rigid_error:.0e` rad/s: they have the same mass and the same inertia.

## A rail along a path

A rail carries a body along a curve. Its coordinate $s$ is the parameter of the natural cubic
spline through the waypoints: 0 at the first waypoint and 1 at the last, with the same step
of $s$ between neighbors, so give waypoints about equally far apart. The waypoints are
`design` Params in the frame of the joint before the rail; the body starts where its site is
at $q = 0$, so put the site at the first waypoint to draw the path as given. A bead on a
circular wire swings as a pendulum. The wire runs well past the swing, because a spline
ends flat:

```{code-cell} python
radius = 0.5  # [m]
phi = np.radians(np.linspace(-100, 100, 17))
wire = np.column_stack([radius * np.sin(phi), 0 * phi,
                        radius * (1 - np.cos(phi))])
bead = vmc.Mechanism("bead", model=SerialChain(
    [("rail", wire)], axes=[None], points=[[0, 0, 0]],
    sites={"bead": (1, wire[0])}))  # at the first waypoint when s = 0
bead.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
bead.add("mass", vmc.PointMass(bead.point("bead"), 0.5))
bead.add("gravity", vmc.Gravity(bead))

start = 0.8  # [rad] from the bottom of the wire
plant = vmc.sim.ModelPlant(
    bead, q0=[(start + phi[-1]) / (2 * phi[-1])], max_step=1e-4)
kin = vmc.Kinematics(bead)
times, angle = np.linspace(0, 2, 81), []
for t in times:
    plant.advance(t - plant.t)
    x, _, z = kin.position(plant.q, "bead")
    angle.append(np.arctan2(x, radius - z))
```

Its angle should be the pendulum's, $\ddot\varphi = -(g/R) \sin\varphi$, which we integrate on
the side:

```{code-cell} python
from scipy.integrate import solve_ivp

exact = solve_ivp(lambda t, y: [y[1], -9.81 / radius * np.sin(y[0])],
                  (0, 2), [start, 0.0], t_eval=times, rtol=1e-10)
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

error = np.abs(angle - exact.y[0]).max()
assert error < 2e-3 and np.ptp(exact.y[0]) > 1.0
glue("rail_error", float(1e3 * error), display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.plot(times, np.degrees(exact.y[0]), label="pendulum")
ax.plot(times[::3], np.degrees(angle)[::3], "o", label="bead on the rail")
ax.set_xlabel("time [s]")
ax.set_ylabel("angle from the bottom [deg]")
ax.legend();
```

The bead follows the pendulum to {glue:text}`rail_error:.1f` mrad over two seconds. A joint
placed after the rail rides on it: a revolute joint gives a pendulum hung from a cart.
