---
file_format: mystnb
kernelspec:
  name: python3
---

# Crawl with a flywheel

In this tutorial a crawler crawls across the ground in simulation. Two cranks push it forward,
one per side, and one virtual flywheel keeps them in step.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## One flywheel, two cranks

A rhythmic gait needs its cranks to turn together, a fixed phase apart. The usual controller
gives each crank a clock and a reference to follow. A moving reference puts energy into the
loop that nothing accounts for, and two clocks drift apart. Here the clock is a physical
object: one virtual flywheel, with an inertia and a constant drive, tied to each crank by a
spring. The phase offset between the sides is a constant, so the gait phase is exact. The
flywheel stores what the springs give it, so the loop is passive. If a foot catches, its spring
stretches and slows the flywheel, and the other crank slows with it.

The flywheel has an angle $\varphi$. Side $i$ runs at the phase $\varphi_i = \varphi - \Delta_i$,
and its spring is stretched by $e_i = q_i - \varphi_i$, with $q_i$ the crank angle. The
stiffness follows the phase, and the torque on the crank is

$$
K_i = \bar K\,(1 \pm u_s)\,\big[1 + m\cos(\varphi_i - \varphi_K)\big],
\qquad u_i = -K_i e_i - C\dot e_i .
$$

The depth $m$ and the peak $\varphi_K$ shape the stiffness, and the steering $u_s$ makes one
side stiffer and the other softer. The flywheel, with drive $b_v\bar\omega$ and friction $b_v$,
obeys

$$
J_v\ddot\varphi = \sum_i \Big(K_i e_i - \tfrac12 K_i' e_i^2 + C\dot e_i\Big)
- b_v\dot\varphi + b_v\bar\omega .
$$

The term $-\tfrac12 K_i' e_i^2$ is the reaction of a stiffness that changes with the phase:
stiffening a stretched spring costs energy, and the flywheel pays it. `PhaseSpring` is the
spring whose energy depends on both the stretch and the phase, so its force on the phase is
exactly this term.

## The controller

The crank angles are the robot of the controller: `turtle.robot()` holds only $q$ = (left,
right) [rad]. The controller adds the flywheel as a virtual state with an inertia, a speed
regulator for the drive, and for each side a `PhaseSpring` on the pair (stretch, phase) and a
damper on the stretch. The right side runs $\Delta$ behind the left one. With $\Delta = \pi$
the cranks are half a turn apart, a diagonal gait.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import turtle

robot = turtle.robot()  # the two cranks

def flywheel(omega, depth=0.0, steer=0.0, peak=0.0, delta=np.pi):
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, 0.24))  # J_v [kg·m²]
    ctrl.add("drive", vmc.SpeedRegulator(phi, 0.3, omega, 0.5))
    behind = vmc.Ref("delta", 1, value=delta, unit="rad")
    for i, (phase, side) in enumerate([(phi, 1.0), (phi - behind, -1.0)]):
        e = robot.joint(i) - phase  # the stretch of the spring
        spring = vmc.PhaseSpring(vmc.Stack(e, phase), 1.0, depth=depth,
                                 peak=peak, steer=steer, side=side)
        ctrl.add(f"spring{i}", spring)  # K [N·m/rad]
        ctrl.add(f"damper{i}", vmc.LinearDamper(e, 0.06))  # C
    return ctrl
```

`omega` is the speed $\bar\omega$ [rad/s] the flywheel is driven to, reached over a ramp of
0.5 s. `side` is $\pm1$ in $(1 \pm u_s)$: the left spring is stiffer than the right one for a
positive `steer`. The controller does not know the body: it turns the two crank angles it
measures into two torques. We call it once at rest, with no simulation:

```{code-cell} python
system = vmc.VirtualMechanismSystem(robot, flywheel(omega=6.0, depth=0.5))
controller = vmc.VMCController(vmc.compile(system))
meas = vmc.Signals(0.0, motor_position=[0.0, -np.pi],
                   motor_velocity=[0.0, 0.0])
controller.reset(0.0, meas, z0=turtle.initial_state(meas))
controller.step(0.01, meas)["motor_torque"]
```

The cranks sit where the springs want them, so the torques are zero. Nothing else is needed on
the real robot: `controller.step(t, measurements)` returns the motor torques.

## The crawler

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("radius", 100 * turtle.CRAWLER["crank_radius"], display=False)
```

To see the controller work we need a body. `turtle.crawler()` is a simple crawler for the
simulator: a floating body that lies on the ground $z = 0$ with gravity, and two cranks that
turn about its lateral axis, each with a foot {glue:text}`radius:.0f` cm from the axis. At each foot and
at the four corners of the underside, a contact spring, a contact damper and friction hold the
body up. The cranks are the only motors; the body has none. It is a stand-in with placeholder
constants (`turtle.CRAWLER`), not the lab's turtle, and its numbers are not measured.

The simulator runs the crawler, and the controller still sees only the two cranks. This is the
split of the table in the [contact tutorial](contact.md): the world belongs to the simulated
robot, not to the controller. The crawler's $q$ holds the body's position and quaternion
$(x, y, z, w, x, y, z)$, then the two cranks. We start the body 4 cm above the ground, level and
facing $+x$, with the left foot down and the right one up:

```{code-cell} python
def crawl(omega=6.0, depth=0.0, steer=0.0, peak=0.0, T=8.0, **body):
    ctrl = flywheel(omega, depth, steer, peak)
    system = vmc.VirtualMechanismSystem(robot, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    q0 = [0, 0, 0.04, 1, 0, 0, 0, 0.0, -np.pi]  # x y z, quaternion, cranks
    plant = vmc.sim.ModelPlant(turtle.crawler(**body), q0=q0, max_step=1e-3)
    clock = vmc.sim.SimClock(1 / turtle.CONTROL_RATE)
    return vmc.sim.run(plant, controller, clock, T=T,
                       z0=turtle.initial_state, record=["energy"])

log = crawl()
rows = log.arrays()
t, q = rows["t"].ravel(), rows["q"]
```

`max_step=1e-3` is a step small enough to follow the contacts. We read the position of the
body and the crank angles from the log.

```{code-cell} python
:tags: [remove-cell]
q_end = q[-1]
assert q_end[0] > 0.5 and abs(q_end[1]) < 0.05 and 0.029 < q[:, 2].min()
glue("distance", float(q_end[0]), display=False)
```

The body moved {glue:text}`distance:.2f` m forward in eight seconds. The top panel shows its
distance and the bottom one the crank angles. The cranks run half a turn apart from the start,
and the left one follows the flywheel. Each foot is on the ground for a short part of its turn,
and the body advances while it is: its distance climbs in steps.

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.0))
top.plot(t, 100 * q[:, 0])
top.set_ylabel("distance [cm]")
bottom.plot(t, rows["z"][:, 0], "--", color=viz.PALETTE[9], label="flywheel")
for k, name in enumerate(turtle.CRANKS):
    bottom.plot(t, q[:, 7 + k], color=viz.PALETTE[k], label=name)
bottom.set_xlabel("time [s]")
bottom.set_ylabel("angle [rad]")
bottom.legend();
```

## The flywheel under load

The flywheel is driven to $\bar\omega$ = 6 rad/s, and the cranks load it, so it runs slower.
Over whole turns, $b_v(\bar\omega - \langle\dot\varphi\rangle)$ is the mean torque the
cranks take from the springs. The slowdown measures the load, and a constant drive needs no
sensor for it.

```{code-cell} python
omega, bv = 6.0, 0.3  # the commanded speed, and b_v
phi, torque = rows["z"][:, 0], rows["motor_torque"]
turns = np.floor(phi / (2 * np.pi)).astype(int)
first, last = (int(np.argmax(turns >= n)) for n in (2, turns.max()))
speed = (phi[last] - phi[first]) / (t[last] - t[first])  # [rad/s]
load = torque[first:last].sum(axis=1).mean()  # [N·m]
print(f"speed {speed:.2f} rad/s, b_v (omega - speed) = "
      f"{bv * (omega - speed):.3f} N·m, mean torque {load:.3f} N·m")
```

```{code-cell} python
:tags: [remove-cell]
assert abs(bv * (omega - speed) - load) < 0.03 * load
assert 0.8 * omega < speed < 0.95 * omega
glue("droop", float(100 * (omega - speed) / omega), display=False)
glue("agree", float(100 * abs(bv * (omega - speed) / load - 1)), display=False)
glue("speed", float(speed), display=False)
```

The flywheel runs {glue:text}`droop:.0f`% below its commanded speed, at
{glue:text}`speed:.2f` rad/s. The two torques printed above agree to
{glue:text}`agree:.0f`%.

## The shape of the stiffness

```{code-cell} python
:tags: [remove-cell]
glue("ratio", (1 + 0.9) / (1 - 0.9), display=False)
```

`depth` is $m$, how much the stiffness swings over a turn. At 0 the spring is a plain one. At
0.9 it is {glue:text}`ratio:.0f` times stiffer at its peak than a half turn later. `peak` is
$\varphi_K$, the phase of the stiffest moment. The left crank has its foot at the bottom at
the phase 0, so `peak=0` is a stiff spring while the foot pushes and a soft one while it
swings. We crawl at the depth 0.9 with eight peaks, and compare with the plain spring.

```{code-cell} python
peaks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
logs = {a: crawl(depth=0.9, peak=a) for a in peaks}
reach = np.array([100 * log.arrays()["q"][-1, 0] for log in logs.values()])
plain = 100 * q[-1, 0]  # [cm], the plain spring of the first run

fig, ax = plt.subplots()
ax.axhline(plain, color=viz.PALETTE[9], ls="--", label="plain spring")
ax.plot(peaks, reach, "o-", label="$m$ = 0.9")
ax.set_xlabel(r"peak phase $\varphi_K$ [rad]")
ax.set_ylabel("distance in 8 s [cm]")
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
margins = [vmc.sim.energy_balance(log)["margin"].min() for log in logs.values()]
assert min(margins) >= 0.0  # passive whatever the peak
best = int(np.argmax(reach))
assert abs(peaks[best] - np.pi) < 1e-9 and reach[best] > 1.2 * plain
assert reach[0] < plain
glue("plain_cm", float(plain), display=False)
glue("stiff_cm", float(reach[0]), display=False)
glue("soft_cm", float(reach[best]), display=False)
glue("gain", float(100 * (reach[best] / plain - 1)), display=False)
```

The plain spring takes the body {glue:text}`plain_cm:.0f` cm. A stiff spring in stance
(`peak=0`) takes it {glue:text}`stiff_cm:.0f` cm, and the opposite (`peak` = $\pi$, soft in
stance) {glue:text}`soft_cm:.0f` cm, {glue:text}`gain:.0f`% further than the plain one. This
depends on the body, so on yours the best peak may be elsewhere. What does not depend on it
is passivity: every run keeps the controller's energy balance above zero (see
[Energy and passivity](energy.md)), because the flywheel pays for the changes of stiffness.

## Steering

`steer` is $u_s$: the left spring has the stiffness $(1 + u_s)$ and the right one
$(1 - u_s)$ times the mean. The two sides then push unequally and the body turns. We crawl
twelve seconds with a steer of $-0.8$, 0 and $0.8$, and draw the path seen from above.

```{code-cell} python
paths = {u: crawl(depth=0.5, steer=u, T=12.0) for u in (-0.8, 0.0, 0.8)}

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for u, log in paths.items():
    q_u = log.arrays()["q"]
    ax.plot(100 * q_u[:, 0], 100 * q_u[:, 1], label=f"$u_s$ = {u}")
ax.set_aspect("equal")
ax.set_xlabel("forward [cm]")
ax.set_ylabel("left [cm]")
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
def yaw(q):  # the heading of a body from its quaternion (w, x, y, z)
    w, x, y, z = q[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

turn = {}
for u, log in paths.items():
    r = log.arrays()
    k = np.floor(r["z"][:, 0] / (2 * np.pi)).astype(int)
    steps = [int(np.argmax(k >= n)) for n in range(2, k.max() + 1)]
    heading = np.degrees(np.unwrap([yaw(x) for x in r["q"]]))
    turn[u] = float(np.diff(heading[steps]).mean())
assert turn[0.8] > 5 and turn[-0.8] < -5 and abs(turn[0.0]) < 0.3
assert abs(turn[0.8] + turn[-0.8]) < 0.05 * turn[0.8]
glue("turn", turn[0.8], display=False)
```

A positive steer turns this crawler to the left, a negative one to the right, by the same
amount: {glue:text}`turn:.1f` degrees per turn of the flywheel at $|u_s|$ = 0.8. The
turning depends on the body: on another robot the stiffer side may push more or less, so
check the sign on yours. The gait phase does not break, because both sides still ride the same
flywheel: steering changes only how hard each side pushes. Keep $|u_s|$ below 1, or one spring
loses all its stiffness.

## Let the crawler find its gait

The best `peak` depends on the body, and we do not know the body. `DitherSeeking` finds it from
the gait's own result. It holds the Param at an estimate plus a slow sinusoidal dither, measures
how the cost, here minus the forward speed, follows the dither, and moves the estimate downhill.
The tone must be slow against the gait: its period is the `window`, six seconds against strides
of a second, and `frequency` is $2\pi n/$`window` for a whole $n$, so that several Params, each with
its own $n$, do not see one another. Every change goes through a `Tank`, whose energy comes from
what the controller's dampers take: a change that would cost more than the tank holds is cut
short. This is the paper's bound on a slow change of the potential. One `Param` is shared by the
two springs, so that they move together:

```{code-cell} python
from virtualmodelcontrol.adaptation import DitherSeeking
from virtualmodelcontrol.control import Tank

peak = vmc.Param("peak", 0.0, unit="rad", scope="stage",
                 bounds=(-np.inf, np.inf))  # one Param, both springs
system = vmc.VirtualMechanismSystem(robot, flywheel(6.0, 0.9, peak=peak))
tank = Tank(vmc.VMCController(vmc.compile(system)), capacity=1.0)
window = 6.0  # [s]
law = DitherSeeking(tank, "ctrl.spring0.peak", amplitude=0.8,
                    frequency=2 * np.pi / window, gain=12.0, window=window)
plant = vmc.sim.ModelPlant(turtle.crawler(), max_step=1e-3,
                           q0=[0, 0, 0.04, 1, 0, 0, 0, 0.0, -np.pi])
tank.reset(0.0, plant.read(), z0=turtle.initial_state(plant.read()))
dt = 1 / turtle.CONTROL_RATE
seen = []
for step in range(int(120 / dt)):
    plant.write(tank.step(plant.t, plant.read()))
    law.step(tank, -plant.v[0], plant.t)  # the cost: minus the speed [m/s]
    if step % 45 == 0:
        seen.append((plant.t, plant.v[0], law.estimate[0], tank.level))
    plant.advance(dt)
seen = np.array(seen)
```

```{code-cell} python
:tags: [remove-input]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.0))
speed = np.convolve(seen[:, 1], np.ones(10) / 10, mode="valid")  # a second
top.plot(seen[9:, 0], 100 * speed)
top.set_ylabel("speed [cm/s]")
bottom.plot(seen[:, 0], seen[:, 2])
bottom.set_xlabel("time [s]")
bottom.set_ylabel(r"estimate of $\varphi_K$ [rad]");
```

```{code-cell} python
:tags: [remove-cell]
first, last = seen[:, 0] < 15, seen[:, 0] > 105
v0, v1 = seen[first, 1].mean(), seen[last, 1].mean()
assert v1 > 1.3 * v0 and seen[:, 3].min() >= -1e-9
glue("seek_start", float(100 * v0), display=False)
glue("seek_end", float(100 * v1), display=False)
glue("seek_gain", float(100 * (v1 / v0 - 1)), display=False)
glue("seek_peak", float(np.mod(seen[-1, 2], 2 * np.pi)), display=False)
```

The crawler starts at {glue:text}`seek_start:.1f` cm/s with the stiff-in-stance spring and, in two
minutes of simulated time, learns to go at {glue:text}`seek_end:.1f` cm/s, {glue:text}`seek_gain:.0f`%
faster. Its estimate of `peak` ends at {glue:text}`seek_peak:.1f` rad (modulo a turn), near the
$\pi$ of the sweep above, and the tank never went empty. The estimate creeps at first because the
cost changes little over a radian, and rings a little at the end: a higher gain rings more. On a
real robot the cost is a measured speed, or a cost of transport from the flywheel's speed, and the
run is a long one that you leave alone.

## Plan the crawler

The planner can run the flywheel controller on the crawler too. `Problem(system, plant=body)`
takes the dynamics from `body`, the crawler, while the controller stays the one written for the
two cranks: it reads the motors of the plant as it does in a simulation. We plan one stride at
the control rate of 150 Hz, from a flywheel that is up to speed (no ramp), once with the
stiffness peak at 0. A `Shooting` is the closed loop of the simulator, step for step, so we
start it from the simulated run, which it must reproduce:

```{code-cell} python
from virtualmodelcontrol import optimization as opt

peak = vmc.Param("peak", 0.0, unit="rad", scope="stage",
                 bounds=(-np.pi, np.pi))
stride = flywheel(6.0, 0.9, peak=peak)
stride.params["drive.ramp_time"].value = 0.0  # the flywheel starts at speed
body = turtle.crawler()
dt, n = 1 / 150, 120  # a control step [s], and the steps of the stride
q0 = [0, 0, 0.04, 1, 0, 0, 0, 0.0, -np.pi]
start = [0.0, 6.0]  # the flywheel's angle [rad] and speed [rad/s]

def simulate(angle):
    peak.value = angle
    system = vmc.VirtualMechanismSystem(robot, stride)
    plant = vmc.sim.ModelPlant(body, q0=q0, max_step=dt)
    controller = vmc.VMCController(vmc.compile(system))
    return vmc.sim.run(plant, controller, vmc.sim.SimClock(dt),
                       T=(n + 1) * dt, z0=lambda meas: np.array(start)
                       ).arrays()

run = simulate(0.0)
system = vmc.VirtualMechanismSystem(robot, stride)
problem = opt.Problem(system, plant=body)
problem.add(opt.Shooting(q0, n * dt, n + 1, v0=np.zeros(8), z0=start,
                         running=False))
guess = {"q": run["q"][: n + 1], "v": run["v"][: n + 1],
         "z": np.vstack([start, run["z"][:n]])}  # z is logged after a step
plan = problem.solve(warm_start=guess)
print(plan.status, plan.iterations, plan.violation)
```

```{code-cell} python
:tags: [remove-cell]
assert plan.converged and plan.violation < 1e-9
assert np.abs(plan.q[:-1] - run["q"][:n]).max() < 1e-9
assert np.abs(plan.u[:-1] - run["law_torque"][:n]).max() < 1e-9
glue("plan_cm", float(100 * plan.q[-1, 0]), display=False)
```

The plan has no work to do: the simulated run satisfies the program's equations to rounding
error ({glue:text}`plan_cm:.1f` cm in the stride), so the planner and the simulator are the same
loop, with the contact, the friction and the flywheel's own motion. To search for a gait with
it we free the `peak` and ask the body to go as far as it can. The distance after the stride, as
a function of the peak, shows what the search is up against:

```{code-cell} python
angles = np.linspace(-np.pi, np.pi, 25)
reach = np.array([simulate(a)["q"][n, 0] for a in angles])  # [m]
fig, ax = plt.subplots()
ax.plot(angles, 100 * reach, "o-")
ax.set_xlabel(r"peak [rad]")
ax.set_ylabel("distance after a stride [cm]");
```

```{code-cell} python
:tags: [remove-cell]
assert 100 * np.ptp(reach) > 2.0  # centimetres, over the peaks of one stride
peak.value = 0.0
problem.add(opt.Cost(body.joint(0) - 1.0, 1.0, t_from=(n - 1) * dt,
                     name="forward"))
problem.free("ctrl.spring0.peak")
planned = problem.solve(warm_start=guess)
assert planned.converged
found = float(planned.params["ctrl.spring0.peak"])
glue("swing_cm", float(100 * np.ptp(reach)), display=False)
glue("best_cm", float(100 * reach.max()), display=False)
glue("plan_peak", found, display=False)
glue("plan_far", float(100 * planned.q[-1, 0]), display=False)
assert reach.max() > planned.q[-1, 0] + 0.005
```

The distance after one stride swings by {glue:text}`swing_cm:.0f` cm as the peak moves, with
several maxima, because the feet catch and slip on the ground. A gradient method follows the
slope under its feet: freeing the peak and starting from 0, the planner ends at
{glue:text}`plan_peak:.2f` rad and {glue:text}`plan_far:.1f` cm, where the sweep has
{glue:text}`best_cm:.1f` cm. This is why the gait above is searched by trial runs
(`DitherSeeking` moves its estimate slowly across many strides). The planner is for smooth
problems: a plan of the crawler is good for what the simulator would do, and for the
derivatives of that, not for finding the best gait of a rugged one.

## Going further

- `limit` is the largest stretch $e_{max}$ of a spring: with it the torque saturates at
  $\bar K(1 + |u_s|)\,e_{max}$, whatever the phase, and the loop stays passive. Saturate the
  potential, never the torque: clipping the torque would remove the reaction on the flywheel.
- `friction`, `crank_radius`, `mass` and the other constants of `turtle.CRAWLER` are keywords of
  `turtle.crawler`, so `crawl(friction=0.4)` gives the feet less grip.
- The crawler is a simple stand-in. Its body, its cranks and its ground are placeholders, not
  the lab's turtle and not measured.
