---
file_format: mystnb
kernelspec:
  name: python3
---

# Tuning stiffness and damping

In this tutorial we choose the stiffness and damping of a fingertip spring from the finger's
mass, and find the damping beyond which the control loop turns unstable.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The mass the spring moves

A spring pulls the fingertip from its start to a goal. The tip responds according to the mass
the spring feels along that motion, the effective mass $m = 1 / (u^\top J M^{-1} J^\top u)$.
Here $M$ is the finger's mass matrix, $J$ the tip's Jacobian and $u$ the direction of motion:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt

finger = adapt.add_dynamics(adapt.finger())
kin = vmc.Kinematics(finger)
q0 = np.array([0.3, 0.3])  # [rad], the motors' start
start = kin.position(q0, "tip")
goal = kin.position([0.9, 0.7], "tip")  # [m], a reachable goal

plant = vmc.sim.ModelPlant(finger, q0=q0)
M = np.array(plant.dynamics.mass(q0, plant.p))
J = kin.jacobian(q0, "tip")
u = (goal - start) / np.linalg.norm(goal - start)
m = 1.0 / (u @ J @ np.linalg.solve(M, J.T) @ u)
print(f"effective mass {1000 * m:.1f} g")
```

## Stiffness and damping

The stiffness $K$ sets how fast the tip moves, with the natural frequency $\sqrt{K/m}$, and how
hard it holds its goal: a push $F$ moves it by $F/K$. The damping $D$ sets the overshoot: the
critical damping $D_c = 2\sqrt{K m}$ separates a tip that overshoots from one that creeps.

```{code-cell} python
K, dt = 100.0, 1 / 500  # [N/m], and the control period [s]
D_c = 2 * np.sqrt(K * m)

def simulate(D, plant=None, T=1.0):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(finger.point("tip") - goal, K))
    ctrl.add("damp", vmc.LinearDamper(finger.point("tip"), D))
    ctrl.add("gravity", vmc.GravityCompensation(finger))
    system = vmc.VirtualMechanismSystem(finger, ctrl)
    controller = vmc.VMCController(vmc.compile(system))
    plant = plant or vmc.sim.ModelPlant(finger, q0=q0)
    return vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=T)

def distance(log):  # [mm] from the tip to the goal
    tips = np.array([kin.position(q, "tip") for q in log.arrays()["q"]])
    return 1000 * np.linalg.norm(tips - goal, axis=1)

fig, ax = plt.subplots()
for ratio in (0.25, 0.5, 1.0, 2.0):
    log = simulate(ratio * D_c)
    label = f"$D = {ratio:g}\\,D_c$"
    ax.plot(log.arrays()["t"], distance(log), label=label)
ax.set_xlabel("time [s]")
ax.set_ylabel("tip to goal [mm]")
ax.set_xlim(0, 0.5)
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

glue("m", float(1000 * m), display=False)
glue("D_c", float(D_c), display=False)
```

With $K$ = 100 N/m and a {glue:text}`m:.1f` g tip, $D_c$ = {glue:text}`D_c:.2f` N·s/m. Below
it the tip overshoots the goal and rings. At twice $D_c$ it arrives later, without overshooting.

```{code-cell} python
:tags: [remove-output]
log = simulate(0.5 * D_c, T=0.4)
viz.animate(finger, log, "tuning.mp4", plane="yz", invert=True,
            springs=[("tip", goal)], trace="tip", speed=0.1)
```

```{video} tuning.mp4
:caption: With half the critical damping, the fingertip overshoots its goal (red cross) and rings; ten times slower than real time.
```

## Search for the damping

The damping above came from a formula. A search finds it from runs instead. We write a cost, the
distance of the tip to its goal summed over a run, and a searcher picks the dampings to try.
`Grid`, `Random`, `CMAES`, `ExtremumSeeking` and `Bayes` of `vmc.optimization` all ask for
candidates and are told their costs. The runs in between are yours, in simulation as here or on a robot in your
own loop:

```{code-cell} python
from virtualmodelcontrol import optimization as opt


def cost(x):  # [mm s]
    return float(distance(simulate(x[0], T=0.5)).sum() * dt)


grid = opt.tune(opt.Grid([0.1 * D_c], [3 * D_c], 25), cost, rounds=1)

search = opt.CMAES([2 * D_c], 0.5 * D_c, [0.1 * D_c], [3 * D_c], size=6)
for _ in range(5):
    candidates = search.ask()  # the dampings to try
    costs = [cost(x) for x in candidates]  # one run each
    search.tell(candidates, costs)

bayes = opt.tune(opt.Bayes([0.1 * D_c], [3 * D_c], initial=4), cost, 12)

fig, ax = plt.subplots()
x, y = (np.array(h) for h in zip(*grid.history, strict=True))
ax.plot(x[:, 0] / D_c, y, ".-", label="grid")
x, y = (np.array(h) for h in zip(*search.history, strict=True))
ax.plot(x[:, 0] / D_c, y, "o", label="CMA-ES")
x, y = (np.array(h) for h in zip(*bayes.history, strict=True))
ax.plot(x[:, 0] / D_c, y, "s", label="Bayes")
ax.set_xlabel("$D / D_c$")
ax.set_ylabel("cost [mm s]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
glue("grid", float(grid.best[0] / D_c), display=False)
glue("cma", float(search.best[0] / D_c), display=False)
glue("runs", len(search.history), display=False)
glue("bayes", float(bayes.best[0] / D_c), display=False)
worst = max(search.best_cost, bayes.best_cost) / grid.best_cost - 1
assert abs(grid.best[0] - search.best[0]) < 0.3 * D_c, (grid.best, search.best)
assert worst < 0.01, (grid.best_cost, search.best_cost, bayes.best_cost)
glue("flat", 100 * float(worst), display=False)
```

The grid, with 25 runs, puts the best damping at {glue:text}`grid:.2f` $D_c$, CMA-ES, with
{glue:text}`runs` runs, at {glue:text}`cma:.2f` $D_c$, and `Bayes`, with 12 runs, at
{glue:text}`bayes:.2f` $D_c$. The cost is flat near its minimum: the three best costs are within
{glue:text}`flat:.1f` % of each other, so the dampings differ more than the costs do. `Bayes`
fits a Gaussian process to the costs it has seen and asks for the damping where it expects the
most improvement, so it suits runs that are few and slow.
When the whole motion can be planned, [Optimizing a virtual mechanism](optimize.md) finds Params
with gradients. A search is for when a run is all you have. `opt.bounds_of(params, names)` gives
the bounds and the current values of named Params as vectors to start from, and
`params.set_vector(x, names)` sets a candidate back.

## The loop limits the damping

The controller computes the damper's force from velocities measured at the start of each
control period, and a real loop applies it about one period later. A damper that is too strong
then overcorrects at every step: the velocity flips sign and grows. For a sampled loop the
limit is $D < 2m/\Delta t$, and with one period of delay $D < m/\Delta t$.

We find the limits by simulation. The plant below applies each command one step late, like the
plant of [Extend the library](extend.md):

```{code-cell} python
class Late:
    """The simulated finger, receiving each command one step late."""

    def __init__(self, plant):
        self.plant, self.u = plant, np.zeros_like(plant.u)

    @property
    def t(self):
        return self.plant.t

    def read(self):
        return self.plant.read()

    def write(self, cmd):
        self.plant.write(vmc.Signals(self.t, motor_torque=self.u))
        self.u = np.array(cmd["motor_torque"], dtype=float)

    def advance(self, dt):
        self.plant.advance(dt)

def unstable(log):  # the tip still moving fast at the end, or diverged
    v = np.diff(distance(log)[-100:]) / dt
    return not np.all(np.isfinite(v)) or np.abs(v).max() > 10.0  # [mm/s]

dampers = np.arange(1.0, 14.0, 0.5)  # [N·s/m]
first = {}
for name, late in (("sampled", False), ("one step late", True)):
    for D in dampers:
        plant = vmc.sim.ModelPlant(finger, q0=q0)
        if unstable(simulate(D, Late(plant) if late else plant)):
            first[name] = float(D)
            break
print(f"m/dt = {m / dt:.1f} N·s/m; first unstable damper: {first}")
```

```{code-cell} python
:tags: [remove-cell]
glue("limit", float(m / dt), display=False)
glue("limit2", float(2 * m / dt), display=False)
glue("late", first["one step late"], display=False)
glue("sampled", first["sampled"], display=False)
```

The sampled loop turns unstable at {glue:text}`sampled:.1f` N·s/m against
$2m/\Delta t$ = {glue:text}`limit2:.1f`, and the delayed loop at {glue:text}`late:.1f` N·s/m
against $m/\Delta t$ = {glue:text}`limit:.1f`. Keep the damper well below the limit of your
loop. A faster loop, or less delay, allows more damping. On the real soft arm, whose loop has
more delay than one step, tip dampers above about 10 N·s/m oscillated.

## Reading the energies

The compiled controller reports its stored energy and the power of its dampers at any state.
Along a run, the spring's energy should fall as the damper dissipates it:

```{code-cell} python
log = simulate(D_c)
rows = log.arrays()
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(finger.point("tip") - goal, K))
ctrl.add("damp", vmc.LinearDamper(finger.point("tip"), D_c))
compiled = vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl))
p, z = compiled.live_values(), np.zeros(0)
V = [float(compiled.energy(q, v, z, p, t)[0])
     for q, v, t in zip(rows["q"], rows["v"], rows["t"].ravel())]
P = [float(compiled.power(q, v, z, p, t)[1])
     for q, v, t in zip(rows["q"], rows["v"], rows["t"].ravel())]
dissipated = -np.cumsum(P) * dt

fig, ax = plt.subplots()
ax.plot(rows["t"], 1000 * np.array(V), label="stored in the spring")
ax.plot(rows["t"], 1000 * dissipated, label="dissipated by the damper")
ax.set_xlabel("time [s]")
ax.set_ylabel("energy [mJ]")
ax.set_xlim(0, 0.5)
ax.legend();
```

```{code-cell} python
:tags: [remove-cell]
V = np.array(V)
glue("V0", float(1000 * V[0]), display=False)
glue("gone", float(rows["t"].ravel()[np.argmax(V < 0.01 * V[0])]), display=False)
glue("dissipated", float(1000 * dissipated[-1]), display=False)
```

The spring starts with {glue:text}`V0:.0f` mJ ($\tfrac12 K d^2$, for the start distance $d$ to
the goal) and has lost 99 % of it after {glue:text}`gone:.2f` s. The damper dissipates
{glue:text}`dissipated:.0f` mJ, close to that. The difference is the work of the other forces on
the finger, gravity and its compensation, which the [energy tutorial](energy.md) accounts for.
