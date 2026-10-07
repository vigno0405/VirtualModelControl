---
file_format: mystnb
kernelspec:
  name: python3
---

# Optimizing a virtual mechanism

In this tutorial we choose the stiffness of a virtual spring by optimization instead of by
hand. We plan the motion of the soft arm together with the spring that drives it, using the
arm's own dynamics, and then check the plan by running the swap on the simulated arm.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The task

The hanging soft arm is held by a soft spring, the controller in place. We want to swap to a
spring that pulls the tip to a target, with a force of at most 2 N, and to choose its
stiffness: stiff enough to arrive, soft enough not to waste effort.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt, viz
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))  # hangs from its base
tip = arm.point(s=1.0)
target = [0.05, 0.02, 0.71]  # [m]

hold = vmc.Mechanism("hold")  # the controller in place
hold.add("drag", vmc.LinearSpring(tip - [0.01, 0.0, 0.70], 10.0))
hold.add("damp", vmc.LinearDamper(tip, 2.0))
hold.add("gravity", vmc.GravityCompensation(arm))
held = vmc.VirtualMechanismSystem(arm, hold)

stiffness = vmc.Param("stiffness", 20.0, bounds=(1.0, 300.0), scope="stage")
reference = vmc.Ref("reference", 3, target)  # where the spring pulls to
new = vmc.Mechanism("new")  # the controller to plan
spring = vmc.TanhSpring(tip - reference, stiffness, 2.0)  # [N/m], [N]
new.add("pull", spring)
new.add("damp", vmc.LinearDamper(tip, 2.0))
new.add("gravity", vmc.GravityCompensation(arm))
system = vmc.VirtualMechanismSystem(arm, new)
```

The stiffness is a `Param` with bounds, which the optimizer keeps to; `scope="stage"` makes it
live in a compiled controller, so that `controller.set` can change it later. The planned motion
starts at rest, so we first let the arm settle under the controller in place:

```{code-cell} python
plant = vmc.sim.ModelPlant(arm)
settle = vmc.VMCController(vmc.compile(held))
vmc.sim.run(plant, settle, vmc.sim.SimClock(1 / 330), T=12.0)
q0 = plant.q.copy()  # the arm at rest under the spring in place
```

## The problem

A `Problem` is a system whose Params we optimize, plus blocks. The `Collocation` block plans the
motion of the closed loop from `q0`:

```{code-cell} python
HORIZON, SWAP, NODES = 5.0, 2.0, 21  # [s], [s]


def build(free=(), parameters=(), terms=()):
    problem = opt.Problem(system)
    problem.add(opt.Collocation(q0, HORIZON, NODES, initial=held,
                                transition=SWAP))
    problem.free(*free)
    problem.parameter(*parameters)
    problem.add(opt.Effort(0.2))
    problem.add(opt.Cost(tip - target, t_from=SWAP, name="reach"))
    for term in terms:
        problem.add(term)
    return problem


problem = build(free=["new.pull.stiffness"])
plan = problem.solve()
print(plan.status, plan.iterations, plan.params, plan.costs)
```

The unknowns are the arm's configuration, velocity and acceleration at 21 nodes, 0.25 s apart.
The constraints are the arm's dynamics at every node, written without inverting its mass matrix,
and the trapezoid rule between the nodes, so the whole motion is solved at once and errors do
not pile up along the horizon. The motion starts at rest where the arm is.

`initial` is the controller in place. Its torques fade out while the new ones fade in over
`transition`, with the same blend as a [swap](swaps.md), so the plan is what the swap executes.

`free` names the Params to optimize, with the names of `problem.params`: the mechanism, the
element and the Param. It takes glob patterns, such as `new.*.stiffness`. Every Param not named
stays at its value when the program is built. Two terms make the cost:

- `Effort(0.2)`: 0.2 times the integral of the squared motor torques over the horizon;
- `Cost(tip - target, t_from=SWAP)`: the integral of the squared distance of the tip to
  the target, from the end of the transition on: arrive, and stay.

Any library coordinate can stand where `tip - target` stands, and `Bound` keeps a coordinate
within limits (see below).

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

k_found = plan.params["new.pull.stiffness"].item()
glue("status", plan.status, display=False)
glue("iterations", plan.iterations, display=False)
glue("k", k_found, display=False)
glue("effort", plan.costs["effort"], display=False)
glue("reach", plan.costs["reach"], display=False)
glue("violation", plan.violation, display=False)
```

The solver ended with {glue:text}`status` after {glue:text}`iterations` iterations. It chose a
stiffness of {glue:text}`k:.0f` N/m. The cost has two parts, {glue:text}`effort:.2e` for the
effort and {glue:text}`reach:.2e` for the distance to the target, and the largest violation of
a constraint is {glue:text}`violation:.0e`. `plan` also holds the motion at the nodes (`t`, `q`,
`v`, `a`), the motor torques `u` and the weight `blend` of the new controller.

## Is it a minimum?

To see it, we solve again with the stiffness fixed at each of several values. `parameter` makes a
Param an input of each solve instead of a variable, so the program is built once and every solve
is only the solver running again:

```{code-cell} python
sweep = build(parameters=["new.pull.stiffness"])
ks = np.geomspace(5.0, 300.0, 25)
costs = np.array([sweep.solve({"new.pull.stiffness": k}).cost
                  for k in ks])

fig, ax = plt.subplots()
ax.semilogx(ks, 1e3 * costs, ".-", label="fixed stiffness")
ax.plot([k_found], [1e3 * plan.cost], "o", ms=10, label="optimized")
ax.set_xlabel("stiffness [N/m]")
ax.set_ylabel(r"cost [$10^{-3}$]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
best = ks[np.argmin(costs)]
assert abs(np.log(best / k_found)) < 0.3, (best, k_found)
assert plan.cost <= costs.min() * (1 + 1e-3)
glue("best", float(best), display=False)
glue("flat", float(100 * (costs[ks > 40].max() / plan.cost - 1)),
     display=False)
```

The optimized stiffness sits at the bottom of the curve, where the sweep's own minimum is
({glue:text}`best:.0f` N/m on this grid). The bottom is flat: from 40 N/m up, the cost is at
most {glue:text}`flat:.0f` % above the minimum. Too soft a spring arrives late, and a stiffer
one spends effort for little.

## Checking the plan in simulation

The plan uses the same dynamics as the simulator, so running it should give the planned motion.
`apply` puts the optimized Params into a controller, and a `SwapController` blends it in over the
planned transition:

```{code-cell} python
controller = vmc.VMCController(vmc.compile(system))
plan.apply(controller)  # the exact energy jump [J], 0 before its first step
swap = vmc.control.SwapController(
    vmc.VMCController(vmc.compile(held)))
swap.swap(controller, SWAP)

sim = vmc.sim.ModelPlant(arm, q0=q0)
log = vmc.sim.run(sim, swap, vmc.sim.SimClock(1 / 330), T=HORIZON)
```

```{code-cell} python
kin = vmc.Kinematics(arm)
rows = log.arrays()
t = rows["t"].ravel()


def distance(q):  # [mm] from the tip to the target
    return 1e3 * np.linalg.norm(kin.position(q, 1.0) - target)


simulated = np.array([distance(q) for q in rows["q"]])
planned = np.array([distance(q) for q in plan.q])
at_nodes = np.interp(plan.t, t, simulated)

fig, ax = plt.subplots()
ax.plot(t, simulated, label="simulated swap")
ax.plot(plan.t, planned, "o", label="plan")
ax.axvline(SWAP, color="gray", linestyle=":")
ax.set_xlabel("time [s]")
ax.set_ylabel("tip to target [mm]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
gap = np.abs(at_nodes - planned).max()
assert gap < 2.0, gap
glue("gap", float(gap), display=False)
glue("start", float(planned[0]), display=False)
glue("end", float(planned[-1]), display=False)
```

The tip starts {glue:text}`start:.0f` mm from the target and ends {glue:text}`end:.0f` mm from
it: the spring is limited to 2 N and the arm pushes back. The simulated distance follows the
plan within {glue:text}`gap:.1f` mm at the nodes. The dotted line marks the end of the swap.

```{code-cell} python
:tags: [remove-output]
viz.animate(arm, log, "optimize.mp4", springs=[(1.0, target)], trace=1.0,
            invert=True)
```

```{video} optimize.mp4
:caption: The swap to the planned spring: the tip moves towards the target (cross) and stays.
```

## Where the spring pulls to

The spring's reference does not have to be the target. We make it a parameter and let the
stiffness be optimized at each of the reference's candidate points. `search_references` tries
the points of a grid around the target, one reference at a time, and keeps the plan with the
lowest cost among those that converged. The program is built once and every point is only
another solve, warm-started from the best plan so far:

```{code-cell} python
plan.apply(system)  # start from the optimized stiffness
search = build(free=["new.pull.stiffness"],
               parameters=["new.pull.reference"])
found = opt.search_references(
    search, {"new.pull.reference": target}, radius=0.06, step=0.06)

key = "new.pull.reference"
for c in found.candidates:
    offset = 1e3 * (np.array(c["references"][key]) - target)  # [mm]
    cost = np.nan if c["cost"] is None else c["cost"]  # None: not converged
    print(f"{offset.round().astype(int).tolist()} mm  "
          f"{c['status']:<26} {cost:.2e}")
```

```{code-cell} python
:tags: [remove-cell]
anchor = found.candidates[0]["cost"]
offset = 1e3 * (found.references["new.pull.reference"] - target)
assert np.count_nonzero(np.round(offset)) == 1 and abs(offset[2]) > 1, offset
glue("offset", float(np.abs(offset).max()), display=False)
glue("gain", float(100 * (1 - found.result.cost / anchor)), display=False)
glue("k_search", found.result.params["new.pull.stiffness"].item(), display=False)
```

Radius and step are in metres: a radius of 6 cm and a step of 6 cm give the target and its six
neighbours. The best reference lies {glue:text}`offset:.0f` mm from the target along z, and it
lowers the cost by {glue:text}`gain:.1f` %, with a stiffness of {glue:text}`k_search:.0f` N/m.
Moving it sideways costs many times more. `found.result` is the plan, `found.references` the
point that gave it, and `found.candidates` every point tried.

## Where the arm rests

A plan spends its time getting there. When only the pose at rest matters, `Equilibrium` takes the
place of `Collocation`: it solves for a configuration where the controller's torques balance the
arm's own forces, with the same terms and the same free Params. The spring is the optimized one
from above, and nothing is free yet:

```{code-cell} python
def miss(rest):  # [mm] from the tip at rest to the target
    return 1e3 * np.linalg.norm(kin.position(rest.q[0], 1.0) - target)


rest = opt.Problem(system)
rest.add(opt.Equilibrium(q0))  # q0 is the first guess
rest.add(opt.Cost(tip - target, name="miss"))
there = rest.solve()

rest.free("new.pull.reference")  # now ask for the best reference
nearest = rest.solve()
```

```{code-cell} python
:tags: [remove-cell]
assert abs(miss(there) - planned[-1]) < 1.0, (miss(there), planned[-1])
assert miss(nearest) < miss(there)
glue("there", float(miss(there)), display=False)
glue("ms", 1e3 * float(there.seconds), display=False)
glue("nearest", float(miss(nearest)), display=False)
```

The arm rests {glue:text}`there:.1f` mm from the target, where the plan ended, and the solve took
{glue:text}`ms:.0f` ms. With the reference free, the nearest it can rest is
{glue:text}`nearest:.1f` mm: the spring saturates at 2 N, so no reference pulls harder. A
`Bound` holds at the equilibrium too. The node counts once in `Cost` and `Effort`.

## Limits

`Bound` keeps a coordinate within limits at every node but the first, which is the fixed start.
Here the tip must stay below 30 mm in x, and the stiffness has to give way:

```{code-cell} python
wall = opt.Bound(tip[0], upper=0.03, name="wall")
capped = build(free=["new.pull.stiffness"], terms=[wall]).solve()
x_free = 1e3 * np.array([kin.position(q, 1.0)[0] for q in plan.q])
x_capped = 1e3 * np.array([kin.position(q, 1.0)[0] for q in capped.q])

fig, ax = plt.subplots()
ax.plot(plan.t, x_free, label="free")
ax.plot(capped.t, x_capped, label="with the bound")
ax.axhline(30.0, color="gray", linestyle=":")
ax.set_xlabel("time [s]")
ax.set_ylabel("tip x [mm]")
ax.legend(loc="lower right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
assert capped.converged and x_capped[1:].max() <= 30.0 + 1e-3
glue("k_capped", capped.params["new.pull.stiffness"].item(), display=False)
glue("x_free", float(x_free.max()), display=False)
```

Without the bound the tip goes to {glue:text}`x_free:.0f` mm; with it the tip stops at the
limit, and the optimizer picks a softer spring, {glue:text}`k_capped:.1f` N/m. The Bound's
coordinate can be anything the library knows: a joint, a distance to a sphere or a plane (see
[Contact](contact.md)), the sum of several, or your own `vmc.Custom`.

## Around an obstacle

A sphere of radius 2 cm stands by the arm's path to a farther target. We add a repulsive
Gaussian field on the arm, at $s = 0.6$, that pushes that point away from the sphere's centre.
The optimizer chooses its strength together with the spring's stiffness. `SphereDistance` is the
signed distance from a point to the sphere's surface, and a `Bound` keeps it above 2 cm at every
node, for 41 points along the arm:

```{code-cell} python
goal = [0.20, 0.0, 0.66]  # [m]
ball, R = [0.15, 0.0, 0.50], 0.02  # the sphere's centre and radius [m]

k = vmc.Param("k", 20.0, bounds=(1.0, 300.0), scope="stage")  # [N/m]
A = vmc.Param("A", 50.0, bounds=(0.0, 500.0), scope="stage")  # [N/m]
around = vmc.Mechanism("around")
around.add("pull", vmc.TanhSpring(tip - goal, k, 2.0))
around.add("push", vmc.GaussianSpring(arm.point(s=0.6) - ball, A, 0.06))
around.add("damp", vmc.LinearDamper(tip, 2.0))
around.add("gravity", vmc.GravityCompensation(arm))
detour = vmc.VirtualMechanismSystem(arm, around)

body = [arm.point(s=s) for s in np.linspace(0.0, 1.0, 41)]
clear = vmc.Stack(*[vmc.SphereDistance(p, ball, R) for p in body])


def plan_for(terms=(), scales=(1, 1, 1)):
    problem = opt.Problem(detour)
    problem.add(opt.Collocation(q0, HORIZON, NODES, initial=held,
                                transition=SWAP, scales=scales))
    problem.free("around.pull.stiffness", "around.push.strength")
    problem.add(opt.Effort(0.2))
    problem.add(opt.Cost(tip - goal, t_from=SWAP, name="reach"))
    for term in terms:
        problem.add(term)
    return problem.solve()


straight = plan_for()  # the sphere is not in the problem
safe = plan_for([opt.Bound(clear, lower=0.02, name="clear")])
```

Without the bound the optimizer has no reason to push: it keeps the field off and takes the
cheapest spring. With the bound, it switches the field on. We run both plans on the simulated
arm, as before, and follow the arm's closest distance to the sphere's surface:

```{code-cell} python
def swap_to(plan):
    controller = vmc.VMCController(vmc.compile(detour))
    plan.apply(controller)
    swap = vmc.control.SwapController(vmc.VMCController(vmc.compile(held)))
    swap.swap(controller, SWAP)
    sim = vmc.sim.ModelPlant(arm, q0=q0)
    return vmc.sim.run(sim, swap, vmc.sim.SimClock(1 / 330), T=HORIZON)


def gap(q):  # [cm] from the arm to the sphere's surface, at its closest
    points = [kin.position(q, s) for s in np.linspace(0.0, 1.0, 101)]
    return 100 * (min(np.linalg.norm(p - ball) for p in points) - R)


runs = {"without the bound": swap_to(straight).arrays(),
        "with the bound": swap_to(safe).arrays()}

fig, ax = plt.subplots()
for name, rows in runs.items():
    ax.plot(rows["t"][::5], [gap(q) for q in rows["q"][::5]], label=name)
ax.plot(safe.t, [gap(q) for q in safe.q], "o", label="plan")
ax.axhline(2.0, color="gray", linestyle=":")
ax.set_xlabel("time [s]")
ax.set_ylabel("distance to the sphere [cm]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
low = [min(gap(q) for q in rows["q"][::5]) for rows in runs.values()]
assert low[0] < 1.0 and low[1] > 1.9, low
glue("low_straight", float(low[0]), display=False)
glue("low_safe", float(low[1]), display=False)
glue("k_safe", safe.params["around.pull.stiffness"].item(), display=False)
glue("a_safe", safe.params["around.push.strength"].item(), display=False)
glue("a_straight", straight.params["around.push.strength"].item(), display=False)
```

Without the bound the arm passes {glue:text}`low_straight:.1f` cm from the sphere. With it, the
optimizer chose a stiffness of {glue:text}`k_safe:.0f` N/m and a field of
{glue:text}`a_safe:.0f` N/m (it was {glue:text}`a_straight:.0f` before), and the simulated arm
keeps {glue:text}`low_safe:.2f` cm: the bound holds at the nodes and at 41 points of the arm,
and between them the arm can come a little closer. The dotted line marks 2 cm.

### Do the scales matter?

Not on this problem. With other scales, `(0.05, 0.3, 20)`, the solver finds the same plan:

```{code-cell} python
other = plan_for([opt.Bound(clear, lower=0.02, name="clear")],
                 scales=(0.05, 0.3, 20.0))
for plan in (safe, other):
    print(plan.iterations, {n: round(v.item(), 1)
                            for n, v in plan.params.items()})
```

```{code-cell} python
:tags: [remove-cell]
assert abs(other.cost / safe.cost - 1) < 1e-3, (other.cost, safe.cost)
for name, value in safe.params.items():
    assert abs(other.params[name].item() / value.item() - 1) < 0.02, name
glue("iters", safe.iterations, display=False)
glue("iters_other", other.iterations, display=False)
```

Both reach the same stiffness and strength, in {glue:text}`iters` and {glue:text}`iters_other`
iterations. This is a property of the problem. The scales change the solver's path, and a
problem with several local optima can end in another one with another scaling (see below), so
when you plan a new task, try two scalings and compare.

## Which fields to keep

We put the field at $s = 0.6$ ourselves. Let the optimizer choose among five places instead.
`Gated` multiplies an element's force by a gate between 0 and 1, a live Param like the others,
and `Sparsity` adds the sum of the gates to the cost: an element that does not pay for itself
closes. We free the gates and the spring's stiffness:

```{code-cell} python
places = [0.3, 0.45, 0.6, 0.75, 0.9]  # candidate fields, along the arm [s]

fields = vmc.Mechanism("fields")
fields.add("pull", vmc.TanhSpring(tip - goal, k, 2.0))
fields.add("damp", vmc.LinearDamper(tip, 2.0))
for i, s in enumerate(places):
    push = vmc.GaussianSpring(arm.point(s=s) - ball, 200.0, 0.06)
    fields.add(f"push{i}", vmc.Gated(push, 0.5))  # open halfway
fields.add("gravity", vmc.GravityCompensation(arm))
choice = vmc.VirtualMechanismSystem(arm, fields)


def choose(weight):
    problem = opt.Problem(choice)
    problem.add(opt.Collocation(q0, HORIZON, NODES, initial=held,
                                transition=SWAP))
    problem.free("fields.pull.stiffness", "fields.push*.gate")
    problem.add(opt.Effort(0.2))
    problem.add(opt.Cost(tip - goal, t_from=SWAP, name="reach"))
    problem.add(opt.Bound(clear, lower=0.02, name="clear"))
    if weight:
        problem.add(opt.Sparsity(weight, "fields.push*.gate"))
    return problem.solve()


many, few = choose(0.0), choose(3e-3)
gates = {name: [plan.params[f"fields.push{i}.gate"].item()
                for i in range(5)]
         for name, plan in (("no sparsity", many), ("sparsity", few))}

fig, ax = plt.subplots()
x = np.arange(5)
for dx, (name, values) in zip((-0.2, 0.2), gates.items()):
    ax.bar(x + dx, values, 0.4, label=name)
ax.set_xticks(x, [f"{s:g}" for s in places])
ax.set_xlabel("place of the field along the arm [s]")
ax.set_ylabel("gate")
ax.legend(fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
open_many = int(np.sum(np.array(gates["no sparsity"]) > 1e-3))
open_few = int(np.sum(np.array(gates["sparsity"]) > 1e-3))
assert many.converged and few.converged and open_few == 1 < open_many, gates
glue("open_many", open_many, display=False)
glue("place", places[int(np.argmax(gates["sparsity"]))], display=False)
glue("gate", float(max(gates["sparsity"])), display=False)
glue("k_few", few.params["fields.pull.stiffness"].item(), display=False)
```

Without the term, {glue:text}`open_many` fields are open. With it, one is: the field at
s = {glue:text}`place`, with a gate of {glue:text}`gate:.2f` and a spring of
{glue:text}`k_few:.0f` N/m. A gate is a number, not a switch: a gate of 0.3 is a weaker field.
The weight decides how much a field must earn to stay. Structure problems are rough, since
elements switch on and off, and another weight, another start or other scales can end in
another structure, so solve a few and compare.

## A periodic motion with a free period

A gait, a swing or a stroke repeats, so we plan one cycle and ask that it closes on itself.
`Collocation(periodic=True)` makes the last node equal to the first (q and v) and leaves the
first node free: `q0` is only the solver's starting guess. With `free_time=(lower, upper)` the
horizon is an unknown too, within these bounds, and `horizon` is where the solver starts: the
period is found, not given.

The simplest case has a known answer. A mass on a spring, with nothing driving it, swings with
the period $2\pi\sqrt{m/k}$. We pin the amplitude and ask for the orbit:

```{code-cell} python
mass, stiff = 1.0, 4.0  # [kg], [N/m]
robot = vmc.Mechanism("swing", model=vmc.models.JointSpace(1, unit="m"))
pos = robot.joint(0)
robot.add("mass", vmc.Inertance(pos, mass))
robot.add("spring", vmc.LinearSpring(pos, stiff))
free_swing = vmc.VirtualMechanismSystem(robot, vmc.Mechanism("none"))

amp, nodes = 0.3, 41  # [m]
swing = opt.Problem(free_swing)
swing.add(opt.Collocation(
    [amp], 3.0, nodes, periodic=True, free_time=(2.0, 4.5),
    scheme="hermite-simpson"))
swing.add(opt.Bound(pos, amp, amp, t_from=0.0, t_to=0.0, name="start"))
swing.add(opt.Cost(pos, name="small"))
```

`Bound` reaches the first node here, since a periodic motion has no fixed start. A window such
as `t_from=0.0, t_to=0.0` picks its nodes by their times at the starting horizon, and keeps them
when the horizon changes.

The amplitude is pinned at the first node, but a swing that is wider and passes through the
pin also repeats. The small `Cost` on the position chooses the narrowest orbit, the one that
turns at the pin. A periodic orbit is also hard to find from a robot at rest, so we start the
solver from a guess of it, a cosine of the wrong period:

```{code-cell} python
start = 3.0  # [s]
t0 = np.linspace(0.0, start, nodes)
guess = {"q": amp * np.cos(2 * np.pi * t0 / start)[:, None],
         "horizon": start}
orbit = swing.solve(warm_start=guess)
spring_period = 2 * np.pi * np.sqrt(mass / stiff)  # [s]
print(orbit.status, orbit.iterations, orbit.horizon, spring_period)
```

```{code-cell} python
:tags: [remove-cell]
err = abs(orbit.horizon / spring_period - 1)
assert orbit.converged and err < 1e-3, (orbit.status, err)
assert orbit.t[-1] == orbit.horizon or abs(orbit.t[-1] - orbit.horizon) < 1e-9
assert abs(orbit.q[-1, 0] - orbit.q[0, 0]) < 1e-6
glue("p_status", orbit.status, display=False)
glue("p_iter", orbit.iterations, display=False)
glue("p_found", orbit.horizon, display=False)
glue("p_exact", float(spring_period), display=False)
glue("p_err", float(err), display=False)
glue("p_start", start, display=False)
glue("p_nodes", nodes, display=False)
```

The solver ended with {glue:text}`p_status` after {glue:text}`p_iter` iterations, from a
horizon of {glue:text}`p_start:.1f` s. It found a period of {glue:text}`p_found:.4f` s, where
$2\pi\sqrt{m/k}$ is {glue:text}`p_exact:.4f` s: they differ by {glue:text}`p_err:.1e` of
the period, the error of Hermite-Simpson with {glue:text}`p_nodes` nodes. `orbit.horizon` is the
horizon it found, and `orbit.t` the nodes' times on it.

```{code-cell} python
tt = np.linspace(0.0, spring_period, 200)

fig, ax = plt.subplots()
ax.plot(tt, amp * np.cos(2 * np.pi * tt / spring_period), label="exact")
ax.plot(orbit.t, orbit.q[:, 0], "o", ms=5, label="periodic plan")
ax.plot(t0, guess["q"][:, 0], ":", color="gray", label="first guess")
ax.set_xlabel("time [s]")
ax.set_ylabel("position [m]")
ax.legend(loc="lower left", fontsize=18);
```

The orbit closes at the period of the spring, and the dotted guess, which closes at 3 s, does
not.

### A controller that keeps time

A controller can repeat too. Its reference is a function of time, `vmc.Time()`, with the period
in a `Param`. The orbit repeats if the horizon is that period, and `opt.Period(name)` ties the
two: it holds the horizon equal to the Param, which can be free like any other. Here a spring
pulls the swinging mass towards a goal that moves as $a\cos\omega t + b\sin\omega t$, with
$a$, $b$ and the period free. We ask for the orbit through the amplitude $0.3$ m that costs the
least effort:

```{code-cell} python
import casadi as ca

period = vmc.Param("period", 3.0, unit="s", bounds=(1.5, 6.0),
                   scope="episode")
cos = vmc.Param("cos", 0.0, unit="m", scope="episode")
sin = vmc.Param("sin", 0.0, unit="m", scope="episode")


def goal(t, period, cos, sin):  # repeats every `period` seconds
    phase = 2 * np.pi * t / period
    return cos * ca.cos(phase) + sin * ca.sin(phase)


rub = 0.8  # friction [N s/m]
rubbed = vmc.Mechanism("rubbed", model=vmc.models.JointSpace(1, unit="m"))
pos2 = rubbed.joint(0)
rubbed.add("mass", vmc.Inertance(pos2, mass))
rubbed.add("spring", vmc.LinearSpring(pos2, stiff))
rubbed.add("friction", vmc.LinearDamper(pos2, rub))

moving = vmc.Custom(goal, [vmc.Time()], dim=1, unit="m", params={
    "period": period, "cos": cos, "sin": sin})
pull = vmc.Mechanism("pull")
pull.add("spring", vmc.LinearSpring(pos2 - moving, 10.0))
keeps_time = vmc.VirtualMechanismSystem(rubbed, pull)

least = opt.Problem(keeps_time)
least.add(opt.Collocation(
    [amp], 3.0, nodes, periodic=True, free_time=(1.5, 6.0),
    scheme="hermite-simpson"))
least.free("pull.spring.period", "pull.spring.cos", "pull.spring.sin")
least.add(opt.Period("pull.spring.period"))
least.add(opt.Bound(pos2, amp, amp, t_from=0.0, t_to=0.0, name="start"))
least.add(opt.Effort(1.0))
best = least.solve()
print(best.status, best.horizon, best.params["pull.spring.period"])
```

The force the controller must give is $m\ddot x + c\dot x + kx$. For $x = a\cos\omega t$ the
effort over a period is $\frac{T}{2}a^2\left((k - m\omega^2)^2 + c^2\omega^2\right)$ with
$T = 2\pi/\omega$, and its minimum in $T$ is at the positive root $s$ of
$3m^2s^2 + (c^2 - 2km)s - k^2 = 0$, with $s = \omega^2$:

```{code-cell} python
m, c, k = mass, rub, stiff
s = max(np.roots([3 * m**2, c**2 - 2 * k * m, -(k**2)]).real)
period_best = 2 * np.pi / np.sqrt(s)


def effort(T):  # [N^2 s] over one period, for x = amp cos(2 pi t / T)
    w = 2 * np.pi / T
    return T / 2 * amp**2 * ((k - m * w**2) ** 2 + (c * w) ** 2)


Ts = np.linspace(1.5, 6.0, 200)
fig, ax = plt.subplots()
ax.plot(Ts, effort(Ts), label="closed form")
ax.plot([best.horizon], [best.cost], "o", ms=10, label="found")
ax.axvline(spring_period, color="gray", linestyle=":")
ax.set_xlabel("period [s]")
ax.set_ylabel(r"effort [N$^2$ s]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
assert best.converged, best.status
assert abs(best.horizon / period_best - 1) < 1e-3, (best.horizon, period_best)
assert abs(best.cost / effort(period_best) - 1) < 1e-3
tied = best.params["pull.spring.period"].item()
assert abs(tied / best.horizon - 1) < 1e-9
glue("b_found", best.horizon, display=False)
glue("b_exact", float(period_best), display=False)
glue("b_cost", best.cost, display=False)
glue("b_spring", float(spring_period), display=False)
glue("b_iter", best.iterations, display=False)
```

The solver found a period of {glue:text}`b_found:.3f` s and an effort of {glue:text}`b_cost:.4f`
N$^2$ s in {glue:text}`b_iter` iterations; the closed form says {glue:text}`b_exact:.3f` s. It
is not the period of the spring alone ({glue:text}`b_spring:.3f` s, the dotted line): friction
moves it. `best.params` holds the goal's $a$ and $b$ and the period, and `apply` puts them into
a controller like any other result.

Good to know about periodic problems and a free horizon:

- **A guess of the orbit.** From a robot at rest the solver may not find a swing, on a coarse
  grid especially. Give `solve` a `warm_start` with the orbit's `q` (and `v`, `a`) and the
  `horizon`.
- **Windows.** `Cost` and `Bound` choose their nodes by the times of the starting horizon and
  keep those nodes whatever the horizon becomes. `Effort` and `Cost` integrate with the horizon
  that is found. To write your own term, use `trajectory.dt`, `trajectory.times` and
  `trajectory.horizon`: they are numbers for a fixed horizon and expressions of the free one
  otherwise (a cost on the time itself is `weight * trajectory.horizon`).
- **Equations and unknowns.** The solver needs no more equations than unknowns. A swing with a
  fixed horizon that pins its start by an equality has one too many: pin it from one side
  (`Bound(pos, amp, None, ...)`) and let the cost close the bound, or free the horizon.
- **Not together.** `initial` and `transition` blend two controllers by the nodes' times, which
  must be numbers: they go with neither `periodic` nor `free_time`.

## A controller with virtual states

A controller can have states of its own, such as a virtual mass that the robot is tied to, or the
flywheel of a gait. The plan then has them as unknowns too, with the controller's own dynamics,
and nothing changes in the problem: `Collocation`, `Equilibrium` and `Shooting` find the states
in the controller. As an example, a mass is held by a spring to a virtual mass $z$, and $z$ is
pulled by a spring to a goal. We free the goal, and ask for the mass to be at 1 m after 2 s:

```{code-cell} python
line = vmc.Mechanism("line", model=vmc.models.JointSpace(1, unit="m"))
pos3 = line.joint(0)
line.add("mass", vmc.Inertance(pos3, 1.0))
line.add("friction", vmc.LinearDamper(pos3, 1.0))

goal = vmc.Param("goal", 1.0, bounds=(-5.0, 5.0), unit="m", scope="stage")
follower = vmc.Mechanism("follower")
z = follower.add_state("z", 1, unit="m")  # the virtual mass
follower.add("inertia", vmc.Inertance(z, 0.5))
follower.add("link", vmc.LinearSpring(pos3 - z, 10.0))
follower.add("anchor", vmc.LinearSpring(z - vmc.Ref("goal", 1, goal), 4.0))
follower.add("damper", vmc.LinearDamper(z, 2.0))
tied = vmc.VirtualMechanismSystem(line, follower)

reach = opt.Problem(tied)
reach.add(opt.Collocation([0.0], 3.0, 31, scheme="hermite-simpson"))
reach.add(opt.Bound(pos3, 1.0, 1.0, t_from=2.0, t_to=2.0, name="arrive"))
reach.add(opt.Effort(0.1))
reach.free("follower.anchor.goal")
tied_plan = reach.solve()
print(tied_plan.status, tied_plan.params, tied_plan.z.shape)
```

`plan.z` holds the virtual state at every node, the position of $z$ and then its velocity, like
`q` and `v` for the robot. The state starts where the controller starts, which is its `initial`
value (0 here), at rest; `Collocation(z0=...)` starts it elsewhere. We check the plan by applying
the goal and running the closed loop on a simulation with a finer step:

```{code-cell} python
tied_plan.apply(tied)
check = vmc.sim.rollout(tied, [0.0], 3.001, 0.001, max_step=0.001,
                      integrator="rk4")
rows = np.round(tied_plan.t / 0.001).astype(int)
error = np.abs(tied_plan.q[:, 0] - check["q"][rows, 0]).max()

fig, ax = plt.subplots()
ax.plot(tied_plan.t, tied_plan.q[:, 0], label="mass")
ax.plot(tied_plan.t, tied_plan.z[:, 0], label="virtual mass")
ax.plot(check["t"], check["q"][:, 0], "k:", label="simulation")
found = tied_plan.params["follower.anchor.goal"]
ax.axhline(found, color="gray", linestyle="--")
ax.set_xlabel("time [s]")
ax.set_ylabel("position [m]")
ax.legend(loc="lower right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
assert tied_plan.converged, tied_plan.status
assert abs(tied_plan.q[20, 0] - 1.0) < 1e-6 and tied_plan.z.shape == (31, 2)
assert error < 2e-3, error
assert np.abs(tied_plan.z[:, 0]).max() > 0.5
glue("v_goal", float(tied_plan.params["follower.anchor.goal"]), display=False)
glue("v_error", float(1e3 * error), display=False)
```

The goal that does it is {glue:text}`v_goal:.3f` m: the mass passes 1 m at 2 s and overshoots
the goal. The plan agrees with the simulation to {glue:text}`v_error:.1f` mm. A controller
advances its virtual states as a step of the simulator does, so a plan from `Shooting` is the
closed loop of that simulation to rounding error.

- **A controller in place with states.** `initial` may have virtual states, as a system (it
  starts with the plan) or as a running controller, whose state is where it is now.
- **A periodic motion** repeats the virtual states too. A state that only grows, such as the
  angle of a flywheel, has no periodic orbit.
- **At rest.** `Equilibrium` puts the states at rest too, with their accelerations at zero, so
  a controller with a drive has none.

## Good to know

- **Scales.** The solver works on q, v and a divided by `Collocation(scales=(s_q, s_v, s_a))`,
  the typical sizes of the three, `(1, 1, 1)` by default. Scaling does not change the problem,
  but it changes the solver's path, and a problem with several local optima can end in a
  different one: try other scales, or other starting values, when a plan looks poor.
- **Schemes.** `Collocation(scheme="hermite-simpson")` integrates to fourth order instead of
  second. It adds the acceleration at the middle of every interval to the unknowns, so each node
  costs more, but far fewer nodes give the same agreement with a simulation.
- **Starting values.** The free Params start at their current values, and the `warm_start`
  argument of `solve` takes a previous `Result` (or a dict of `q`, `v`, `a` and `params`) to
  start from its plan.
- **What is rebuilt.** The values of the free Params and of the parameters, and the bounds of
  the free Params, are read at every solve. Everything else (a gain, a limit such as the 2 N
  here) is folded in when the program is built: to change it, make it free or a parameter, or
  build a new `Problem`.
- **Options.** `Problem(system, options={...})` overrides IPOPT's defaults, listed in
  `opt.IPOPT`, and takes any other IPOPT option, such as `"ipopt.max_wall_time"`. Changing
  `problem.options` later makes the next solve create the solver again.
- **Status.** `plan.converged` is true for a solution and for an acceptable one; read
  `plan.status` and `plan.violation` before trusting a plan.
- **Your own term.** Subclass `opt.Term` and write `cost(trajectory)` and
  `constraints(trajectory)`, and `coordinates()` to list the coordinates it uses;
  `trajectory.coordinate(c)` gives a coordinate and its rate at every node.
- **Stopping.** `progress(iteration, cost, params, q)` of `solve` is called at every
  iteration; return `True` from it to stop the solve there (the status is
  `User_Requested_Stop`).
- **A larger plant.** `Problem(system, plant=robot)` takes the dynamics from `robot` while the
  controller stays written for other coordinates (the two cranks of the turtle, a robot's
  motors): it reads the plant's motors as it does in a simulation. `q0` and the terms are in
  the plant's coordinates, and a plan equals the simulation step for step
  ([Crawl with a flywheel](crawl.md) shows it on the crawler).
- **Not yet.** `Collocation` needs the robot's configuration in a flat space (every robot
  template here does); `Shooting` also plans a floating body, with unit quaternions at its
  nodes.

To change a running controller within an energy budget, see [Energy and
passivity](energy.md), and to tune one by trial runs instead of a plan, see
[Tuning](tuning.md). To plan again at every step from the measured state, with multiple shooting
and a tank, see [Model predictive control](mpc.md). The next steps are in the
[roadmap](../development/roadmap.md): other solvers.
