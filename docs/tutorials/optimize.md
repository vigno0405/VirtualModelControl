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
hold_system = vmc.VirtualMechanismSystem(arm, hold)

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
settle = vmc.VMCController(vmc.compile(hold_system))
vmc.sim.run(plant, settle, vmc.sim.SimClock(1 / 330), T=12.0)
q0 = plant.q.copy()  # the arm at rest under the spring in place
```

## The problem

A `Problem` is a system whose Params we optimize, plus blocks. The `Collocation` block plans the
motion of the closed loop from `q0`:

```{code-cell} python
HORIZON, TRANSITION, NODES = 5.0, 2.0, 21  # [s], [s]


def build(free=(), parameters=(), terms=()):
    problem = opt.Problem(system)
    problem.add(opt.Collocation(q0, HORIZON, NODES, initial=hold_system,
                                transition=TRANSITION))
    problem.free(*free)
    problem.parameter(*parameters)
    problem.add(opt.Effort(0.2))
    problem.add(opt.Cost(tip - target, t_from=TRANSITION, name="reach"))
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
- `Cost(tip - target, t_from=TRANSITION)`: the integral of the squared distance of the tip to
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
stiffnesses = np.geomspace(5.0, 300.0, 25)
costs = np.array([sweep.solve({"new.pull.stiffness": k}).cost
                  for k in stiffnesses])

fig, ax = plt.subplots()
ax.semilogx(stiffnesses, 1e3 * costs, ".-", label="fixed stiffness")
ax.plot([k_found], [1e3 * plan.cost], "o", ms=10, label="optimized")
ax.set_xlabel("stiffness [N/m]")
ax.set_ylabel(r"cost [$10^{-3}$]")
ax.legend(loc="upper right", fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
best = stiffnesses[np.argmin(costs)]
assert abs(np.log(best / k_found)) < 0.3, (best, k_found)
assert plan.cost <= costs.min() * (1 + 1e-3)
glue("best", float(best), display=False)
glue("flat", float(100 * (costs[stiffnesses > 40].max() / plan.cost - 1)),
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
    vmc.VMCController(vmc.compile(hold_system)))
swap.swap(controller, TRANSITION)

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
ax.axvline(TRANSITION, color="gray", linestyle=":")
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

## Good to know

- **Scales.** The solver works on q, v and a divided by `Collocation(scales=(s_q, s_v, s_a))`,
  the typical sizes of the three, `(1, 1, 1)` by default. Scaling does not change the problem,
  but it changes the solver's path, and a problem with several local optima can end in a
  different one: try other scales, or other starting values, when a plan looks poor.
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
- **Not yet.** The controller to plan cannot have virtual states, and the robot's
  configuration must live in a flat space (every robot template here does).

The next steps are in the [roadmap](../development/roadmap.md): other collocation schemes,
shooting, gradient-free tuning and an energy tank for online updates.
