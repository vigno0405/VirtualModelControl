---
file_format: mystnb
kernelspec:
  name: python3
---

# Plan around an obstacle

In this tutorial the optimizer plans the soft arm's swap around a sphere: it chooses the
strength of a repulsive field with the spring's stiffness, then which of five fields to keep.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The arm and the controller in place

They are those of [Optimizing a virtual mechanism](optimize.md): the hanging soft arm, held by a
soft spring, which we let settle first, and the same horizon, swap time and number of nodes:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))  # hangs from its base
tip = arm.point(s=1.0)
kin = vmc.Kinematics(arm)

hold = vmc.Mechanism("hold")  # the controller in place
hold.add("drag", vmc.LinearSpring(tip - [0.01, 0.0, 0.70], 10.0))
hold.add("damp", vmc.LinearDamper(tip, 2.0))
hold.add("gravity", vmc.GravityCompensation(arm))
held = vmc.VirtualMechanismSystem(arm, hold)

plant = vmc.sim.ModelPlant(arm)
settle = vmc.VMCController(vmc.compile(held))
vmc.sim.run(plant, settle, vmc.sim.SimClock(1 / 330), T=12.0)
q0 = plant.q.copy()  # the arm at rest under the spring in place
HORIZON, SWAP, NODES = 5.0, 2.0, 21  # [s], [s]
```

## Around an obstacle

A sphere of radius 2 cm stands by the arm's path to a farther target. We add a repulsive
Gaussian field on the arm, at $s = 0.6$, that pushes that point away from the sphere's center.
The optimizer chooses its strength together with the spring's stiffness. `SphereDistance` is the
signed distance from a point to the sphere's surface, and a `Bound` keeps it above 2 cm at every
node, for 41 points along the arm:

```{code-cell} python
goal = [0.20, 0.0, 0.66]  # [m]
ball, R = [0.15, 0.0, 0.50], 0.02  # the sphere's center and radius [m]

k = vmc.Param("k", 20.0, bounds=(1.0, 300.0), scope="stage")  # [N/m]
A = vmc.Param("A", 50.0, bounds=(0.0, 500.0), scope="stage")  # [N/m]
around = vmc.Mechanism("around")
around.add("pull", vmc.TanhSpring(tip - goal, k, 2.0))
around.add("push", vmc.GaussianSpring(arm.point(s=0.6) - ball, A,
                                      0.06))  # strength A, width 0.06 m
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

The optimizer finds the Params by their place in the system, whatever the Python variables are
called: `around.pull.stiffness` is the stiffness of the element `pull` of the mechanism `around`.
Without the bound the optimizer has no reason to push: it keeps the field off and takes the
cheapest spring. With the bound, it switches the field on. We run both plans on the simulated
arm, as [before](optimize.md#checking-the-plan-in-simulation), and follow the arm's closest distance
to the sphere's surface:

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
from myst_nb import glue

low = [min(gap(q) for q in rows["q"][::5]) for rows in runs.values()]
assert low[0] < 1.0 and 1.9 < low[1] < 2.0, low  # a little under the 2 cm
assert straight.params["around.push.strength"].item() < 1.0  # field off
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

Not on this problem. With other scales, `(0.05, 0.3, 20)`, the solver finds the same plan (the
[scales](optimize.md#good-to-know) are the typical sizes of q, v and a):

```{code-cell} python
other = plan_for([opt.Bound(clear, lower=0.02, name="clear")],
                 scales=(0.05, 0.3, 20.0))
for result in (safe, other):
    print(result.iterations, {n: round(v.item(), 1)
                              for n, v in result.params.items()})
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
problem with several local optima can end in another one with another scaling, so when you
plan a new task, try two scalings and compare.

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

To plan again at every step from the measured state, see [Model predictive control](mpc.md), and
for motions that repeat, [Plan periodic motions and virtual states](optimize-periodic.md).
