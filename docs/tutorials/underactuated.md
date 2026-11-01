---
file_format: mystnb
kernelspec:
  name: python3
---

# Robots with fewer motors than joints

Some robots have joints that no motor drives: a finger whose last joint follows a spring, a
soft arm whose tendons bend it in fewer ways than it can bend. A virtual spring on such a robot
asks for torques that the motors cannot all give. In this tutorial we control a planar arm
with three joints and two motors, see what is lost, and compare the two ways the library
renders the torques, with and without a correction that keeps the arm passive. If you only want
to know which controller to take, read [Choosing](#choosing) at the end.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The arm

The arm moves in the vertical plane. Motors drive joints 1 and 3; joint 2 is passive, with a
torsional spring and a damper of its own. `planar.arm` builds the robot and
`planar.add_dynamics` gives it gravity and the passive joint's spring and damper.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.control import underactuated as ua
from virtualmodelcontrol.robots import planar

arm = planar.add_dynamics(planar.arm("three-link"))
kin = vmc.Kinematics(arm)
B = arm.actuation.params["B"].value  # the motors' input matrix
print(B)
```

```{code-cell} python
:tags: [remove-input]
q = np.array([0.5, 0.3, 0.2])
fig, ax = plt.subplots()
viz.draw_robot(ax, arm, q, plane="xy")
passive = viz.skeleton(arm, q)[0][1]  # joint 2
ax.plot(passive[0], passive[1], "o", ms=26, mfc="none", mew=3,
        color=viz.PALETTE[1])
ax.set_xlabel("x [m]")
ax.set_ylabel("height [m]");
```

The motor torques $u$ reach the joints as $\tau = B u$. Joint 2 gets nothing from them. This is
the actuation `models.Underactuated`; `Direct` is the case with a motor on every joint.

## What the motors can give

A virtual spring and damper on the tip give a force $F$, and the torque it asks of the joints
is $J^\top F$. The motors give the part of it they can reach, $B B^+ J^\top F$, with
$B^+ = (B^\top B)^{-1} B^\top$. What is left is the torque defect $E J^\top F$, with
$E = I - B B^+$ the projector on the torques no motor gives. Here $E$ keeps only joint 2.

```{code-cell} python
q = np.array([0.5, 0.3, 0.2])
J = kin.jacobian(q, "tip")[:2]  # the tip in the plane: x and y
F = np.array([2.0, -1.0])  # [N]
lost = ua.defect(B, J, F)  # [N·m]
asked = J.T @ F
print("asked", asked, "lost", lost)
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

assert abs(lost[[0, 2]]).max() < 1e-12 and abs(lost[1]) > 0.01
assert np.allclose(lost, ua.projector(B) @ asked)
glue("lost_joint", float(lost[1]), display=False)
glue("asked_joint", float(asked[1]), display=False)
```

The defect is {glue:text}`lost_joint:.2f` N·m at joint 2, the whole of the
{glue:text}`asked_joint:.2f` N·m that the force asked of it. Joint 2 moves the tip along one
direction only, and a force at the tip is realized in full when it asks nothing of joint 2: when
it is perpendicular to that direction, so that its torque at joint 2 is zero. These forces are the
feasible set, here a line:

```{code-cell} python
basis = ua.feasible(B, J)  # columns: forces the motors give in full
print(basis.T, ua.defect(B, J, 3.0 * basis[:, 0]))
```

The basis is that perpendicular direction, and the defect of a force along it is zero.

A spring of the controller can pull the tip only along such a direction without error; any
other pull is partly absorbed by the passive joint, which moves until its own spring balances
what is left.

## A reach, rendered three ways

The controller is a spring from the tip to a goal, a damper, and the compensation of the
arm's weight. `ua.controller(compiled, base, correction=None, *, rest="constant", at=None,
tank=1.0)` builds it, and `base` chooses how its torques are rendered
([the theory](../concepts/underactuated.md) says why):

- `"naive"` sends the motors $u = B^+ J^\top F$, the least-squares torques at the measured
  state. It needs the whole state $(q, v)$, passive joint included, so it is a
  `StateController`: it reads $q$ and $v$ of the measurement, not the motors'.
- `"frozen"` evaluates the same law with the passive joint held at a rest, whatever it does.
  It needs the motors only, and its torque is the exact gradient of a potential. The rest is
  constant, `rest="constant"`: the `q_rest` of the actuation, or with `at=` the motor angles
  of a pose, the balance there of the passive joint's spring against the arm's own weight.
  Or the rest moves, `rest="moving"`: the balance at the current motor angles, found again at
  every reading, so there is no pose to choose.

```{code-cell} python
def reach(robot, compensate=True, stiffness=150.0, damping=25.0):
    tip = robot.point("tip")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - [0.45, 0.15, 0.0], stiffness))
    ctrl.add("damp", vmc.LinearDamper(tip, damping))
    if compensate:
        ctrl.add("gravity", vmc.GravityCompensation(robot))
    return vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))

def run(compiled, base, correction=None, T=4.0, **options):
    controller = ua.controller(compiled, base, correction, **options)
    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(5e-4), T=T)
    return log.arrays()

compiled = reach(arm)
runs = {"naive": run(compiled, "naive"),
        "frozen": run(compiled, "frozen"),
        "frozen, moving rest": run(compiled, "frozen", rest="moving")}
```

```{code-cell} python
def tip_error(rows):
    tips = np.array([kin.position(x, "tip")[:2] for x in rows["q"][::10]])
    return np.linalg.norm(tips - [0.45, 0.15], axis=1)  # [m]

fig, ax = plt.subplots()
for name, rows in runs.items():
    ax.plot(np.ravel(rows["t"])[::10], 1000 * tip_error(rows), label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel("distance to the goal [mm]")
ax.legend(fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
final = {name: float(tip_error(rows)[-1]) for name, rows in runs.items()}
assert final["naive"] < 1e-3 and final["frozen, moving rest"] < 1e-3 and final["frozen"] > 0.05
bare = reach(arm, compensate=False)
bare_final = {
    "naive": float(tip_error(run(bare, "naive"))[-1]),
    "frozen": float(tip_error(run(bare, "frozen"))[-1]),
    "frozen, moving rest": float(tip_error(run(bare, "frozen", rest="moving"))[-1]),
}
assert bare_final["frozen"] > 1.5 * bare_final["naive"]
assert bare_final["frozen, moving rest"] < bare_final["frozen"]
glue("final_naive", 1000 * final["naive"], display=False)
glue("final_frozen", 1000 * final["frozen"], display=False)
glue("final_moving", 1000 * final["frozen, moving rest"], display=False)
glue("bare_naive", 1000 * bare_final["naive"], display=False)
glue("bare_frozen", 1000 * bare_final["frozen"], display=False)
glue("bare_moving", 1000 * bare_final["frozen, moving rest"], display=False)
```

With the weight compensated, the naive controller settles on the goal, within
{glue:text}`final_naive:.1e` mm. The frozen controller with its constant rest at zero stops
{glue:text}`final_frozen:.0f` mm short: it renders the force as if the passive joint were at its
rest angle, but the joint carries the weight of the links beyond it and sags. With
`rest="moving"` it knows that and settles within {glue:text}`final_moving:.1e` mm. Without the
compensation, the arm's own weight pulls the tip away, and the controllers end
{glue:text}`bare_naive:.0f` mm from the goal (naive), {glue:text}`bare_moving:.0f` mm (moving
rest) and {glue:text}`bare_frozen:.0f` mm (frozen, rest at zero). Where the passive joint is at
rest and carries nothing, naive and frozen send the same torques.

## A passive joint under weight

The frozen controller keeps the arm passive because its torque is the gradient of a potential:
the field of motor angles $\theta \mapsto u(\theta)$ has no curl, so the motors cannot gain
energy by going around a loop. That holds for both rests, however the rest is chosen
([why](../concepts/underactuated.md#why-the-frozen-controller-is-passive)). We measure the curl as
the asymmetry of the Jacobian of $u$ with respect to $\theta$, relative to its size, for the
constant rest at zero, for the constant rest balanced at the motor angles where the naive reach
ends, and for the moving rest:

```{code-cell} python
def curl(theta, h=1e-6, **options):
    controller = ua.controller(compiled, "frozen", **options)

    def torque(theta):
        reading = vmc.Signals(0.0, motor_position=theta, q=[0.0] * 3,
                              motor_velocity=[0.0, 0.0], v=[0.0] * 3)
        controller.reset(0.0, reading)
        return np.asarray(controller.step(0.0, reading)["law_torque"])

    jac = np.column_stack([(torque(theta + h * e) - torque(theta - h * e))
                           / (2 * h) for e in np.eye(2)])
    return abs(jac - jac.T).max() / abs(jac).max()

pose = B.T @ runs["naive"]["q"][-1]  # the motor angles where the reach ends
curls = {"constant": curl(pose), "balanced": curl(pose, at=pose),
         "moving": curl(pose, rest="moving")}
```

```{code-cell} python
:tags: [remove-cell]
assert max(curls.values()) < 1e-7
balanced = run(compiled, "frozen", at=pose)
final_balanced = 1000 * float(tip_error(balanced)[-1])
assert final_balanced < 1.0, final_balanced
glue("curl_constant", float(curls["constant"]), display=False)
glue("curl_balanced", float(curls["balanced"]), display=False)
glue("curl_moving", float(curls["moving"]), display=False)
glue("final_balanced", final_balanced, display=False)
```

None has a curl: {glue:text}`curl_constant:.0e` for the constant rest,
{glue:text}`curl_balanced:.0e` for the balanced one and {glue:text}`curl_moving:.0e` for the
moving one. What differs is where the arm ends. The constant rest at zero holds the passive joint
straight while it sags, hence the {glue:text}`final_frozen:.0f` mm above. Three ways to give the
frozen controller the rest the arm really has:

- **Balance it once.** `at=` is the motor angles of the pose the arm works around, such as where
  the reach ends. The controller then ends {glue:text}`final_balanced:.1e` mm from the goal. The
  rest it found is `ua.Frozen(arm.actuation, dynamics, at=pose).rest`, which can be set as the
  `q_rest` of the actuation, the way [a plan](#planning-with-these-controllers) reads it.
- **Let a plan find it.** `q_rest` is a Param, so a plan can free it (below).
- **Let it move.** `rest="moving"` solves the balance at every reading, from the robot's model:
  the springs, the weight, any constant load in it. The arm reaches the goal from wherever it
  starts, with no pose to choose. The weight compensation is rendered at that balance, which
  cancels the arm's weight on the motors exactly; the springs and dampers are pulled back
  through the balance's derivative, which keeps them passive even if the model is a little wrong.

## Keeping the arm passive

The naive torque is not the gradient of anything, so a controller built on it can put energy
into the arm that no damper took out. Two output stages of the controller limit that. The
passive correction removes the least torque that keeps the power the motors inject below the
power the arm's own dampers dissipate, $\dot\theta^\top u \le P_D$, with $\dot\theta$ the motor
rates and $P_D$ the power of those dampers. The tank correction
lets the motors inject more while a tank, which starts with a budget of energy and is refilled
by the dissipation, is not empty. Both read the passive joint's velocity, so the measurement
must hold $q$ and $v$ (a simulated plant reports them).

```{code-cell} python
dynamics = vmc.compile_dynamics(arm)
p = dynamics.live_values()

def energy(rows):  # [J], kinetic and stored energy of the arm itself
    return np.array([sum(float(e) for e in dynamics.energy(q, v, p, 0.0))
                     for q, v in zip(rows["q"][::10], rows["v"][::10])])

corrected = {name: run(compiled, "naive", name)
             for name in (None, "passive", "tank")}
fig, ax = plt.subplots()
for name, rows in corrected.items():
    ax.plot(np.ravel(rows["t"])[::10], energy(rows),
            label=name or "no correction")
ax.set_xlabel("time [s]")
ax.set_ylabel("energy of the arm [J]")
ax.legend(fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
peak = {name: float(energy(rows).max() - energy(rows)[0]) for name, rows in corrected.items()}
assert peak[None] > 1.0 and peak["passive"] < 0.05 and 0.9 < peak["tank"] < 1.1


def gain(dt):  # [J] the passive correction lets through in the first second
    controller = ua.controller(compiled, "naive", "passive")
    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    rows = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt), T=1.0)
    return float(energy(rows.arrays()).max() - energy(rows.arrays())[0])


assert gain(2.5e-4) < 0.5 * gain(5e-4)  # it falls with the control period
glue("peak_none", peak[None], display=False)
glue("peak_passive", peak["passive"], display=False)
glue("peak_tank", peak["tank"], display=False)
```

Without a correction the arm gains up to {glue:text}`peak_none:.2f` J from the controller. With
the passive correction it gains {glue:text}`peak_passive:.3f` J at most, the rounding of the
control period (the guarantee holds in continuous time and the gain falls with the period). With
the tank it gains {glue:text}`peak_tank:.2f` J, the budget it started with (the default
`tank=1.0` J of `ua.controller`).

The `width` of a correction, in W, smooths the point where it starts. Its default, 1 W, is as
large as the power of this arm, so the correction also slows the reach when little needs
correcting: after 4 s the passive correction is still {glue:text}`slow_default:.0f` mm from the
goal, and with `width=1e-2` it is {glue:text}`slow_narrow:.1f` mm. Take the smallest power that
matters in your loop.

```{code-cell} python
:tags: [remove-cell]
narrow = run(compiled, "naive", "passive", width=1e-2)
slow_default = 1000 * float(tip_error(corrected["passive"])[-1])
slow_narrow = 1000 * float(tip_error(narrow)[-1])
assert slow_default > 10.0 and slow_narrow < 1.0, (slow_default, slow_narrow)
glue("slow_default", slow_default, display=False)
glue("slow_narrow", slow_narrow, display=False)
```

The frozen controller needs no correction to keep the energy of its own virtual spring from
growing, since its torque is a gradient, with either rest.
Do not stack them. A correction compares the power the motors inject with what the arm's own
dampers dissipate, and a virtual spring that is releasing its energy injects power, so it also
cuts the torque of the frozen controller:

```{code-cell} python
def removed(base, T=4.0, dt=5e-4):
    """The torque [N·m s] that the passive correction takes off the law."""
    controller = ua.controller(compiled, base, "passive", width=1e-4)
    plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=1e-4)
    controller.reset(0.0, plant.read())
    total = 0.0
    for _ in range(round(T / dt)):
        command = controller.step(plant.t, plant.read())
        total += dt * float(np.abs(command["law_torque"]
                                   - command["motor_torque"]).sum())
        plant.write(command)
        plant.advance(dt)
    return total

cut = {base: removed(base) for base in ("naive", "frozen")}
```

```{code-cell} python
:tags: [remove-cell]
assert cut["frozen"] > 0.1 and cut["naive"] > 0.1, cut
glue("cut_frozen", float(cut["frozen"]), display=False)
glue("cut_naive", float(cut["naive"]), display=False)
```

Over this reach the correction takes {glue:text}`cut_frozen:.1f` N·m·s off the frozen torque and
{glue:text}`cut_naive:.1f` N·m·s off the naive one. Use the corrections on the naive controller.

## A force along a direction

`DirectionalForce` makes the arm push with a chosen force along one direction by adapting one
number of the stiffness. It keeps $K' = K + s\,n n^\top$, which is symmetric for every $s$, and
moves $s$ to the value that makes the force the motors realize along $n$ equal the wanted one,
without ever losing $K' \succ 0$. The spring's stiffness must be a live $3 \times 3$ matrix: the
spring acts on the tip in 3D, $z$ included, although the arm moves in a plane. Here the arm is
held at a pose and asked for 3 N along $x$. `rate` is the share of the way to the right $s$
that it covers per second, and `step(controller, q, v, dt)` takes the state the law is
evaluated at and the time step, 10 ms here:

```{code-cell} python
compiled = reach(arm, stiffness=150.0 * np.eye(3))
controller = ua.controller(compiled, "naive")
tracker = ua.DirectionalForce(controller, "tip", "ctrl.reach.stiffness",
                              [1, 0, 0], 3.0, rate=2.0)
q, v = np.array([0.5, 0.3, 0.2]), np.zeros(3)
motors = B.T
hold = vmc.Signals(0.0, motor_position=motors @ q,
                   motor_velocity=motors @ v, q=q, v=v)
controller.reset(0.0, hold)
controller.step(0.0, hold)
pushed = []
for _ in range(500):
    tracker.step(controller, q, v, 1e-2)
    pushed.append(tracker.reading)
fig, ax = plt.subplots()
ax.plot(1e-2 * np.arange(500), pushed)
ax.axhline(3.0, color=viz.PALETTE[1], ls="--")
ax.set_xlabel("time [s]")
ax.set_ylabel("force along $x$ [N]");
```

```{code-cell} python
:tags: [remove-cell]
K = np.reshape(controller.live_params()["ctrl.reach.stiffness"], (3, 3))
assert abs(pushed[-1] - 3.0) < 0.05 and np.linalg.eigvalsh(K).min() > 0
glue("force_final", float(pushed[-1]), display=False)
glue("force_s", float(tracker.s), display=False)
glue("force_eig", float(np.linalg.eigvalsh(K).min()), display=False)
```

The force settles at {glue:text}`force_final:.2f` N. The stiffness along $x$ changed by
$s$ = {glue:text}`force_s:.1f` N/m, and the smallest eigenvalue of $K'$ is
{glue:text}`force_eig:.1f` N/m, still positive. In your own loop, call `tracker.step` after
each control step with the state the law was evaluated at.

## Harder cases

The five-link arm has joints 2, 3 and 4 passive; the continuum arm has three bending sections
and two tendons, so the bending of section 2 against 3 has no motor. Both are built the same
way and compile with the same controllers.

```{code-cell} python
robots = {"three-link": planar.arm("three-link"),
          "five-link": planar.arm("five-link"),
          "continuum": planar.continuum()}
for name, robot in robots.items():
    n = robot.model.space.nq
    Bn = robot.actuation.params["B"].value
    rank = round(float(np.trace(ua.projector(Bn))))  # rank of E
    print(f"{name:11s} {n} coordinates, {Bn.shape[1]} motors, "
          f"{rank} passive directions")
```

```{code-cell} python
:tags: [remove-input]
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
poses = {"three-link": [0.5, 0.3, 0.2], "five-link": [0.4, 0.3, 0.2, 0.1, 0.2],
         "continuum": [0.8, 0.5, 0.4]}
for ax, (name, robot) in zip(axes, robots.items()):
    plane = "xz" if name == "continuum" else "xy"
    viz.draw_robot(ax, robot, poses[name], plane=plane)
    ax.set_title(name)
    ax.set_xlabel("x [m]")
axes[0].set_ylabel("height [m]");
```

## Planning with these controllers

A plan of the [optimization layer](optimize.md) is the closed loop, so it has to read what the
controller on the robot reads. For a robot with fewer motors than joints, `Problem(system)`
plans the controller that reads the motors, as `VMCController` does: the frozen one, with the
passive joints at their constant rest. `Problem(system, rest="moving")` plans the frozen
controller with the moving rest, and `Problem(system, reads="state")` plans the whole-state
controller of `"naive"` instead. Here is where each plan puts the arm at rest, started near
where the matching simulation of the reach ended:

```{code-cell} python
from virtualmodelcontrol import optimization as opt

def planned_end(base, **options):
    start = runs[base]["q"][-1] + 0.05  # a guess near where it ends
    problem = opt.Problem(compiled.system, **options)
    problem.add(opt.Equilibrium(start))
    q = problem.solve().q[0]
    return 1000 * np.linalg.norm(kin.position(q, "tip")[:2] - [0.45, 0.15])

planned = {"naive": planned_end("naive", reads="state"),
           "frozen": planned_end("frozen"),
           "frozen, moving rest": planned_end("frozen, moving rest",
                                              rest="moving")}
```

```{code-cell} python
:tags: [remove-cell]
for name, mm in planned.items():
    assert abs(mm - 1000 * final[name]) < 0.1, (name, mm)
glue("planned_naive", float(planned["naive"]), display=False)
glue("planned_frozen", float(planned["frozen"]), display=False)
glue("planned_moving", float(planned["frozen, moving rest"]), display=False)
```

The plans end {glue:text}`planned_naive:.1f` mm, {glue:text}`planned_frozen:.1f` mm and
{glue:text}`planned_moving:.1f` mm from the goal, where the simulations above end. The frozen
controller's constant rest is the `q_rest` Param of the actuation, so a plan can optimize it with
`problem.free("arm.q_rest")` and a cost on the tip. Started where the constant rest at zero ends,
the plan finds the rest that balances the passive joint where the arm should end:

```{code-cell} python
goal = [0.45, 0.15, 0.0]
problem = opt.Problem(compiled.system)
problem.add(opt.Equilibrium(runs["frozen"]["q"][-1] + 0.05))
problem.add(opt.Cost(arm.point("tip") - goal, 1.0, name="reach"))
problem.free("arm.q_rest")
best = problem.solve()
tip = kin.position(best.q[0], "tip")[:2]
best_error = 1000 * np.linalg.norm(tip - goal[:2])
best_rest = float(np.ravel(best.params["arm.q_rest"])[1])
```

```{code-cell} python
:tags: [remove-cell]
assert best_error < 0.01 and best_rest < -0.1, (best_error, best_rest)
glue("best_error", float(best_error), display=False)
glue("best_rest", best_rest, display=False)
```

The plan ends {glue:text}`best_error:.1e` mm from the goal, with the passive joint's rest at
{glue:text}`best_rest:.2f` rad.

The passive correction can be in the plan too: `Problem(system, output=[stage])` with
`stage = ua.Passivation(dynamics)`, the same stage that the controller runs with, so that the plan
is the closed loop of the corrected controller. It reads the plant's motor rates and the power
its own dampers take from the plan's state. The tank correction cannot: the level of the tank is
a state that a plan does not carry (`TankBudget` is the tank of the plan's own changes of the
Params). To see that the plan has the correction, we simulate a corrected, underdamped arm
(10 ms control period), start the plan from that simulation and let the solver stop there. What it
reports is the largest violation of the plan's dynamics at the simulation: nothing, if the plan is
the simulation's closed loop.

```{code-cell} python
# underdamped, so that the arm gains energy
light = reach(arm, stiffness=150.0, damping=3.0)
stage = ua.Passivation(dynamics, width=1e-2)
corrected = ua.controller(light, "frozen", "passive", width=1e-2)
T, nodes, sub = 0.5, 6, 10
h = T / (nodes - 1) / sub  # [s] the control period

plant = vmc.sim.ModelPlant(arm, q0=[0.3, 0.3, 0.3], max_step=h)
clock = vmc.sim.SimClock(h)
log = vmc.sim.run(plant, corrected, clock, T=T + 2 * h).arrays()
start = {"q": log["q"][::sub][:nodes], "v": log["v"][::sub][:nodes]}

def violation(output):
    stop = {"ipopt.max_iter": 0}  # the solver stops at the start
    problem = opt.Problem(light.system, output=output, options=stop)
    problem.add(opt.Shooting([0.3, 0.3, 0.3], T, nodes, substeps=sub))
    return problem.solve(warm_start=start).violation

gaps = {"with the stage": violation([stage]), "without": violation([])}
```

```{code-cell} python
:tags: [remove-cell]
assert gaps["with the stage"] < 1e-4 < 1e-2 < gaps["without"], gaps
glue("gap_with", float(gaps["with the stage"]), display=False)
glue("gap_without", float(gaps["without"]), display=False)
```

With the stage the plan's dynamics are violated by {glue:text}`gap_with:.0e` at the simulation of
the corrected controller, which is rounding; without it by {glue:text}`gap_without:.2f`, because
that plan is the closed loop of another controller.

## When the passive joint is not measured

The naive controller needs the angle and the rate of every joint, and the encoders give the
two motors' only. A Kalman filter on the arm's own model can estimate the rest: the passive
joint moves the other two through the coupling, and the filter reads it from them.
`encoder` knows that this arm has fewer motors than joints. It reports what the motors
see, $\theta = B^\top q$, as combinations of $q$ (the measurement's `matrix`), not as a reading
of $q$, so the passive joint is left to the model. Each period the loop updates the filter with
the encoders, hands the estimate to the controller as the signals `q` and `v`, and predicts
the next period from the torques it sent. A `MovingHorizon` estimator takes the same `encoder`.

```{code-cell} python
from virtualmodelcontrol.estimation import Encoders, KalmanFilter

def estimated_run(stiffness, Q, T=3.0, dt=5e-4, estimate=True):
    compiled = reach(arm, stiffness=stiffness)
    onboard = ua.controller(compiled, "naive")
    plant = vmc.sim.ModelPlant(arm, q0=[0.5, -0.3, 0.8], max_step=1e-4)
    encoders = Encoders(arm, noise=1e-4, rate_noise=1e-3)
    kf = KalmanFilter(compiled.system, dt, Q=Q)
    kf.reset([0.5, -0.1, 0.8])  # the passive joint starts 0.2 rad off
    onboard.reset(0.0, plant.read())
    rows = []
    try:
        for _ in range(round(T / dt)):
            theta, rate = encoders.read(plant.q, plant.v)
            kf.update([kf.encoder(theta, rate, 1e-8, 1e-6)])  # variances
            seen = vmc.Signals(plant.t, motor_position=theta,
                               motor_velocity=rate, q=kf.q, v=kf.v)
            if not estimate:  # the controller reads the true state instead
                seen = plant.read()
            cmd = onboard.step(plant.t, seen)
            plant.write(cmd)
            plant.advance(dt)
            kf.predict(cmd["motor_torque"], plant.t)
            rows.append([plant.t, *plant.q, *kf.q])
    except np.linalg.LinAlgError:  # the covariance is no longer a number
        pass
    return np.array(rows), kf.rejected_total
```

Process noise `Q` says how far the filter trusts its model over one period. We run the arm,
with a spring of 30 N/m to the goal, from near the goal and with the passive joint's estimate
0.2 rad off, at the default $Q = 10^{-6}$ and at $Q = 10^{-3}$:

```{code-cell} python
estimates = {"Q = 1e-6": estimated_run(30.0, 1e-6),
             "Q = 1e-3": estimated_run(30.0, 1e-3)}

def tip_gap(rows):  # [m]
    tips = np.array([kin.position(q, "tip")[:2] for q in rows[::20, 1:4]])
    return np.linalg.norm(tips - [0.45, 0.15], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (name, (rows, _)) in zip(axes, estimates.items()):
    ax.plot(rows[:, 0], rows[:, 2], label="passive joint")
    ax.plot(rows[:, 0], rows[:, 5], "--", label="estimate")
    ax.set_title(name)
    ax.set_xlabel("time [s]")
axes[0].set_ylabel("angle [rad]")
axes[1].legend(fontsize=18);
```

```{code-cell} python
:tags: [remove-cell]
tuned, tuned_lost = estimates["Q = 1e-3"]
loose, loose_lost = estimates["Q = 1e-6"]
stiff, stiff_lost = estimated_run(150.0, 1e-3)
gap = np.abs(tuned[:, 2] - tuned[:, 5])
after = tuned[:, 0] > 0.02  # the first 20 ms remove the 0.2 rad of the start
settled = float(tuned[np.flatnonzero(gap >= 5e-3)[-1] + 1, 0])
true_state = estimated_run(30.0, 1e-3, estimate=False)[0]
assert abs(tip_gap(tuned)[-1] - tip_gap(true_state)[-1]) < 2e-3  # as with the true state
assert settled < 2.0 and gap[after].max() < 0.2
assert gap[tuned[:, 0] >= 0.02][0] < 0.05  # the 0.2 rad of the start is gone
assert len(tuned) == 6000 and np.isfinite(tuned).all() and tuned_lost == 0
assert tip_gap(tuned)[-1] < 0.015 and gap[-2000:].max() < 5e-3
assert loose_lost > 0.9 * len(loose) > 1000  # all the readings after the first ones
assert stiff_lost > 100 and tip_gap(stiff)[-1] > 0.3
glue("e_tip", 1000 * float(tip_gap(tuned)[-1]), display=False)
glue("e_gap", 1000 * float(gap[-2000:].max()), display=False)
glue("e_peak", 1000 * float(gap[after].max()), display=False)
glue("e_settle", settled, display=False)
glue("e_lost", int(stiff_lost), display=False)
glue("e_loose", int(loose_lost), display=False)
glue("e_loose_steps", len(loose), display=False)
```

With $Q = 10^{-3}$ the estimate reaches the passive joint in 20 ms, then stays within
{glue:text}`e_peak:.0f` mrad of it through the first swing, within 5 mrad from
{glue:text}`e_settle:.1f` s on, and within {glue:text}`e_gap:.1f` mrad over the last second. The tip
ends {glue:text}`e_tip:.1f` mm from the goal, which is where the controller on the true state
ends (the gravity compensation is not exact on the passive joint). With the default $Q$ the filter
believes its model to the last digit. In the first swing the model is wrong by more than that,
the gate throws the readings out, and it goes on throwing them out: {glue:text}`e_loose` of the
{glue:text}`e_loose_steps` it saw. The estimate then runs on the model alone, and its covariance
grows until the filter breaks. A count of rejected readings that rises at every
step (`kf.rejected_total`) means the filter has lost its sensors: raise `Q`, widen the `gate`
of `KalmanFilter` (40 by default) or pass `gate=None`.

The estimate costs gain. With the spring at 150 N/m instead of the 30 N/m of these runs, the
same filter loses the encoders in the first swing ({glue:text}`e_lost` readings thrown out)
and the arm runs away from the goal: a stiff controller moves the arm faster than the filter
follows. Soften the controller, or use `"frozen"`, which needs the motors only.

## Take it to the robot

On the robot the same law runs in the loop that talks to the motors, with nothing from
`vmc.sim`. The loop builds the controller once, and each period it passes the reading to
`step` and sends back the torques. The frozen controller needs only the motors' angles and
rates; the corrections also read $q$ and $v$, from a plant that reports them or from an
estimate of the unmeasured joints (see the section above). Here is one period with a
made-up reading:

```{code-cell} python
compiled = reach(arm)
onboard = ua.controller(compiled, "frozen", "tank", tank=0.5)  # [J]
reading = vmc.Signals(
    0.0, motor_position=[0.3, 0.3], motor_velocity=[0.0, 0.0],
    q=[0.3, 0.3, 0.3], v=[0.0, 0.0, 0.0])
onboard.reset(0.0, reading)
onboard.step(0.0, reading)["motor_torque"]  # [N·m]
```

```{code-cell} python
:tags: [remove-cell]
torque = onboard.step(0.01, reading)["motor_torque"]
assert torque.shape == (2,) and np.isfinite(torque).all()
```

## Choosing

- Start with `"frozen"`: it needs only the motors, its torque is a gradient, and it agrees with
  the naive one where the passive joints rest. If the arm carries weight, give it the rest the
  arm really has: `at=` the motor angles of the pose it works around, a plan that frees
  `q_rest`, or `rest="moving"`, which follows the balance and needs the robot's model.
- Take `"naive"` when the whole state is measured or estimated well and you want the force the
  spring asks for even away from the passive joints' rest.
- Add `"passive"` to the naive controller when the arm must never gain energy, and `"tank"` to
  allow short bursts up to a budget. A correction removes torque, so it can slow the reach: set
  `tank` to what the task needs and `width` to the smallest power that matters. They also cut
  the frozen torque, which needs no correction.
- The laws and estimators that read a controller's motors (`ForceTracking`, `StiffnessTracking`,
  `PositionRegulation`, `HoldingGoals`, `ContactForce`, `TaskStiffness`) refuse a
  `StateController`, which is what `"naive"` and `"frozen"` with `at=` or `rest="moving"` give,
  with a `ValueError`. Plain `"frozen"` is a `VMCController` and works with them; for a force on
  an underactuated arm use `DirectionalForce`.
- The weight of the arm is compensated by a `GravityCompensation` in the controller, as for any
  robot; the rest only says where the passive joints are taken to be.
- `planar.arm(rest=...)` sets the passive joints' rest angles, one for all, one per passive
  joint or one per joint. The frozen controller reads them as the `q_rest` of the actuation,
  and `add_dynamics` copies them into the passive springs' rest: the two do not follow each
  other after that.
