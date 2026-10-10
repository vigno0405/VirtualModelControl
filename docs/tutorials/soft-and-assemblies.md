---
file_format: mystnb
kernelspec:
  name: python3
---

# Soft arms and assemblies

In this tutorial we build a continuum arm with tendons, spread its mass along it, and mount parts on
each other as an assembly. Then we take the soft arm's controller to the robot.

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

## Other kinds of robots

A continuum arm is a `PCC` model, driven by a `TendonTransmission`:

```{code-cell} python
from virtualmodelcontrol.models import PCC, TendonTransmission

spacing = np.radians([[0, 120, -120], [60, 180, -60]])  # per segment
soft = vmc.Mechanism("soft", model=PCC([0.2, 0.2], 0.03),
                     actuation=TendonTransmission(spacing, 0.003))
soft.actuation.motor_sizes(soft.space)  # motor angles, motor rates
```

The arm's mass is not lumped by itself. `soft.add_mass_along(name, mass, s0, s1, n)` spreads a mass
evenly between two arc parameters as $n$ point masses at the nodes of Gauss-Legendre quadrature,
which is exact for any polynomial of degree $2n - 1$ in the arc length. Here the height of the
center of the mass of a rod bent through 1.2 rad in all converges with $n$ to rounding error:

```{code-cell} python
q_bent = np.array([0.018, 0, 0, 0.018, 0, 0])  # 0.6 rad in each segment
kin_soft = vmc.Kinematics(soft)

def moment(n):
    """The mass times the height of its center [kg·m], with n masses."""
    rod = vmc.Mechanism("rod", model=soft.model)
    names = rod.add_mass_along("rod", 1.0, 0.0, 1.0, n)
    parts = [rod.components[k] for k in names]

    def height(c):
        return kin_soft.position(q_bent, float(c.coord.s.value))[2]

    return sum(float(c.mass.value) * height(c) for c in parts)

counts = np.arange(1, 9)
fine = moment(64)
mistake = [abs(moment(n) - fine) / fine for n in counts]
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

assert mistake[0] > 1e-2 and mistake[3] < 1e-8
glue("one_mass", float(100 * mistake[0]), display=False)
glue("four_masses", float(mistake[3]), display=False)
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots()
ax.semilogy(counts, np.maximum(mistake, 1e-16), "o-")
ax.set_xlabel("masses $n$")
ax.set_ylabel("error of the center's height");
```

One mass is {glue:text}`one_mass:.0f` % off, four masses {glue:text}`four_masses:.0e`.

A robot known only by its joints, such as the turtle's cranks, is a `JointSpace` model: its
controllers act on `robot.joint(i)`. A `LinearCoupling` lets one motor drive several joints.
An `Assembly` mounts parts on a base or on another part's frame, as the hand on the UR5's
flange:

```{code-cell} python
from virtualmodelcontrol.models import Assembly, LinearCoupling

gripper = LinearCoupling(SerialChain(
    ["revolute", "revolute"], axes=[[0, 0, 1]] * 2,
    points=[[0, 0, 0], [0.05, 0, 0]], sites={"tip": (2, [0.09, 0, 0])},
), [[1.0], [0.5]])  # one motor; the second joint turns half as much
two_link = SerialChain(
    ["revolute", "revolute"], axes=[[0, 0, 1]] * 2,
    points=[[0, 0, 0], [0.30, 0, 0]], sites={"tip": (2, [0.55, 0, 0])})
both = Assembly({
    "arm": (two_link, (0, 0, 0), (0, 0, 0)),
    "gripper": (gripper, (0, 0, 0), (0, 0, 0), "arm/tip"),  # on the tip
})
vmc.Kinematics(both).position([0.0, 0.0, 0.3], "gripper/tip")  # [m]
```

The motors of an assembly are those of its parts, stacked in order. A part that nothing drives,
such as a floating body, gets `models.Passive(nv)` in `assembly.stacked_actuation` (a dict of
the actuations by part) and adds no motors: [Crawl with a flywheel](crawl.md) builds a crawler
that way.

To write a new kind of model, see [Extend the library](extend.md).

## Take the soft arm to the robot

The controller of a robot needs nothing from `vmc.sim`. Here a spring takes the tip of the soft arm
above to a goal; one control period of your control loop is a `step` with the motors' reading, which
for this arm is the tendons' angles and rates:

```{code-cell} python
ctrl = vmc.Mechanism("ctrl")
tip = soft.point(s=1.0)
ctrl.add("reach", vmc.LinearSpring(tip - [0.05, 0.0, 0.3], 100.0))
ctrl.add("damp", vmc.LinearDamper(tip, 2.0))
system = vmc.VirtualMechanismSystem(soft, ctrl)
controller = vmc.VMCController(vmc.compile(system))

n_angles, n_rates = soft.actuation.motor_sizes(soft.space)
reading = vmc.Signals(0.0, motor_position=np.zeros(n_angles),
                      motor_velocity=np.zeros(n_rates))
controller.reset(0.0, reading)
torques = controller.step(0.0, reading)["motor_torque"]  # [N·m]
```
