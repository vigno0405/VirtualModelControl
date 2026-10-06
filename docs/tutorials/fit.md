---
file_format: mystnb
kernelspec:
  name: python3
---

# Fit Params to a run

In this tutorial we find the mass, the stiffness and the damping of a robot from a log of its
motion, as one does when a model has to match a real robot.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The run

The robot is a mass on a spring and a damper that slides along $x$. A controller moves it with a
virtual spring to a goal that wanders, which gives motions at several speeds, and we log
the run. This robot is the truth, and the fit does not see its numbers:

```{code-cell} python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc


def slider(mass, stiffness, damping):
    robot = vmc.Mechanism("slider", model=vmc.models.JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("m", vmc.Inertance(x, vmc.Param("mass", mass, unit="kg", bounds=(0, np.inf))))
    robot.add("spring", vmc.LinearSpring(x, stiffness))  # [N/m]
    robot.add("damper", vmc.LinearDamper(x, damping))  # [N·s/m]
    return robot

truth = slider(2.0, 40.0, 1.5)
goal = vmc.Custom(lambda t: 0.1 * ca.sin(2 * t) + 0.05 * ca.sin(9 * t),
                  [vmc.Time()], dim=1, unit="m")
ctrl = vmc.Mechanism("ctrl")
ctrl.add("pull", vmc.LinearSpring(truth.joint(0) - goal, 60.0))
ctrl.add("damp", vmc.LinearDamper(truth.joint(0), 3.0))
controller = vmc.VMCController(
    vmc.compile(vmc.VirtualMechanismSystem(truth, ctrl)))
plant = vmc.sim.ModelPlant(truth, max_step=1e-3)
log = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 200), T=10.0)
```

## The fit

We start from a model that has the right structure and the wrong numbers, and name the Params
to find. `fit_params` takes the robot, the names (or globs) of its Params and the runs. It asks
which values make the robot's own dynamics, at the logged motion, agree with the torques that
were sent:

```{code-cell} python
from virtualmodelcontrol.identification import fit_params

guess = slider(1.0, 10.0, 0.2)
names = ["m.inertance", "spring.stiffness", "damper.damping"]
fit = fit_params(guess, names, [log], smoothing=11)
for name in names:
    print(f"{name:18s} {float(fit.values[name]):8.4f} ± {float(fit.std[name]):.4f}")
```

```{code-cell} python
:tags: [remove-cell]
from myst_nb import glue

for name, true in zip(names, (2.0, 40.0, 1.5)):
    assert abs(float(fit.values[name]) - true) < 0.05 * true
glue("fit_rms", fit.rms, display=False)
```

The true values are 2 kg, 40 N/m and 1.5 N·s/m. The log holds no accelerations, so the fit
gets them by smoothing the velocity and differentiating it, which is where its error comes from
(`smoothing` is the number of samples of the smoothing). A run that has an `a` of its own, as a
simulation can give, is used as it is. After the fit, {glue:text}`fit_rms:.2f` N is left over
per sample, and the numbers after ± are the standard errors, which say how far the values may be
off if that is noise.

## What it asks of the data

A Param shows only in motion that depends on it. The mass shows in the accelerations, the
stiffness in the position, the damping in the velocity, so a run that only sits still fits
none of them. The standard errors tell: a Param that the run barely excites has a large one.
A Param of a nonlinear component, such as the saturating damper `TanhDamper`, is fitted the
same way, by iterating from the values the robot has, and the fit keeps every Param within its
bounds.
