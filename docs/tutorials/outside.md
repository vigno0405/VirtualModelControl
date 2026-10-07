---
file_format: mystnb
kernelspec:
  name: python3
---

# Use a model outside CasADi

Every model of the library is a CasADi function: the kinematics of a robot, its dynamics, a
controller's law. CasADi may not be where the model has to run: a computer that only has numpy,
a network that is trained through a model, a colleague's code. In this tutorial we write models
as plain Python, for numpy or for PyTorch, from the same expression graph, so that the numbers
are the same ones.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
from myst_nb import glue
```

## A model as numpy code

`Kinematics.functions` takes a `backend`. With `"numpy"` it returns a Python function in place
of the CasADi one: the same arguments, the same outputs, here the position, rotation, Jacobians
and Hessian of the tip of the soft arm:

```{code-cell} python
import numpy as np
import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

arm = helyx.add_dynamics(helyx.arm("145-290-290"))
kin = vmc.Kinematics(arm)
tip = kin.functions(1.0, backend="numpy")  # the end of the arm, s = 1
reference = kin.functions(1.0)  # CasADi's

q = np.random.default_rng(0).normal(size=9) * 0.004
position, rotation, J, Jw, H = tip(q)
print(position, J.shape, H.shape)
```

```{code-cell} python
:tags: [remove-cell]
gap = max(np.abs(np.asarray(a) - np.array(b).reshape(np.shape(a))).max()
          for a, b in zip(tip(q), reference(q), strict=True))
lines = tip.source.count("\n")
assert gap < 1e-14 and lines > 1000 and "casadi" not in tip.source.lower().split("def ")[1]
glue("bound", 1e-14, display=False)
glue("lines", lines, display=False)
```

```{code-cell} python
:tags: [remove-cell]
import time


def per_call(fn, arg, repeat=100):  # [s]
    fn(arg)
    start = time.perf_counter()
    for _ in range(repeat):
        fn(arg)
    return (time.perf_counter() - start) / repeat


casadi_call, numpy_call = per_call(reference, q), per_call(tip, q)
batch = next(n for n in (10, 30, 100, 300, 1000, 3000)
             if per_call(tip, np.tile(q, (n, 1)), 10) / n < casadi_call)
assert numpy_call > 2 * casadi_call  # for one input CasADi is faster
glue("slow", numpy_call / casadi_call, display=False)
glue("batch", batch, display=False)
```

The code agrees with CasADi's own result to better than {glue:text}`bound:.0e`: it does the
same operations in the same order. The function is generated: `tip.source` is its text, a
straight line of {glue:text}`lines` statements, one for each operation of the expression graph,
that imports numpy and nothing else.

```{code-cell} python
body = tip.source.split("def kinematics")[1].splitlines()
print("def kinematics" + "\n".join(body[:14]))
```

Arguments follow CasADi's shapes: a number, a vector of shape `(n,)` or a matrix of shape
`(n, m)`. Any leading dimensions are a batch: a thousand configurations at once is one call,
and `tip(Q)` with `Q` of shape `(1000, 9)` returns arrays with 1000 as their first dimension.

## Take the code away

The text is all a machine needs. We write it to a file and run it in a Python that cannot
import CasADi, which we make sure of by blocking the import:

```{code-cell} python
import pathlib
import subprocess
import sys
import tempfile

folder = pathlib.Path(tempfile.mkdtemp())
(folder / "tip.py").write_text(tip.source)
np.save(folder / "q.npy", q)
program = f"""
import sys
sys.modules["casadi"] = None  # any import of casadi fails
import numpy as np
sys.path.insert(0, {str(folder)!r})
import tip
print(repr(tip.kinematics(np.load({str(folder / "q.npy")!r}))[0].tolist()))
"""
run = subprocess.run([sys.executable, "-I", "-c", program],
                     capture_output=True, text=True)
print(np.array(eval(run.stdout)), run.stderr[-200:])
```

```{code-cell} python
:tags: [remove-cell]
assert run.returncode == 0, run.stderr
assert np.allclose(eval(run.stdout), position, rtol=0, atol=1e-15)
```

## A controller as numpy code

A compiled controller exports the same way, with `compiled.export("numpy")`. On a robot the
loop that talks to the motors needs `fast`: the motor angles and rates, the virtual state, the
live Params and the time, in one vector, to the motor torques and the rate of the virtual
state. Here is the three-link arm of [the underactuated tutorial](underactuated.md) held at
a goal by a spring and a damper on its tip, one step of its law in numpy, against the library's
own controller:

```{code-cell} python
from virtualmodelcontrol.control import underactuated as ua
from virtualmodelcontrol.robots import planar

link = planar.add_dynamics(planar.arm("three-link"))
ctrl = vmc.Mechanism("ctrl")
end = link.point("tip")
ctrl.add("reach", vmc.LinearSpring(end - [0.45, 0.15, 0.0], 150.0))
ctrl.add("damp", vmc.LinearDamper(end, 25.0))
ctrl.add("gravity", vmc.GravityCompensation(link))
compiled = vmc.compile(vmc.VirtualMechanismSystem(link, ctrl))
law = compiled.export("numpy").fast  # no CasADi from here on

theta, rate, t = np.array([0.3, 0.5]), np.array([0.1, -0.2]), 0.0
# the state, the live Params and the time (no virtual state here)
x = np.concatenate([theta, rate, compiled.live_values(), [t]])
u = law(x)[: compiled.n_u]  # [N·m]

frozen = ua.controller(compiled, "frozen")
reading = vmc.Signals(t, motor_position=theta, motor_velocity=rate)
frozen.reset(t, reading)
print(u, frozen.step(t, reading)["law_torque"])
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(u, frozen.step(0.01, reading)["law_torque"], rtol=1e-12, atol=1e-12)
assert np.abs(u).max() > 0.1
```

The two agree. The numpy function is what such a loop needs, `law(x)` once a period, with nothing
of the library but its text.

## PyTorch

With `backend="torch"` the same text is written for torch tensors. The function is made of
torch operations, so autograd differentiates it, and a batch of states is one call. This needs
PyTorch (`pip install virtualmodelcontrol[torch]`); the code below runs as it is where PyTorch is
installed, with the soft arm's dynamics:

```python
import torch

dynamics = vmc.compile_dynamics(arm)
model = dynamics.export("torch")  # forward, residual, mass, ...
p = torch.as_tensor(dynamics.live_values())
q = torch.zeros(64, 9, dtype=torch.float64, requires_grad=True)  # a batch
v = torch.zeros_like(q)
u = torch.zeros_like(q)
acceleration = model.forward(q, v, u, p, 0.0)  # shape (64, 9)
acceleration.sum().backward()  # d(acceleration)/dq, from the same graph
```

The tests compare these values and derivatives with CasADi's (the Jacobian and the Hessian of
the kinematics included) to $10^{-12}$ of the largest entry. The soft arm's dynamics agree to
no more than a few parts in $10^{11}$, depending on the build of PyTorch: PyTorch's `sin` and `cos` differ from the C library's in the
last place, and the arm's mass matrix has a condition number of a few million, which
multiplies the difference.

## Good to know

- **What can be written.** Anything made of CasADi's scalar operations: the models and
  controllers of the library are. A function that calls code outside CasADi (a callback, an
  external library) cannot, and an operation without a translation says its name.
- **Speed.** The code is a long straight line of Python. For one input CasADi's compiled
  evaluation is much faster ({glue:text}`slow:.0f` times for the arm's kinematics, measured
  here), and numpy is faster per input from a batch of {glue:text}`batch`. Take the code away for portability, not for speed.
- **Where the Params went.** The kinematics are written at the Params' values when you ask
  (`Kinematics` folds them in); the dynamics and the controllers take the live Params as an
  argument `p`, as their CasADi functions do. Ask again after changing a design Param.
- **dtype.** numpy computes in float64. The torch code follows the dtype and the device of
  its arguments, and makes whole numbers floats.
- **Branches.** `if_else` becomes `where`: both branches are computed. A branch that is not
  a number where it is not taken gives PyTorch a `nan` gradient. Keep the unused branch
  finite, as for any torch function.
