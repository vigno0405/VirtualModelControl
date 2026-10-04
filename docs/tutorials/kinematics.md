---
file_format: mystnb
kernelspec:
  name: python3
---

# Kinematics on the UR5

In this tutorial we compute where a robot's points are and how they move: the position,
rotation, Jacobians and Hessian of the UR5's tool, each checked against finite differences, and
the joint stiffness that a spring at the tool produces.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

## The arm

The UR5 template is built from the arm's Denavit–Hartenberg table, with the flange as the site
`tool`. Its six joints are position-controlled on the real arm, so here we use its kinematics
only.

```{code-cell} python
:tags: [remove-input]
from schematics import ur5 as schematic
schematic.figure();
```

```{code-cell} python
:tags: [remove-input]
schematic.dh_table()
```

## Positions and rotations

`vmc.Kinematics` evaluates any site of a robot at a configuration: its position [m], its
rotation, its Jacobians and its Hessian, all exact, by automatic differentiation.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import ur5

arm = ur5.arm()
kin = vmc.Kinematics(arm)
q = np.array([0.3, -1.1, 1.4, -1.6, -1.4, 0.2])  # [rad]
kin.position(q, "tool"), kin.rotation(q, "tool").round(3)
```

## Jacobians

The Jacobian $J$ maps joint velocities to the tool's velocity, $\dot p = J \dot q$, and the
angular Jacobian $J_\omega$ to its angular velocity, $\omega = J_\omega \dot q$, in the base
frame. Central finite differences of the position and the rotation agree with both:

```{code-cell} python
h, eye = 1e-6, np.eye(6)
J, Jw = kin.jacobian(q, "tool"), kin.angular_jacobian(q, "tool")
R = kin.rotation(q, "tool")
J_fd, Jw_fd = np.zeros((3, 6)), np.zeros((3, 6))
for i in range(6):
    up, down = q + h * eye[i], q - h * eye[i]
    dp = kin.position(up, "tool") - kin.position(down, "tool")
    J_fd[:, i] = dp / (2 * h)
    dR = (kin.rotation(up, "tool") - kin.rotation(down, "tool")) / (2 * h)
    W = dR @ R.T  # the skew matrix of the angular velocity
    Jw_fd[:, i] = [W[2, 1], W[0, 2], W[1, 0]]
print("J error:", np.abs(J - J_fd).max())
print("Jw error:", np.abs(Jw - Jw_fd).max())
```

## Hessians

The Hessian $H_k = \partial^2 p_k / \partial q^2$ is the second derivative of each coordinate
of the position. It tells how the Jacobian changes, and makes a second-order prediction of the
position exact up to third order in the step:

```{code-cell} python
H = kin.hessian(q, "tool")  # (3, 6, 6)
direction = np.ones(6) / np.sqrt(6)
steps = np.logspace(-4, -1, 13)  # [rad]
first, second = [], []
for s in steps:
    dq = s * direction
    exact = kin.position(q + dq, "tool")
    linear = kin.position(q, "tool") + J @ dq
    first.append(np.linalg.norm(exact - linear))
    second.append(np.linalg.norm(exact - linear - 0.5 * H @ dq @ dq))

fig, ax = plt.subplots()
ax.loglog(steps, first, "o-", label=r"$p + J\,\delta q$")
label = r"$+ \frac{1}{2}\,\delta q^\top H\,\delta q$"
ax.loglog(steps, second, "s-", label=label)
ax.set_xlabel(r"step $\|\delta q\|$ [rad]")
ax.set_ylabel("prediction error [m]")
ax.legend();
```

The first-order error falls with the square of the step and the second-order one with its
cube, as they should.

## The stiffness a tool spring gives the joints

A spring of stiffness $K$ between the tool and a goal pushes with $f = -K (p - g)$, which the
joints feel as $\tau = J^\top f$. Its joint stiffness, $-\partial\tau/\partial q$, has two parts:
$J^\top K J$, and a term from the Hessian that appears as soon as the spring is stretched,
$-\sum_k f_k H_k$. A finite difference of $\tau$ confirms the sum:

```{code-cell} python
K = 500.0 * np.eye(3)  # [N/m]
goal = kin.position(q, "tool") + [0.05, 0.0, -0.03]  # [m]

def torque(q):
    f = -K @ (kin.position(q, "tool") - goal)
    return kin.jacobian(q, "tool").T @ f

f = -K @ (kin.position(q, "tool") - goal)
geometric = J.T @ K @ J
stretched = -np.einsum("k,kij->ij", f, H)
fd = -np.column_stack(
    [(torque(q + h * e) - torque(q - h * e)) / (2 * h) for e in eye])
print("error with both terms:", np.abs(geometric + stretched - fd).max())
print("error without the Hessian term:", np.abs(geometric - fd).max())
```

## Moving the joints

A log needs only the time and the configurations, so `viz.animate` also replays a motion that
comes from no simulation at all, here a smooth sweep of three joints:

```{code-cell} python
:tags: [remove-output]
t = np.linspace(0.0, 4.0, 401)  # [s]
sweep = np.outer(np.sin(np.pi * t / 2), [0.5, 0.4, -0.5, 0.0, 0.0, 0.0])
log = {"t": t[:, None], "q": q + sweep}
viz.animate(arm, log, "kinematics.mp4", plane="xz", trace="tool")
```

```{video} kinematics.mp4
:caption: A joint sweep of the UR5 replayed from its kinematics, with the path of the tool.
```
