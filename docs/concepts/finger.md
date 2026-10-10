---
file_format: mystnb
kernelspec:
  name: python3
---

# Finger and hand kinematics

The ADAPT finger and hand are rigid chains of phalanges driven through tendons and cables, with
fewer motors than joints. This page gives the equations of `robots.adapt`: how motor angles set
joint angles, and how joint angles place the fingertip, and then how the hand's motors drive its joints.

```{code-cell} python
:tags: [remove-cell]
import docs_setup
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np
from myst_nb import glue
from virtualmodelcontrol.robots import adapt

a, b, c = adapt.LINK_LENGTHS
glue("a", 1000 * a, display=False)
glue("b", 1000 * b, display=False)
glue("c", 1000 * c, display=False)
glue("mcp", float(adapt.COUPLING[0, 0]), display=False)
glue("pip", float(adapt.COUPLING[1, 1]), display=False)
```

## The finger

The finger has three phalanges, proximal, middle and distal, of {glue:text}`a:.1f`,
{glue:text}`b:.1f` and {glue:text}`c:.1f` mm, joined by three revolute joints, MCP, PIP and
DIP (the metacarpophalangeal, proximal interphalangeal and distal interphalangeal joints), all
turning about $x$. Straight, it points along $y$. Bending it turns the tip towards $z$.

```{code-cell} python
:tags: [remove-input]
from schematics import finger as schematic
schematic.figure();
```

### Motors to joints

Two motors drive the three joints. The first pulls a cable on a pulley of the MCP joint. The
second drives the PIP joint through a cable, and the DIP joint follows the PIP joint. With
motor angles $q_0, q_1$,

$$
\theta_\text{MCP} = \frac{r_p}{r_m}\,q_0, \qquad
\theta_\text{PIP} = \theta_\text{DIP} = \frac{r_m}{c_p}\,q_1,
$$

with $r_m$ the motor pulley's radius, $r_p$ that of the MCP pulley and $c_p$ the cable constant
of the PIP drive. For the template, $r_p/r_m$ and $r_m/c_p$ are {glue:text}`mcp:.3f` and
{glue:text}`pip:.3f` rad per radian of motor angle. In matrix form $\theta = C\,q$, a
`LinearCoupling` of the chain.

### Joints to the fingertip

Each joint turns everything beyond it about $x$, so with
$\phi_1 = \theta_\text{MCP}$, $\phi_2 = \phi_1 + \theta_\text{PIP}$ and
$\phi_3 = \phi_2 + \theta_\text{DIP}$ the fingertip is at

$$
p = \begin{bmatrix}
0 \\
a\cos\phi_1 + b\cos\phi_2 + c\cos\phi_3 \\
a\sin\phi_1 + b\sin\phi_2 + c\sin\phi_3
\end{bmatrix},
$$

in the finger's base frame. Its Jacobian with respect to the motor angles is
$J = (\partial p/\partial\theta)\,C$, which the library computes by automatic differentiation.

```{code-cell} python
:tags: [remove-cell]
import virtualmodelcontrol as vmc

kin = vmc.Kinematics(adapt.finger())
for q in np.random.default_rng(0).uniform(-1.0, 1.0, (5, 2)):
    phi = np.cumsum(adapt.COUPLING @ q)
    p = [0.0, a * np.cos(phi[0]) + b * np.cos(phi[1]) + c * np.cos(phi[2]),
         a * np.sin(phi[0]) + b * np.sin(phi[1]) + c * np.sin(phi[2])]
    assert np.allclose(kin.position(q, "tip"), p, atol=1e-12)
```

## The hand

```{code-cell} python
:tags: [remove-cell]
glue("motors", len(adapt.HAND_MOTORS), display=False)
glue("joints", adapt.hand_coupling().shape[0], display=False)
C = adapt.hand_coupling()
spread = adapt.HAND_MOTORS.index("spread")
rows = {f: 4 + 4 * k for k, f in enumerate(("index", "middle", "ring", "pinky"))}
coefficients = [f"{C[rows[f], spread]:+.3f}" if C[rows[f], spread] else "0" for f in rows]
glue("spread", ", ".join(coefficients[:3]) + " and " + coefficients[3], display=False)
```

The hand has a thumb and four fingers, each a chain of revolute joints from the hand's base.
The thumb has the joints CMC1, CMC2 (carpometacarpal), MCP and IP (interphalangeal), and each
finger has the joints spread, MCP, PIP and DIP. The origins and axes of the joints are tables in `robots.adapt` (`HAND_JOINTS`, `HAND_FINGER_BASES`).
{glue:text}`motors:.0f` motors drive its {glue:text}`joints:.0f` joints, $\theta = C\,q$:

- each thumb joint has a motor of its own;
- each finger's MCP joint has a motor, and one more drives its PIP and DIP joints together, as
  on the finger;
- one motor spreads the fingers, turning the index, middle, ring and pinky spread joints by
  {glue:text}`spread` rad per radian of its angle: the middle finger stays in place.

```{code-cell} python
:tags: [remove-input]
from schematics import hand as hand_schematic
hand_schematic.figure();
```

The joint-limit springs keep each joint within the ranges in `adapt.JOINT_LIMITS` and
`adapt.HAND_JOINT_LIMITS`.
