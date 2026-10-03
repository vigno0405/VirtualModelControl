---
file_format: mystnb
kernelspec:
  name: python3
---

# Helyx soft arm

Three tendon-driven PCC segments, with three motors per segment. `robots.helyx.arm(geometry)`
returns the arm as a robot mechanism; every default below is a `Param` you can change.

| Geometry | Segment rest lengths, base to tip | Mounting (gravity in the base frame) |
|---|---|---|
| `145-145-145` | 145, 145, 145 mm | side-mounted, g along −y |
| `145-290-290` | 145, 290, 290 mm | hanging, g along +z |
| `290-145-145` | 290, 145, 145 mm | pointing up, g along −z |

Common to all: section radius 30 mm, spool radius 3 mm, tendon angles at each segment's base
[0°, 120°, −120°], [60°, 180°, −60°], [150°, 270°, 30°], and 40 g per 145 mm of segment lumped at
each segment's midpoint.

## Motor signs

The library uses θ > 0 to pull a tendon. `helyx.ENCODER_SIGN` gives each arm's encoder sign
against that convention; multiply raw motor angles and torques by it at the hardware boundary.

```{code-cell} python
from virtualmodelcontrol.robots import helyx
helyx.ENCODER_SIGN
```

## Shape at a pose

```{code-cell} python
import numpy as np
from virtualmodelcontrol.models import evaluate_frame

arm = helyx.arm("145-290-290")
delta = np.array([0.01, 0.0, 0.0] * 3)   # bend every segment towards +x
for s in (0.2, 0.6, 1.0):
    print(s, evaluate_frame(arm.model, delta, s)[1].round(4))
```
