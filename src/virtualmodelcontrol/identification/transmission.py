"""A cable transmission ratio fitted to joint angles set and motor angles read."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def fit_transmission(motor: ArrayLike, joint: ArrayLike) -> float:
    """The joint angle per motor angle that best explains a sweep of the joint.

    Set the joint to known angles and read the motor's: ``motor = k * joint`` is fitted by least
    squares through the origin and ``1 / k`` is returned, the entry of a coupling matrix. Both
    angles in the same unit; the ratio keeps the sign of the data.
    """
    m = np.asarray(motor, dtype=float).ravel()
    j = np.asarray(joint, dtype=float).ravel()
    return float(j @ j / (j @ m))
