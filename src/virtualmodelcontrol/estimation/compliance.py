"""Object compliance by probing: how far the tip moves for the force it adds."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def object_compliance(
    position: ArrayLike,
    force: ArrayLike,
    baseline_position: ArrayLike,
    baseline_force: ArrayLike,
) -> float:
    """The compliance C = ‖x − x₀‖ / ‖F − F₀‖ [m/N] of an object a tip presses, ``nan`` if the
    force did not change.

    ``position`` [m] and ``force`` [N] are samples (rows of 3) of the tip held on the object at a
    stiff setting, and the baseline ones are the same at a gentle setting. Each is taken as the
    median of its samples, so a few wild ones do not count.
    """

    def middle(samples: ArrayLike) -> np.ndarray:
        return np.nanmedian(np.atleast_2d(np.asarray(samples, dtype=float)), axis=0)

    pushed = np.linalg.norm(middle(force) - middle(baseline_force))
    if pushed <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(middle(position) - middle(baseline_position)) / pushed)
