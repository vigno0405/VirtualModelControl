"""A transmission efficiency fitted to static measurements, by least squares through the origin."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..models.efficiency import Efficiency


def fit_efficiency(
    commanded: ArrayLike,
    measured: ArrayLike,
    degree: int = 1,
    *,
    weights: ArrayLike | None = None,
    shared: bool = False,
) -> Efficiency:
    """The polynomial efficiency, of ``degree``, that best explains static measurements.

    ``commanded`` [N·m] has one row per static point and one column per motor. Without
    ``weights``, ``measured`` holds each motor's delivered torque, laid out the same way. With
    ``weights`` (same layout), it holds one value per point, Σⱼ wⱼ τⱼ: a force along a direction,
    say, with wⱼ from the Jacobian. Each motor gets its own coefficients, or one set for all with
    ``shared``. No constant term: no command, no torque.
    """
    u = np.asarray(commanded, dtype=float)
    u = u.reshape(len(u), -1)
    powers = np.stack([u**k for k in range(1, degree + 1)], axis=-1)  # point, motor, power
    if weights is None:
        tau = np.asarray(measured, dtype=float).reshape(u.shape)
        if shared:
            return Efficiency(*_least_squares(powers.reshape(-1, degree), tau.ravel()))
        c = np.array([_least_squares(powers[:, j], tau[:, j]) for j in range(u.shape[1])])
        return Efficiency(*c.T)
    w = np.asarray(weights, dtype=float).reshape(u.shape)
    y = np.asarray(measured, dtype=float).ravel()
    phi = w[:, :, None] * powers
    if shared:
        return Efficiency(*_least_squares(phi.sum(axis=1), y))
    return Efficiency(*_least_squares(phi.reshape(len(u), -1), y).reshape(u.shape[1], degree).T)


def _least_squares(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]
