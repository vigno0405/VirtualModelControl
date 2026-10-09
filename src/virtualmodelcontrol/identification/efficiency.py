"""A transmission efficiency fitted to static measurements, by least squares through the origin."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..models.efficiency import Efficiency
from ..models.friction import StaticFriction, friction_torque


def fit_efficiency(
    commanded: ArrayLike,
    measured: ArrayLike,
    degree: int = 1,
    *,
    weights: ArrayLike | None = None,
    shared: bool = False,
    friction: bool = False,
    width: float = 0.01,
) -> Efficiency:
    """The polynomial efficiency, of ``degree``, that best explains static measurements.

    ``commanded`` [N·m] has one row per static point and one column per motor. Without
    ``weights``, ``measured`` holds each motor's delivered torque, laid out the same way. With
    ``weights`` (same layout), it holds one value per point, Σⱼ wⱼ τⱼ: a force along a direction,
    say, with wⱼ from the Jacobian. Each motor gets its own coefficients, or one set for all with
    ``shared``. No constant term: no command, no torque.

    With ``friction``, the actuator's static friction is fitted too, the breakaway and the kinetic
    torque of a ``StaticFriction`` (``width`` [N·m] is its smoothing, which static points do not
    determine): the points must then span the dead band, commands from well below to well above
    the breakaway torque, in both directions. It needs each motor's delivered torque: no
    ``weights``.
    """
    u = np.asarray(commanded, dtype=float)
    u = u.reshape(len(u), -1)
    if friction:
        if weights is not None:
            raise ValueError("friction needs the delivered torque of each motor: leave out weights")
        return _with_friction(
            u, np.asarray(measured, dtype=float).reshape(u.shape), degree, shared, width
        )
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


def _with_friction(
    u: np.ndarray, tau: np.ndarray, degree: int, shared: bool, width: float
) -> Efficiency:
    """The polynomial and the friction that best explain delivered torques, motor by motor or one
    set for all. For given friction the polynomial is linear, so only the breakaway and the
    kinetic torque are searched: on a grid, then refined."""
    from scipy.optimize import minimize

    def fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        def residual(f: np.ndarray) -> tuple[float, np.ndarray]:
            net = x - friction_torque(x, abs(f[0]), abs(f[1]), width)
            powers = np.stack([net**k for k in range(1, degree + 1)], axis=1)
            c = _least_squares(powers, y)
            return float(((powers @ c - y) ** 2).sum()), c

        top = np.abs(x).max()
        grid = [(F, Fc) for F in np.linspace(0.0, 0.95 * top, 40) for Fc in np.linspace(0.0, F, 6)]
        best = min(grid, key=lambda f: residual(np.array(f))[0])
        f = minimize(
            lambda f: residual(f)[0],
            np.array(best),
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-14},
        ).x
        return residual(f)[1], abs(float(f[0])), abs(float(f[1]))

    if shared:
        c, F, Fc = fit(u.T.ravel(), tau.T.ravel())  # the points of every motor, one after another
        return Efficiency(*c, friction=StaticFriction(F, Fc, width))
    fits = [fit(u[:, j], tau[:, j]) for j in range(u.shape[1])]
    c = np.array([f[0] for f in fits])
    friction = StaticFriction([f[1] for f in fits], [f[2] for f in fits], width)
    return Efficiency(*c.T, friction=friction)
