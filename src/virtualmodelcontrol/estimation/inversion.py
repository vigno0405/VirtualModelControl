"""The configuration that puts a robot's points where a position sensor saw them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..dynamics import OPTS
from ..models.kinematics import Kinematics

DAMPING = 1e-6
"""Default damping λ of a step, in (JᵀJ + λ I) δ = -Jᵀ r."""

TOLERANCE = 1e-9
"""Default tolerance [m]: the largest error of a point's position in a converged fit."""

MAX_ITER = 50
"""Default number of steps a call may take."""

GROWTH = 10.0
"""Factor by which the damping grows after a step that does not lower the error."""

RETRIES = 8
"""How many times a step is tried again with more damping before the fit gives up."""


class Inversion:
    """The configuration q that puts the robot's points ``at`` where a position sensor saw them.

    Damped Gauss-Newton from the last result (the neutral configuration at first), for any robot:
    the lab's closed form serves a three-section PCC arm only. ``converged``: the points are within
    ``tol`` [m]. From neutral it finds the Helyx arm up to 0.6 rad per section; beyond, give ``q0``.
    """

    def __init__(
        self,
        robot: Any,
        at: Sequence[Any],
        damping: float = DAMPING,
        tol: float = TOLERANCE,
        max_iter: int = MAX_ITER,
    ) -> None:
        model = Kinematics(robot).model
        space = model.space
        if space.nq != space.nv:
            raise ValueError(
                f"the inversion needs as many velocity as configuration coordinates, "
                f"the robot's space has nq={space.nq} and nv={space.nv}"
            )
        q = ca.SX.sym("q", space.nq)
        G, pm = space.velocity_map(q), constants(model.params)
        points = [model.frame(q, a, pm)[1] for a in at]
        J = [ca.mtimes(ca.jacobian(p, q), G) for p in points]
        self._f = ca.Function("inversion", [q], [ca.vertcat(*points), ca.vertcat(*J)], OPTS)
        self._n, self._m = space.nq, len(points)
        self._neutral = np.asarray(space.neutral(), dtype=float)
        self._damping, self._tol, self._max_iter = damping, tol, max_iter
        self.reset()

    def reset(self, q: ArrayLike | None = None) -> None:
        """Forget the warm start: the next fit starts at ``q``, by default the neutral one."""
        self._q = self._neutral.copy() if q is None else np.array(q, dtype=float)
        self.converged = False

    def _eval(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p, J = self._f(q)
        return np.array(p).ravel(), np.array(J)

    def __call__(self, positions: ArrayLike, q0: ArrayLike | None = None) -> np.ndarray:
        """The q whose points are at ``positions`` (m, 3), from ``q0`` or the last result."""
        y = np.asarray(positions, dtype=float)
        if y.shape != (self._m, 3):
            raise ValueError(f"positions must be ({self._m}, 3), one row per point, not {y.shape}")
        y = y.ravel()
        q = self._q if q0 is None else np.array(q0, dtype=float)
        p, J = self._eval(q)
        r = p - y
        for _ in range(self._max_iter):
            if np.abs(r).max() <= self._tol:
                break
            JtJ, g, lam = J.T @ J, J.T @ r, self._damping
            for _ in range(RETRIES + 1):
                trial = q - np.linalg.solve(JtJ + lam * np.eye(self._n), g)
                p_trial, J_trial = self._eval(trial)
                r_trial = p_trial - y
                if r_trial @ r_trial < r @ r:
                    q, J, r = trial, J_trial, r_trial
                    break
                lam *= GROWTH
            else:  # no damping lowered the error: the fit is stuck
                break
        self.converged = bool(np.abs(r).max() <= self._tol)
        self._q = q
        return q.copy()
