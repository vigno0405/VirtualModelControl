"""IMUs on a soft arm: the curvature of each section and its rate, from gyros and accelerometers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..math import exp_so3, skew
from ..models.continuum.pcc import segment_frame

GRAVITY = 9.81
"""Default size of what a still accelerometer reads [m/s²]."""

STILL_STEPS = 1000
"""Default gravity-only steps of a calibration: 20 time constants at the default gain."""

STILL_DT = 0.01
"""Default length [s] of the steps of a calibration."""

TINY = 1e-9
"""Below this the tilt of a section is taken as zero when its curvature is read from a rotation."""


def _rate_map(phi: np.ndarray) -> np.ndarray:
    """The matrix that takes a body angular velocity to the rate of the rotation vector ``phi``."""
    angle, S = np.linalg.norm(phi), skew(phi)
    c = 1 / 12 if angle < 1e-6 else 1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
    return np.eye(3) + 0.5 * S + c * S @ S


class ImuFilter:
    """The curvature (Dx, Dy) of each section of a PCC arm and its rate, from IMUs at its base and
    at the end of each section.

    Per section, the gyros turn the relative rotation and the gravity both accelerometers see
    corrects it (gain ``kp`` [1/s], only while both read ``gravity`` within the fraction
    ``acc_tol``). ``mounts`` is one 3 by 3 per IMU, base first, v_arm = M v_sensor (identity by
    default). The arm is read at construction.
    """

    def __init__(
        self,
        arm: Any,
        mounts: ArrayLike | None = None,
        kp: float = 2.0,
        acc_tol: float = 0.1,
        gravity: float = GRAVITY,
    ) -> None:
        model = arm.model if hasattr(arm, "components") else arm
        if not hasattr(model, "n_segments"):
            raise ValueError("the IMU filter needs a PCC arm (for several arms, one arm's model)")
        n = model.n_segments
        self._radius = [float(model.params[f"seg{i + 1}.d"].value) for i in range(n)]
        self._eps = float(model.eps)
        self._mounts = (
            np.tile(np.eye(3), (n + 1, 1, 1)) if mounts is None else np.asarray(mounts, dtype=float)
        )
        if self._mounts.shape != (n + 1, 3, 3):
            raise ValueError(f"{n + 1} IMUs need mounts of shape ({n + 1}, 3, 3)")
        self.kp, self.acc_tol, self.gravity = kp, acc_tol, gravity
        self.gyro_bias = np.zeros((n + 1, 3))
        self._n = n
        self.reset()

    @property
    def observed(self) -> np.ndarray:
        """The indices of the arm's q that the IMUs see: Dx and Dy of every section, not Dl."""
        return np.array([3 * i + j for i in range(self._n) for j in (0, 1)])

    @property
    def q(self) -> np.ndarray:
        """The curvatures (Dx, Dy) of the sections, base to tip, as of the last update."""
        return self._q.copy()

    @property
    def v(self) -> np.ndarray:
        """Their rates, as of the last update."""
        return self._v.copy()

    def reset(self) -> None:
        """Back to the straight pose."""
        self._R = [np.eye(3) for _ in range(self._n)]
        self._q = np.zeros(2 * self._n)
        self._v = np.zeros(2 * self._n)

    def calibrate(
        self, gyro: ArrayLike, acc: ArrayLike, steps: int = STILL_STEPS, dt: float = STILL_DT
    ) -> None:
        """Gyro bias and starting pose from samples (N, IMUs, 3) of the arm held still."""
        self.gyro_bias = np.mean(np.asarray(gyro, dtype=float), axis=0)
        still = np.mean(np.asarray(acc, dtype=float), axis=0)
        self.reset()
        for _ in range(steps):  # the gyros read their bias only: the gravity corrects alone
            self.update(self.gyro_bias, still, dt)

    def update(self, gyro: ArrayLike, acc: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Take one reading of all IMUs (rows, base first) after ``dt`` [s]; give q and v."""
        gyro, acc = np.asarray(gyro, dtype=float), np.asarray(acc, dtype=float)
        w = [self._mounts[k] @ (gyro[k] - self.gyro_bias[k]) for k in range(self._n + 1)]
        a = [self._mounts[k] @ acc[k] for k in range(self._n + 1)]
        trusted = [abs(np.linalg.norm(x) / self.gravity - 1.0) < self.acc_tol for x in a]
        for i in range(self._n):
            R = self._R[i]
            w_rel = w[i + 1] - R.T @ w[i]
            if trusted[i] and trusted[i + 1]:
                up_base, up_tip = a[i] / np.linalg.norm(a[i]), a[i + 1] / np.linalg.norm(a[i + 1])
                w_rel_turned = w_rel + self.kp * np.cross(up_tip, R.T @ up_base)
            else:
                w_rel_turned = w_rel
            dx, dy = self._curvature(i, R @ exp_so3(w_rel_turned * dt))
            self._R[i] = self._rotation(i, dx, dy)
            d = self._radius[i]
            rate = _rate_map(np.array([-dy, dx, 0.0]) / d) @ w_rel  # the twist is dropped
            self._q[2 * i : 2 * i + 2] = dx, dy
            self._v[2 * i : 2 * i + 2] = d * rate[1], -d * rate[0]
        return self.q, self.v

    def _rotation(self, i: int, dx: float, dy: float) -> np.ndarray:
        """The relative rotation of section ``i``: the arm model's, at the end of the section."""
        R, _ = segment_frame(dx, dy, 0.0, 1.0, self._radius[i], 1.0, self._eps)
        return np.array(R, dtype=float)

    def _curvature(self, i: int, R: np.ndarray) -> tuple[float, float]:
        """(Dx, Dy) of section ``i`` from its relative rotation, by the tip's z axis, which a
        twist about the backbone leaves alone."""
        tilt = np.hypot(R[0, 2], R[1, 2])
        d = self._radius[i]
        scale = d * np.arctan2(tilt, R[2, 2]) / tilt if tilt > TINY else d
        return float(scale * R[0, 2]), float(scale * R[1, 2])
