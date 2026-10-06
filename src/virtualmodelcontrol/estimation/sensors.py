"""Simulated sensors: the true state in, a noisy reading out, as the estimators take it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..models.actuation import Direct
from ..models.kinematics import Kinematics

GRAVITY = 9.81
"""[m/s²] What a still accelerometer reads."""


class Encoders:
    """The motor angles [rad] and rates [rad/s] of a robot, with noise and a slack.

    ``slack`` [rad] is a constant offset per motor, drawn once in ±``slack``: a tendon that is
    slack makes the encoder read the motor, not the arm. ``noise`` and ``rate_noise`` are standard
    deviations. ``robot`` is a mechanism with its actuation (a ``Direct`` one if it has none).
    """

    def __init__(
        self,
        robot: Any,
        noise: float = 0.01,
        rate_noise: float = 0.05,
        slack: float = 0.0,
        seed: int = 0,
    ) -> None:
        actuation = robot.actuation if robot.actuation is not None else Direct()
        self._actuation, self._p = actuation, constants(actuation.params)
        self._nq = robot.model.space.nq
        self.noise, self.rate_noise = noise, rate_noise
        self._rng = np.random.default_rng(seed)
        angles, _ = actuation.motor_sizes(robot.model.space)
        self.slack = self._rng.uniform(-slack, slack, angles)
        """The constant offset of each motor [rad]."""

    def read(self, q: ArrayLike, v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The motor angles and rates of the state (q, v), noisy."""
        theta = np.array(self._actuation.motor_angles(np.asarray(q, dtype=float), self._p)).ravel()
        rates = np.array(
            self._actuation.motor_rates(
                np.asarray(q, dtype=float), np.asarray(v, dtype=float), self._p
            )
        ).ravel()
        return (
            theta + self.slack + self._rng.normal(0.0, self.noise, theta.shape),
            rates + self._rng.normal(0.0, self.rate_noise, rates.shape),
        )


class Markers:
    """The positions [m] of points of a robot, as a motion-capture system reads them.

    ``at`` lists the points: an arc parameter s, a site name or ``(part, s)``, as ``Kinematics``
    takes them. ``noise`` [m] is the standard deviation on every coordinate.
    """

    def __init__(self, robot: Any, at: Sequence[Any], noise: float = 3e-4, seed: int = 0) -> None:
        self._kin, self.at, self.noise = Kinematics(robot), list(at), noise
        self._rng = np.random.default_rng(seed)

    def read(self, q: ArrayLike) -> np.ndarray:
        """The positions of the points at the configuration ``q``, (len(at), 3), noisy."""
        q = np.asarray(q, dtype=float)
        seen = np.array([self._kin.position(q, s) for s in self.at])
        return seen + self._rng.normal(0.0, self.noise, seen.shape)


class Imus:
    """Gyros and accelerometers on the frames of a robot, each in its own axes.

    The gyro reads the angular velocity, plus a constant bias drawn once (standard deviation
    ``bias``) and noise; the accelerometer reads what a still sensor feels, ``gravity`` [m/s²]
    along +z of the base frame, plus noise. ``sites`` are the frames, as ``Kinematics`` takes them.
    """

    def __init__(
        self,
        robot: Any,
        sites: Sequence[Any],
        bias: float = 0.02,
        gyro_noise: float = 0.005,
        acc_noise: float = 0.05,
        gravity: float = GRAVITY,
        seed: int = 0,
    ) -> None:
        self._kin, self.sites, self.gravity = Kinematics(robot), list(sites), gravity
        self.gyro_noise, self.acc_noise = gyro_noise, acc_noise
        self._rng = np.random.default_rng(seed)
        self.bias = self._rng.normal(0.0, bias, (len(self.sites), 3))
        """The constant bias of each gyro [rad/s]."""

    def read(self, q: ArrayLike, v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The gyros and the accelerometers at the state (q, v), (len(sites), 3) each, noisy."""
        q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        gyro, acc = np.zeros((len(self.sites), 3)), np.zeros((len(self.sites), 3))
        for i, site in enumerate(self.sites):
            R = self._kin.rotation(q, site)
            gyro[i] = R.T @ (self._kin.angular_jacobian(q, site) @ v)
            acc[i] = R.T @ np.array([0.0, 0.0, self.gravity])
        turning = gyro + self.bias + self._rng.normal(0.0, self.gyro_noise, gyro.shape)
        return turning, acc + self._rng.normal(0.0, self.acc_noise, acc.shape)


class LoadCell:
    """A force sensor [N]: the true force with a bias and noise."""

    def __init__(self, noise: float = 0.01, bias: float = 0.0, seed: int = 0) -> None:
        self.noise, self.bias = noise, bias
        self._rng = np.random.default_rng(seed)

    def read(self, force: ArrayLike) -> np.ndarray:
        """The reading of the force ``force``, as an array of the same shape."""
        force = np.asarray(force, dtype=float)
        return force + self.bias + self._rng.normal(0.0, self.noise, force.shape)
