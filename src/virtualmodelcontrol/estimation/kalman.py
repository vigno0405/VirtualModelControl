"""Kalman filter of a robot's state (q, v), with the process model taken from its own dynamics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..dynamics import OPTS, compile_dynamics
from .measurement import Measurement, block_diagonal, covariance

PROCESS_NOISE = 1e-6
"""Default variance of the process noise, on every entry of (q, v)."""

INITIAL_VARIANCE = 1e-2
"""Default variance of the state after a reset, on every entry of (q, v)."""

SIGMA_DISTANCE = 3**0.5
"""How far the sigma points of the unscented prediction lie from the estimate, in standard
deviations (or less, in a state of two entries)."""


class KalmanFilter:
    """Kalman filter of a robot's state (q, v): its dynamics predict, the sensors' readings correct.

    ``dt`` [s] is the step; ``Q`` and ``P0`` are the covariances of the process noise and of the
    state at a reset; ``gate`` is the largest innovation d² a measurement may have (None: no
    test). ``robot`` is the arm alone when the system's has surroundings. Read at construction.

    The prediction linearises the dynamics at the estimate. With ``unscented``, it takes sigma
    points of the estimate through the dynamics instead, in ``substeps`` steps of the robot's own
    integrator each, which follows a strongly nonlinear arm (the sensors stay linear).
    """

    def __init__(
        self,
        system: Any,
        dt: float,
        Q: ArrayLike | None = None,
        gate: float | None = 40.0,
        robot: Any = None,
        P0: ArrayLike | None = None,
        unscented: bool = False,
        substeps: int = 10,
    ) -> None:
        model = system.robot if robot is None else robot
        space = model.model.space
        if space.nq != space.nv:
            raise ValueError(
                f"the filter needs as many velocity as configuration coordinates, "
                f"the robot's space has nq={space.nq} and nv={space.nv}"
            )
        n = space.nq
        act, pa = system.actuation, constants(system.actuation.params)
        n_angles, n_rates = act.motor_sizes(space)
        dynamics = compile_dynamics(model, actuation=act)
        q, v, a = ca.SX.sym("q", n), ca.SX.sym("v", n), ca.SX.sym("a", n)
        u, t = ca.SX.sym("u", n_rates), ca.SX.sym("t")
        p = ca.SX.sym("p", dynamics.params.size(dynamics.live))
        own = dynamics.residual(q, v, a, ca.SX.zeros(dynamics.n_u), p, t)  # no command in it
        r = own - act.generalized_force(u, q, pa)
        # about a robot at rest, before the derivatives: the terms in v and a drop out of K
        still, nil = ca.vertcat(v, a), ca.SX.zeros(2 * n)
        M = ca.jacobian(r, a)
        K = ca.jacobian(ca.substitute(r, still, nil), q)
        D = ca.substitute(ca.jacobian(ca.substitute(r, a, nil[:n]), v), v, nil[:n])
        self._rest = ca.Function(
            "rest", [q, u, p, t], [M, K, D, ca.substitute(r, still, nil)], OPTS
        )
        theta, theta_dot = ca.SX.sym("theta", n_angles), ca.SX.sym("theta_dot", n_rates)
        q_enc = act.config_from_motors(theta, pa)
        v_enc = act.velocity_from_motors(q_enc, theta_dot, pa)
        self._config = ca.Function("config", [theta, theta_dot], [q_enc, v_enc])
        self._motors = ca.Function(
            "motors", [q, v], [act.motor_angles(q, pa), act.motor_rates(q, v, pa)]
        )
        self._motor_jacobian = ca.Function(
            "motor_jacobian", [q], [ca.jacobian(act.motor_angles(q, pa), q)]
        )
        self._dynamics = dynamics
        self._n, self._n_angles, self._n_rates = n, n_angles, n_rates
        self._unscented, self._substeps = unscented, substeps
        self._points = dynamics.step.map(4 * n + 1)  # one column per sigma point
        self._neutral = np.asarray(space.neutral(), dtype=float)
        self.dt = dt
        self.gate = gate
        self._Q = covariance(PROCESS_NOISE if Q is None else Q, 2 * n, "Q")
        self._P0 = covariance(INITIAL_VARIANCE if P0 is None else P0, 2 * n, "P0")
        self.rejected_total = 0
        self.reset()

    def reset(self, q0: ArrayLike | None = None) -> None:
        """Start again at rest at ``q0`` (the space's neutral configuration by default)."""
        q0 = self._neutral if q0 is None else np.asarray(q0, dtype=float)
        self._x = np.concatenate([q0, np.zeros(self._n)])
        self._P = self._P0.copy()
        self.rejected: tuple[str | None, ...] = ()
        self.missing: tuple[str, ...] = ()

    def predict(self, u: ArrayLike, t: float = 0.0) -> None:
        """Advance ``dt`` under the motor command ``u``, held, with the model linearised here (or,
        unscented, through sigma points)."""
        from scipy.linalg import expm  # SciPy loads when a filter first predicts

        n, q = self._n, self._x[: self._n]
        u, p = np.asarray(u, dtype=float), self._dynamics.live_values()
        if self._unscented:
            self._predict_unscented(u, p, t)
            return
        M, K, D, r = (np.array(m) for m in self._rest(q, u, p, t))
        # M a = -r - K (q - q0) - D v about the estimate q0, so x' = A x + c; A and c go in one
        # matrix whose exponential holds the step of x' = A x + c (c rides on the last column)
        Ac = np.zeros((2 * n + 1, 2 * n + 1))
        Ac[:n, n : 2 * n] = np.eye(n)
        MK, MD, Mc = np.hsplit(
            np.linalg.solve(M, np.column_stack([K, D, K @ q - r.ravel()])), [n, 2 * n]
        )
        Ac[n : 2 * n, :n], Ac[n : 2 * n, n : 2 * n], Ac[n : 2 * n, 2 * n :] = -MK, -MD, Mc
        phi = expm(Ac * self.dt)
        F = phi[: 2 * n, : 2 * n]
        self._x = F @ self._x + phi[: 2 * n, 2 * n]
        self._P = F @ self._P @ F.T + self._Q

    def _predict_unscented(self, u: np.ndarray, p: np.ndarray, t: float) -> None:
        """The scaled unscented transform of the state through ``dt`` of the robot's dynamics."""
        n2 = 2 * self._n
        alpha = min(1.0, SIGMA_DISTANCE / n2**0.5)  # the points lie alpha sqrt(n2) out
        lam = alpha**2 * n2 - n2  # κ = 0
        root = np.linalg.cholesky((n2 + lam) * self._P + 1e-12 * np.eye(n2))
        points = np.column_stack([self._x, *(self._x[:, None] + s * root for s in (1, -1))])
        wm = np.full(2 * n2 + 1, 0.5 / (n2 + lam))
        wc = wm.copy()
        wm[0] = lam / (n2 + lam)
        wc[0] = wm[0] + 1.0 - alpha**2 + 2.0  # β = 2
        cols, h = points.shape[1], self.dt / self._substeps
        U, live = np.tile(u[:, None], (1, cols)), np.tile(np.ravel(p)[:, None], (1, cols))
        q, v = points[: self._n], points[self._n :]
        for k in range(self._substeps):
            q, v = (np.array(m) for m in self._points(q, v, U, live, t + k * h, h))
        after = np.vstack([q, v])
        self._x = after @ wm
        spread = after - self._x[:, None]
        self._P = (spread * wc) @ spread.T + self._Q

    def update(self, measurements: Sequence[Measurement], expected: Iterable[str] = ()) -> None:
        """Fuse the measurements; ``rejected`` names those the gate left out, ``missing`` the
        ``expected`` sensors that offered none."""
        self.missing = tuple(
            name for name in expected if name not in [m.name for m in measurements]
        )
        self.rejected = ()
        n, x, P = self._n, self._x, self._P
        if self.gate is not None:
            accepted, rejected = [], []
            for m in measurements:
                H = m.H(n)
                nu = m.y - H @ x
                S = H @ P @ H.T + m.R
                if nu @ np.linalg.solve(S, nu) <= self.gate:
                    accepted.append(m)
                else:
                    rejected.append(m)
            self.rejected = tuple(m.name for m in rejected)
            self.rejected_total += len(rejected)
            measurements = accepted
        if not measurements:
            return
        H = np.vstack([m.H(n) for m in measurements])
        y = np.concatenate([m.y for m in measurements])
        S = H @ P @ H.T + block_diagonal([m.R for m in measurements])
        K = P @ H.T @ np.linalg.inv(S)
        self._x = x + K @ (y - H @ x)
        self._P = (np.eye(2 * n) - K @ H) @ P

    def encoder(
        self,
        theta: ArrayLike,
        theta_dot: ArrayLike | None,
        Rq: ArrayLike,
        Rv: ArrayLike | None = None,
        name: str = "encoder",
    ) -> Measurement:
        """The measurement of motor angles [rad] and rates [rad/s], through the transmission.

        ``Rq`` and ``Rv`` are the covariances of the q and v they give, not of the motors. A robot
        with fewer motors than coordinates does not give q: the motors see combinations of it
        (``Underactuated`` sees Bᵀ q), so the measurement is of the motors themselves, linearised
        at the estimate, and ``Rq`` and ``Rv`` are the covariances of the motors' readings.
        """
        if self._n_angles < self._n:
            return self._motor_reading(theta, theta_dot, Rq, Rv, name)
        rates = np.zeros(self._n_rates) if theta_dot is None else theta_dot
        seen = self._config(
            ca.DM(np.asarray(theta, dtype=float)), ca.DM(np.asarray(rates, dtype=float))
        )
        q, v = (np.array(m).ravel() for m in seen)
        return Measurement(q, None if theta_dot is None else v, Rq, Rv, name=name)

    def _motor_reading(
        self,
        theta: ArrayLike,
        theta_dot: ArrayLike | None,
        Rq: ArrayLike,
        Rv: ArrayLike | None,
        name: str,
    ) -> Measurement:
        """Motor readings as a measurement of (q, v): the motors' Jacobian at the estimate is the
        matrix on q and on v (the rates are J v), and the angles are moved by what the matrix
        leaves out, so that the innovation is the reading minus what the model gives here."""
        J = np.array(self._motor_jacobian(self.q))
        model = np.array(self._motors(self.q, self.v)[0]).ravel()
        y_q = np.asarray(theta, dtype=float) - model + J @ self.q
        y_v = None if theta_dot is None else np.asarray(theta_dot, dtype=float)
        return Measurement(y_q, y_v, Rq, Rv, name=name, matrix=J)

    @property
    def q(self) -> np.ndarray:
        """The estimated configuration."""
        return self._x[: self._n].copy()

    @property
    def v(self) -> np.ndarray:
        """The estimated velocity."""
        return self._x[self._n :].copy()

    @property
    def theta(self) -> np.ndarray:
        """The motor angles of the estimated configuration [rad]."""
        return np.array(self._motors(self.q, self.v)[0]).ravel()

    @property
    def theta_dot(self) -> np.ndarray:
        """The motor rates of the estimated state [rad/s]."""
        return np.array(self._motors(self.q, self.v)[1]).ravel()

    @property
    def P(self) -> np.ndarray:
        """The covariance of the estimated state (q, v)."""
        return self._P.copy()
