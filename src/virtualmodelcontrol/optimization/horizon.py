"""Moving-horizon estimation: the state that best explains the last readings, as a least squares."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..dynamics import OPTS, compile_dynamics
from ..estimation.kalman import INITIAL_VARIANCE, PROCESS_NOISE
from ..estimation.measurement import Measurement, covariance


class MovingHorizon:
    """The state (q, v) of a robot over its last ``window`` steps that best agrees with what the
    sensors read, with what its own dynamics allow, and with the estimate before the window.

    It minimizes ‖x₀ − x̄‖²_P⁻¹ + Σ ‖xₖ₊₁ − f(xₖ, uₖ)‖²_Q⁻¹ + Σ ‖yₖ − Hₖ xₖ‖²_R⁻¹ over the states in
    the window, by Gauss-Newton with a trust region. f is one step of the robot's own integrator
    over ``dt`` [s], so ``dt`` is a control period, not a long step. ``P`` and ``Q`` are the
    covariances of the state at the first reset and of the process noise; ``robot`` is the arm
    alone when the system's has surroundings. A reading may be missing at any step. When the window
    slides on, the state before it, x̄ with its covariance, goes one step of an extended Kalman
    filter on: with a linear robot the estimate is the Kalman filter's, however short the window.
    """

    def __init__(
        self,
        system: Any,
        dt: float,
        window: int = 10,
        Q: ArrayLike | None = None,
        P: ArrayLike | None = None,
        robot: Any = None,
        iterations: int = 30,
    ) -> None:
        model = system.robot if robot is None else robot
        space = model.model.space
        if space.nq != space.nv:
            raise ValueError(
                f"the estimator needs as many velocity as configuration coordinates, "
                f"the robot's space has nq={space.nq} and nv={space.nv}"
            )
        n = space.nq
        act, pa = system.actuation, constants(system.actuation.params)
        dynamics = compile_dynamics(model, actuation=act)
        x, u, t = ca.SX.sym("x", 2 * n), ca.SX.sym("u", dynamics.n_u), ca.SX.sym("t")
        p = ca.SX.sym("p", dynamics.params.size(dynamics.live))
        q_next, v_next = dynamics.step(x[:n], x[n:], u, p, t, dt)
        after = ca.vertcat(q_next, v_next)
        self._step = ca.Function("step", [x, u, p, t], [after, ca.jacobian(after, x)], OPTS)
        self._dynamics, self._n = dynamics, n
        self._maps: dict[int, ca.Function] = {}
        theta, theta_dot = (
            ca.SX.sym("theta", act.motor_sizes(space)[0]),
            ca.SX.sym("theta_dot", act.motor_sizes(space)[1]),
        )
        q_enc = act.config_from_motors(theta, pa)
        self._config = ca.Function(
            "config", [theta, theta_dot], [q_enc, act.velocity_from_motors(q_enc, theta_dot, pa)]
        )
        self._neutral = np.asarray(space.neutral(), dtype=float)
        self.dt, self.window, self.iterations = dt, window, iterations
        self._Q = covariance(PROCESS_NOISE if Q is None else Q, 2 * n, "Q")
        self._P0 = covariance(INITIAL_VARIANCE if P is None else P, 2 * n, "P")
        self.reset()

    def reset(self, q0: ArrayLike | None = None) -> None:
        """Start again at rest at ``q0`` (the space's neutral configuration by default)."""
        q0 = self._neutral if q0 is None else np.asarray(q0, dtype=float)
        self._prior = np.concatenate([q0, np.zeros(self._n)])
        self._arrival = self._P0.copy()
        self._states = np.zeros((0, 2 * self._n))
        self._commands: list[np.ndarray] = []
        self._readings: list[tuple[np.ndarray, np.ndarray]] = []
        self._times: list[float] = []
        self.cost = 0.0

    def step(self, u: ArrayLike, measurements: Sequence[Measurement], t: float = 0.0) -> np.ndarray:
        """Take the motor command ``u`` held since the last call, and the readings of this one:
        the estimate (q, v) of the present state. The first call has no command to give."""
        n, first = self._n, not self._times
        if not first:
            self._commands.append(np.asarray(u, dtype=float).ravel())
        self._times.append(t)
        weighted = []
        for m in measurements:
            # ‖root (y − H x)‖² is (y − H x)ᵀ R⁻¹ (y − H x)
            root = np.linalg.cholesky(np.linalg.inv(m.R)).T
            weighted.append((root @ m.H(n), root @ m.y))
        rows = np.vstack([a for a, _ in weighted]) if weighted else np.zeros((0, 2 * n))
        self._readings.append(
            (rows, np.concatenate([z for _, z in weighted]) if weighted else np.zeros(0))
        )
        if len(self._times) > self.window + 1:  # slide: a Kalman step moves the prior on
            self._advance_prior()
            self._states = self._states[1:]
            self._commands, self._readings, self._times = (
                self._commands[1:],
                self._readings[1:],
                self._times[1:],
            )
        start = self._states
        guess = start[-1] if len(start) else self._prior
        if len(start) and self._commands:
            live = self._dynamics.live_values()
            guess = np.array(
                self._step(start[-1], self._commands[-1], live, self._times[-2])[0]
            ).ravel()
        self._states = np.vstack([start, guess]) if len(self._times) > len(start) else start
        self._solve()
        return self._states[-1].copy()

    def _advance_prior(self) -> None:
        """The state before the window, one step on: the first stage's readings update it and the
        dynamics carry it to the next stage (the extended Kalman filter's two steps)."""
        x, P = self._prior, self._arrival
        rows, z = self._readings[0]
        if len(z):  # the readings are weighted: their covariance is the identity
            gain = P @ rows.T @ np.linalg.inv(rows @ P @ rows.T + np.eye(len(z)))
            x, P = x + gain @ (z - rows @ x), P - gain @ rows @ P
        after, jacobian = self._step(
            x, self._commands[0], self._dynamics.live_values(), self._times[0]
        )
        jacobian = np.array(jacobian)
        self._prior = np.array(after).ravel()
        self._arrival = jacobian @ P @ jacobian.T + self._Q
        self._arrival = 0.5 * (self._arrival + self._arrival.T)

    def _solve(self) -> None:
        """Gauss-Newton with a trust region on the window's states."""
        n2, m = 2 * self._n, len(self._times)
        live = self._dynamics.live_values()
        sqrt_p = np.linalg.cholesky(np.linalg.inv(self._arrival)).T
        sqrt_q = np.linalg.cholesky(np.linalg.inv(self._Q)).T
        if m > 1 and m - 1 not in self._maps:
            self._maps[m - 1] = self._step.map(m - 1)

        def system(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """The residuals r and their Jacobian J with respect to the window's states."""
            first = np.zeros((n2, m * n2))
            first[:, :n2] = sqrt_p
            parts, rows = [sqrt_p @ (states[0] - self._prior)], [first]
            if m > 1:
                commands = np.column_stack(self._commands)
                times = np.array(self._times[:-1])[None, :]
                tiled = np.tile(live[:, None], (1, m - 1))
                after, derivative = (
                    np.array(a) for a in self._maps[m - 1](states[:-1].T, commands, tiled, times)
                )
                for k in range(m - 1):  # the dynamics' share: x_{k+1} − f(x_k, u_k)
                    parts.append(sqrt_q @ (states[k + 1] - after[:, k]))
                    block = np.zeros((n2, m * n2))
                    block[:, k * n2 : (k + 1) * n2] = -sqrt_q @ derivative[:, k * n2 : (k + 1) * n2]
                    block[:, (k + 1) * n2 : (k + 2) * n2] = sqrt_q
                    rows.append(block)
            for k, (A, z) in enumerate(self._readings):  # the sensors': y_k − H_k x_k
                if len(z):
                    parts.append(z - A @ states[k])
                    block = np.zeros((len(z), m * n2))
                    block[:, k * n2 : (k + 1) * n2] = -A
                    rows.append(block)
            return np.concatenate(parts), np.vstack(rows)

        x = self._states.copy()
        r, J = system(x)
        cost, damping = 0.5 * r @ r, 1e-6
        for _ in range(self.iterations):
            H, g = J.T @ J, J.T @ r
            step = np.linalg.solve(H + damping * np.diag(np.maximum(np.diag(H), 1e-12)), -g)
            trial = x + step.reshape(m, n2)
            r_new, J_new = system(trial)
            if 0.5 * r_new @ r_new < cost:
                gain = cost - 0.5 * r_new @ r_new
                x, r, J = trial, r_new, J_new
                cost, damping = 0.5 * r @ r, max(damping / 5, 1e-12)
                if gain <= 1e-14 * max(cost, 1e-12):
                    break
            else:
                damping *= 10
                if damping > 1e12:
                    break
        self._states, self.cost = x, float(cost)
        self._covariance = np.linalg.pinv(J.T @ J)[-n2:, -n2:]

    def encoder(
        self,
        theta: ArrayLike,
        theta_dot: ArrayLike | None,
        Rq: ArrayLike,
        Rv: ArrayLike | None = None,
        name: str = "encoder",
    ) -> Measurement:
        """The measurement of motor angles [rad] and rates [rad/s], as the Kalman filter's."""
        rates = np.zeros(self._dynamics.n_u) if theta_dot is None else theta_dot
        seen = self._config(
            ca.DM(np.asarray(theta, dtype=float)), ca.DM(np.asarray(rates, dtype=float))
        )
        q, v = (np.array(a).ravel() for a in seen)
        return Measurement(q, None if theta_dot is None else v, Rq, Rv, name=name)

    @property
    def q(self) -> np.ndarray:
        """The estimated configuration."""
        return self._states[-1][: self._n].copy()

    @property
    def v(self) -> np.ndarray:
        """The estimated velocity."""
        return self._states[-1][self._n :].copy()

    @property
    def P(self) -> np.ndarray:
        """The covariance of the present state, from the Gauss-Newton approximation."""
        return self._covariance.copy()

    @property
    def states(self) -> np.ndarray:
        """The estimated states of the window, one row per step."""
        return self._states.copy()
