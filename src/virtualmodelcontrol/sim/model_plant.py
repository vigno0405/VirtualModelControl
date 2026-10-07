"""ModelPlant: simulates a robot from its own components, and reads like the hardware."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..core.signals import Signals
from ..dynamics import compile_dynamics, needs_energy
from ..mechanisms.mechanism import Mechanism
from ..models.actuation import Direct


class ModelPlant:
    """Simulated robot: linearly implicit Euler steps of at most ``max_step`` [s].

    ``read`` returns motor angles and rates (as the hardware does) plus q and v. The command
    ``motor_torque`` is held between calls to ``advance``.
    """

    def __init__(
        self,
        robot: Mechanism,
        q0: ArrayLike | None = None,
        v0: ArrayLike | None = None,
        *,
        max_step: float = 1e-3,
        runtime: Iterable[str] = (),
    ) -> None:
        self.dynamics = compile_dynamics(robot, runtime)
        self.space = robot.model.space
        self.max_step = max_step
        self.p = self.dynamics.live_values()
        self._q0 = self.space.neutral() if q0 is None else np.asarray(q0, dtype=float)
        self._v0 = np.zeros(self.space.nv) if v0 is None else np.asarray(v0, dtype=float)
        self.reset()

    def reset(self, x0: Any = None, seed: int | None = None) -> None:
        """Back to (q0, v0), or to ``x0 = (q, v)``; time 0, zero command."""
        q, v = (self._q0, self._v0) if x0 is None else x0
        self.q: np.ndarray = np.array(q, dtype=float)
        self.v: np.ndarray = np.array(v, dtype=float)
        self.u: np.ndarray = np.zeros(self.dynamics.n_u)
        self.t = 0.0

    def read(self) -> Signals:
        """Motor angles and rates, plus the configuration and velocity."""
        theta, theta_dot = self.dynamics.motors(self.q, self.v, self.p)
        return Signals(
            self.t,
            motor_position=np.array(theta).ravel(),
            motor_velocity=np.array(theta_dot).ravel(),
            q=self.q,
            v=self.v,
        )

    def write(self, cmd: Signals) -> None:
        """Hold ``motor_torque`` [N·m] until the next write."""
        self.u = np.array(cmd["motor_torque"], dtype=float)

    def advance(self, dt: float) -> None:
        """Integrate over ``dt`` [s]."""
        n = max(1, math.ceil(dt / self.max_step - 1e-9))
        h = dt / n
        for _ in range(n):
            q, v = self.dynamics.step(self.q, self.v, self.u, self.p, self.t, h)
            self.q, self.v = np.array(q).ravel(), np.array(v).ravel()
            self.t += h

    def state_from_motors(self, theta: ArrayLike, theta_dot: ArrayLike) -> tuple[Any, Any]:
        """(q, v) for motor angles [rad] and rates [rad/s], through the transmission's exact
        inverse, at the current Param values."""
        robot = self.dynamics.robot
        actuation = robot.actuation if robot.actuation is not None else Direct()
        p = constants(actuation.params)
        q = actuation.config_from_motors(ca.DM(np.asarray(theta, dtype=float)), p)
        v = actuation.velocity_from_motors(q, ca.DM(np.asarray(theta_dot, dtype=float)), p)
        return np.array(ca.evalf(q)).ravel(), np.array(ca.evalf(v)).ravel()

    def elements(self) -> dict[str, dict[str, np.ndarray]]:
        """Each component of the robot that is not a mass (its springs, dampers, contacts), now:
        its coordinate ``y``, rate ``ydot``, ``force`` and ``torque``, its share of the generalized
        forces."""
        out = self.dynamics.elements(self.q, self.v, self.p, self.t)
        values = [np.array(x).ravel() for x in (out if isinstance(out, tuple) else [out])]
        quantities = ("y", "ydot", "force", "torque")
        return {
            name: dict(zip(quantities, values[4 * k : 4 * k + 4], strict=True))
            for k, name in enumerate(self.dynamics.element_names)
        }

    def energy(self) -> float:
        """Kinetic plus stored energy of the robot [J]."""
        energy, _ = needs_energy(self.dynamics, "the energy of a simulated robot")
        T, V = energy(self.q, self.v, self.p, self.t)
        return float(T) + float(V)

    def close(self) -> None:
        """Nothing to release."""
