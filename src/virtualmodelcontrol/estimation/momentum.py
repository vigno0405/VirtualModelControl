"""External forces on a robot from its motion, without accelerations: a momentum observer."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import compile_dynamics
from ..models.kinematics import Kinematics, contact_map


class MomentumObserver:
    """The generalized force [N or N·m] that the robot's model does not explain: the surroundings'.

    With the momentum p = M(q) v, the observer integrates p̂' = ṗ_model + r̂ and holds r̂ = K (p − p̂),
    so that r̂' = K (r − r̂): a first-order low-pass of the true force r, of bandwidth ``gain`` K
    [1/s], with no acceleration in it. ``robot`` is the model without the surroundings (the
    system's own if left out); ``dt`` [s] is the step between calls. The robot's velocity must be
    the rate of its configuration (nq = nv, no spinning bodies).
    """

    def __init__(self, system: Any, dt: float, gain: float, robot: Any = None) -> None:
        model = system.robot if robot is None else robot
        space = model.model.space
        if space.nq != space.nv:
            raise ValueError(
                f"the observer needs as many velocity as configuration coordinates, "
                f"the robot's space has nq={space.nq} and nv={space.nv}"
            )
        dynamics = compile_dynamics(model, actuation=system.actuation)
        q, v, t = ca.SX.sym("q", space.nq), ca.SX.sym("v", space.nv), ca.SX.sym("t")
        u = ca.SX.sym("u", dynamics.n_u)
        p = ca.SX.sym("p", dynamics.params.size(dynamics.live))
        M = dynamics.mass(q, p)
        momentum = ca.mtimes(M, v)
        rate = ca.mtimes(ca.jacobian(momentum, q), v) + ca.mtimes(
            M, dynamics.forward(q, v, u, p, t)
        )  # ṗ without the surroundings: Ṁ v + M a
        self._terms = ca.Function("momentum", [q, v, u, p, t], [momentum, rate])
        self._dynamics, self._model = dynamics, model
        self._sites: dict[Any, ca.Function] = {}
        self.dt, self.gain = dt, gain
        self.reset()

    def reset(self) -> None:
        """Forget the estimate: the next call starts from the momentum it is given."""
        self._p_hat: np.ndarray | None = None
        self.r = np.zeros(0)
        self._q = np.zeros(0)

    def __call__(self, q: ArrayLike, v: ArrayLike, u: ArrayLike, t: float = 0.0) -> np.ndarray:
        """Take the configuration, the velocity and the motor command of one step: the estimate of
        the external generalized force (n,)."""
        live = self._dynamics.live_values()
        qa, va, ua = (np.asarray(x, dtype=float).ravel() for x in (q, v, u))
        p, rate = (np.asarray(x, dtype=float).ravel() for x in self._terms(qa, va, ua, live, t))
        if self._p_hat is None:
            self._p_hat = p.copy()
        self.r = self.gain * (p - self._p_hat)
        self._p_hat = self._p_hat + self.dt * (rate + self.r)
        self._q = qa
        return self.r

    def force(self, site: Any, normal: ArrayLike | None = None) -> np.ndarray:
        """The force [N] the robot exerts at ``site`` on the surroundings that the last estimate
        stands for, (3,), as ``ContactForce`` gives it: along ``normal`` if given (the
        pseudo-inverse of the site's Jacobian carries it)."""
        if site not in self._sites:
            self._sites[site] = Kinematics(self._model).functions(site)
        J = np.asarray(self._sites[site](self._q)[2])
        return -contact_map(J, normal) @ self.r
