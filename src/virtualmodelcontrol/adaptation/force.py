"""Force tracking: gradient descent of a contact force error on a controller's live Params."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from ..models.kinematics import Kinematics


class ForceTracking:
    """Gradient descent of the contact force error ½‖f − f_des‖² on a controller's live Params.

    ``params`` are glob patterns of live Params: references move the springs' goals, stiffnesses
    change the springs. The contact is at ``site`` of the robot; with a ``normal`` the force is the
    one along it. A step ``p ← p + α g`` takes the largest α that keeps the predicted change of the
    force below ``max_force_step`` [N] and the change of each Param (its norm) below ``max_step``.
    """

    def __init__(
        self,
        controller: Any,
        site: Any,
        params: str | Sequence[str],
        normal: ArrayLike | None = None,
        *,
        max_force_step: float = 0.01,
        max_step: float | None = None,
    ) -> None:
        compiled = controller.compiled
        patterns = [params] if isinstance(params, str) else list(params)
        names = [
            n for n in compiled.live if any(fnmatch.fnmatchcase(n, pattern) for pattern in patterns)
        ]
        if not names:
            raise ValueError(f"no live Param matches {patterns}; compile with runtime=[...]")
        self.names, self.max_force_step, self.max_step = names, max_force_step, max_step
        slices = compiled.live_slices()
        self._shapes = {n: compiled.params[n].shape for n in names}
        self._slices = {n: slices[n] for n in names}
        n_angles, n_rates = compiled.n_motors
        self._offset = n_angles + n_rates + compiled.z0.size
        picked = np.concatenate([np.arange(s.start, s.stop) for s in self._slices.values()])
        self._picked = picked
        self._force = self._build(compiled, site, normal, picked + self._offset)

    def _build(self, compiled: Any, site: Any, normal: Any, columns: np.ndarray) -> ca.Function:
        system = compiled.system
        act, pa = system.actuation, constants(system.actuation.params)
        x = ca.SX.sym("x", compiled.fast.size1_in(0))
        theta = x[: compiled.n_motors[0]]
        u = compiled.fast(x)[: compiled.n_u]
        q = act.config_from_motors(theta, pa)
        tau = act.allocate(act.generalized_force(u, q, pa), q, pa)  # delivered, as motor torques
        J = Kinematics(system.robot, coordinates="motors").functions(site)(theta)[2]
        if normal is None:
            A = ca.solve(ca.mtimes(J, J.T), J)
        else:
            n = np.asarray(normal, dtype=float).ravel()
            j = ca.mtimes(J.T, ca.DM(n / np.linalg.norm(n)))
            A = ca.mtimes(ca.DM(n / np.linalg.norm(n)), j.T) / ca.dot(j, j)
        f = ca.mtimes(A, tau)
        return ca.Function("force", [x], [f, ca.jacobian(f, x[columns.tolist()])])

    def direction(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> dict[str, Any]:
        """The descent direction g = −(∂f/∂p)ᵀ (f_meas − f_des) of each Param, in its own shape."""
        return self._split(self._gradient(controller, f_meas, f_des)[0])

    def step(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> tuple[float, float]:
        """One step on ``controller`` (or a ``Tank`` around it), after its last ``step``: returns
        the step size α and the jump of the controller's energy [J]. A tank applies only the
        fraction of the step it pays for."""
        g, dfdp = self._gradient(controller, f_meas, f_des)
        if not np.any(g):
            return 0.0, 0.0
        alpha = self.max_force_step / max(float(np.linalg.norm(dfdp @ g)), 1e-12)
        blocks = self._split(g, shaped=False)
        if self.max_step is not None:
            largest = max(float(np.linalg.norm(b)) for b in blocks.values())
            alpha = min(alpha, self.max_step / max(largest, 1e-12))
        live = controller.live_params()
        new = {n: np.ravel(live[n], order="F") + alpha * b for n, b in blocks.items()}
        return alpha, float(controller.set(new))

    def _gradient(
        self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(controller.inputs(), dtype=float)
        n = x.size - 1 - controller.params.size
        x[n:-1] = controller.params
        _, dfdp = self._force(x)
        dfdp = np.array(dfdp)
        error = np.asarray(f_meas, dtype=float).ravel() - np.asarray(f_des, dtype=float).ravel()
        return -dfdp.T @ error, dfdp

    def _split(self, g: np.ndarray, shaped: bool = True) -> dict[str, Any]:
        out, k = {}, 0
        for name, s in self._slices.items():
            block = g[k : k + (s.stop - s.start)]
            out[name] = np.reshape(block, self._shapes[name], order="F") if shaped else block
            k += s.stop - s.start
        return out
