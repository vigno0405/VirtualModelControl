"""Force tracking: gradient descent of a contact force error on a controller's live Params."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..control.state import require_motor_layout
from ..core.params import constants
from ..models.kinematics import Kinematics, contact_map
from .limits import TINY, admissible, live_matching


class ForceTracking:
    """Gradient descent of a contact force error on live Params of a controller, step by step.

    ``params`` are glob patterns of the Params (goals, stiffnesses); ``normal`` keeps the force
    along it. A step is the largest that changes the force by ``max_force_step`` [N], or by
    ``max_step`` in a Param (its norm), or ``rate`` if given. See the force tutorial.
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
        rate: float | None = None,
    ) -> None:
        require_motor_layout(controller, "ForceTracking")
        compiled = controller.compiled
        self.names = live_matching(compiled, params)
        self.max_force_step, self.max_step, self.rate = max_force_step, max_step, rate
        self.alpha = 0.0
        """The step size of the last step."""
        slices = compiled.live_slices()
        self._shapes = {n: compiled.params[n].shape for n in self.names}
        self._params = {n: compiled.params[n] for n in self.names}
        self._slices = {n: slices[n] for n in self.names}
        n_angles, n_rates = compiled.n_motors
        offset = n_angles + n_rates + compiled.z0.size  # where the live Params start in x
        picked = np.concatenate([np.arange(s.start, s.stop) for s in self._slices.values()])
        self._normal = normal
        self._terms = self._build(compiled, site, picked + offset)

    @staticmethod
    def _build(compiled: Any, site: Any, columns: np.ndarray) -> ca.Function:
        system = compiled.system
        act, pa = system.actuation, constants(system.actuation.params)
        x = ca.SX.sym("x", compiled.fast.size1_in(0))
        theta = x[: compiled.n_motors[0]]
        u = compiled.fast(x)[: compiled.n_u]
        q = act.config_from_motors(theta, pa)
        tau = act.allocate(act.generalized_force(u, q, pa), q, pa)  # delivered, as motor torques
        J = Kinematics(system.robot, coordinates="motors").functions(site)(theta)[2]
        return ca.Function("terms", [x], [tau, ca.jacobian(tau, x[columns.tolist()]), J])

    def direction(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> dict[str, Any]:
        """The descent direction of each Param, in its own shape, without applying it."""
        return self._split(self._gradient(controller, f_meas, f_des)[0])

    def step(self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike) -> float:
        """One step on ``controller`` (or a ``Tank`` around it), after its last ``step``.

        ``f_meas`` and ``f_des`` are forces [N], vectors of three entries. Returns the jump of
        the controller's energy [J] that was applied; a tank applies only the part it pays for.
        """
        g, dfdp = self._gradient(controller, f_meas, f_des)
        if not np.any(g):
            self.alpha = 0.0
            return 0.0
        if self.rate is None:
            alpha = self.max_force_step / max(float(np.linalg.norm(dfdp @ g)), TINY)
        else:
            alpha = self.rate
        blocks = self._split(g, shaped=False)
        if self.max_step is not None:
            largest = max(float(np.linalg.norm(b)) for b in blocks.values())
            alpha = min(alpha, self.max_step / max(largest, TINY))
        self.alpha = alpha
        live = controller.live_params()
        new = {
            n: admissible(self._params[n], np.ravel(live[n], order="F") + alpha * b)
            for n, b in blocks.items()
        }
        return float(controller.set(new))

    def _gradient(
        self, controller: Any, f_meas: ArrayLike, f_des: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        _, dtau, J = (np.array(m) for m in self._terms(controller.inputs()))
        dfdp = contact_map(J, self._normal) @ dtau
        error = np.reshape(np.asarray(f_meas, float), 3) - np.reshape(np.asarray(f_des, float), 3)
        return -dfdp.T @ error, dfdp

    def _split(self, g: np.ndarray, shaped: bool = True) -> dict[str, Any]:
        out, k = {}, 0
        for name, s in self._slices.items():
            block = g[k : k + (s.stop - s.start)]
            out[name] = np.reshape(block, self._shapes[name], order="F") if shaped else block
            k += s.stop - s.start
        return out
