"""Direct stiffness tracking: the stiffness Params that give a wanted stiffness at a site."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..estimation import TaskStiffness
from .limits import NOISE, admissible, live_matching


class StiffnessTracking:
    """Moves live stiffness Params towards the ones that give a wanted stiffness at ``site``.

    The stiffness of ``TaskStiffness`` is affine in the springs' stiffness, so the Params that
    give ``K_des`` solve a linear system: the solution of the smallest norm, which leaves out
    the stiffness that does not show at the site. A step goes a ``fraction`` of the way there.
    """

    def __init__(
        self,
        controller: Any,
        site: Any,
        params: str | Sequence[str],
        normal: ArrayLike | None = None,
        *,
        robot: Any = None,
        fraction: float = 1.0,
    ) -> None:
        compiled = controller.compiled
        self.names, self.fraction = live_matching(compiled, params), fraction
        self._params = {n: compiled.params[n] for n in self.names}
        self._slices = {n: compiled.live_slices()[n] for n in self.names}
        self._stiffness = TaskStiffness(controller, site, normal, robot)
        n_angles, n_rates = compiled.n_motors
        self._offset = n_angles + n_rates + compiled.z0.size  # where the live Params start in x
        self._columns = np.concatenate([np.arange(s.start, s.stop) for s in self._slices.values()])
        x = ca.SX.sym("x", compiled.fast.size1_in(0))
        K_motors = self._stiffness.terms(x)[1]
        columns = (self._columns + self._offset).tolist()
        self._dK = ca.Function("dK", [x], [ca.jacobian(ca.vec(K_motors), x[columns])])

    def target(
        self, controller: Any, K_des: ArrayLike, f_ext: ArrayLike | None = None
    ) -> dict[str, np.ndarray]:
        """The Params, flat, that give ``K_des`` (3 by 3) at the contact force ``f_ext``."""
        x = controller.inputs()
        A, K = self._stiffness.motors(controller, f_ext)
        B = np.kron(A, A) @ np.array(self._dK(x))  # d vec(K at the site) / d the Params
        now = np.ravel(A @ K @ A.T, order="F")
        wanted = np.ravel(np.asarray(K_des, dtype=float), order="F")
        p = x[self._columns + self._offset]
        solution = np.linalg.pinv(B, rcond=NOISE) @ (wanted - now + B @ p)
        out, k = {}, 0
        for name, s in self._slices.items():
            out[name] = solution[k : k + s.stop - s.start]
            k += s.stop - s.start
        return out

    def step(self, controller: Any, K_des: ArrayLike, f_ext: ArrayLike | None = None) -> float:
        """One step on ``controller`` (or a ``Tank`` around it); returns the energy jump [J]."""
        live = controller.live_params()
        new = {}
        for name, aim in self.target(controller, K_des, f_ext).items():
            here = np.ravel(live[name], order="F")
            new[name] = admissible(self._params[name], here + self.fraction * (aim - here))
        return float(controller.set(new))
