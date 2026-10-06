"""Direct stiffness tracking: the stiffness Params that give a wanted stiffness at a site."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..estimation import TaskStiffness
from .limits import admissible, live_matching

NOISE = 1e-10
"""Singular values of the map below this share of the largest are rounding noise, not stiffness."""


class StiffnessTracking:
    """Moves live stiffness Params towards the ones that give a wanted stiffness at ``site``.

    The task-space stiffness of ``estimation.TaskStiffness`` is affine in the stiffness of the
    springs, so the Params that give ``K_des`` solve a linear system: the solution of the smallest
    norm, which leaves out the stiffness that does not show at the site. A step moves the Params
    ``rate`` of the way there, within their bounds, a square matrix staying symmetric and
    positive semidefinite. ``f_ext`` is the contact force the stiffness is taken at (default:
    the model's own).
    """

    def __init__(
        self,
        controller: Any,
        site: Any,
        params: str | Sequence[str],
        normal: ArrayLike | None = None,
        *,
        robot: Any = None,
        rate: float = 1.0,
    ) -> None:
        compiled = controller.compiled
        self.names, self.rate = live_matching(compiled, params), rate
        self._params = {n: compiled.params[n] for n in self.names}
        self._slices = {n: compiled.live_slices()[n] for n in self.names}
        self._stiffness = TaskStiffness(controller, site, normal, robot)
        n_angles, n_rates = compiled.n_motors
        offset = n_angles + n_rates + compiled.z0.size  # where the live Params start in x
        self._columns = np.concatenate([np.arange(s.start, s.stop) for s in self._slices.values()])
        x, f = ca.SX.sym("x", compiled.fast.size1_in(0)), ca.SX.sym("f", 3)
        K = ca.vec(self._stiffness.function(x, f))
        jacobian = ca.jacobian(K, x[(self._columns + offset).tolist()])
        self._affine = ca.Function("affine", [x, f], [K, jacobian])
        self._offset = offset

    def target(
        self, controller: Any, K_des: ArrayLike, f_ext: ArrayLike | None = None
    ) -> dict[str, np.ndarray]:
        """The Params, flat, that give ``K_des`` (3 by 3) at the contact force ``f_ext``."""
        x = controller.inputs()
        f = self._stiffness.force(controller) if f_ext is None else np.ravel(f_ext)
        now, B = (np.array(m) for m in self._affine(x, f))
        wanted = np.ravel(np.asarray(K_des, dtype=float), order="F")
        p = x[self._columns + self._offset]
        solution = np.linalg.pinv(B, rcond=NOISE) @ (wanted - now.ravel() + B @ p)
        out, k = {}, 0
        for name, s in self._slices.items():
            out[name] = solution[k : k + s.stop - s.start]
            k += s.stop - s.start
        return out

    def step(self, controller: Any, K_des: ArrayLike, f_ext: ArrayLike | None = None) -> float:
        """One step on ``controller`` (or a ``Tank`` around it); returns the energy jump [J]."""
        goal = self.target(controller, K_des, f_ext)
        live = controller.live_params()
        new = {}
        for name, aim in goal.items():
            here = np.ravel(live[name], order="F")
            new[name] = admissible(self._params[name], here + self.rate * (aim - here))
        return float(controller.set(new))
