"""Equilibrium: the closed loop at rest, as a static problem."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from .builder import Builder
from .trajectory import Trajectory


class Equilibrium:
    """A configuration where the robot is at rest under the controller: the torques of the
    controller balance the robot's own forces. The configuration is an unknown, found with the
    free Params, from the guess ``q0``.

    It stands in for a ``Collocation``, as a motion of one node: the terms see it as their only
    node, and ``Cost`` and ``Effort`` count it once. A ``Bound`` holds there too, as the node is
    not a fixed start. ``scale`` is the typical size of q.
    """

    name = "equilibrium"
    motion = True

    def __init__(self, q0: ArrayLike, *, scale: float = 1.0) -> None:
        if scale <= 0.0:
            raise ValueError("the scale of q must be positive")
        self.q0, self.scale = np.asarray(q0, dtype=float).ravel(), float(scale)

    def coordinates(self) -> tuple[Any, ...]:
        """No task coordinates."""
        return ()

    def build(self, builder: Builder) -> None:
        """Add the configuration, and the balance of the closed loop there."""
        space = builder.system.robot.model.space
        nq, nv = space.nq, space.nv
        if self.q0.size != nq:
            raise ValueError(f"q0 needs {nq} entries, got {self.q0.size}")
        inf = np.inf
        q = builder.variables.add("q", nq, -inf, inf, self.q0, self.scale)
        v = builder.variables.add("v", nv, 0.0, 0.0, np.zeros(nv), 1.0)  # at rest: fixed at zero
        a = builder.variables.add("a", nv, 0.0, 0.0, np.zeros(nv), 1.0)
        compiled, dynamics = builder.compiled, builder.dynamics
        p_law = builder.pack(compiled.params, compiled.live)
        p_dyn = builder.pack(dynamics.params, dynamics.live)
        u = compiled.law(q, v, ca.DM.zeros(0, 1), p_law, 0.0)[0]
        builder.constrain("equilibrium", dynamics.residual(q, v, a, u, p_dyn, 0.0), 0.0, 0.0)
        builder.output("u", u)
        builder.trajectory = Trajectory(
            t=np.zeros(1),
            dt=2.0,  # one node counts once: the sums give the end nodes half of dt
            q=[q],
            v=[v],
            a=[a],
            u=[u],
            blend=np.ones(1),
            shapes={"q": (1, nq), "v": (1, nv), "a": (1, nv)},
            evaluate=builder.evaluate,
            fixed_start=False,
        )
