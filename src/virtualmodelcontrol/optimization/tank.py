"""TankBudget: the planned changes of the controller's Params stay within a tank's energy."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ..mechanisms.coordinates.references import Ref
from .builder import Builder
from .shooting import ShootingTrajectory
from .terms import Term


class TankBudget(Term):
    """The Params that step in a ``Shooting`` are changed within the energy of a ``Tank``.

    Changing them at the start of an interval gives the controller's energy a jump (``Tank.set``
    pays it from the tank, a step that releases energy refills it). The plan tracks the level of
    the tank interval by interval, from ``level`` [J] now (the Param ``tank.level``, a parameter
    of each solve in a receding-horizon controller), and keeps it at least zero. With ``refill``
    the energy that the controller's own dampers take refills it, as ``Tank.step`` does. The
    tank's capacity is not planned.
    """

    def __init__(self, level: float = 0.0, *, refill: bool = True, name: str = "tank") -> None:
        self.level, self.refill, self.name = float(level), refill, name
        self._level = Ref("level", 1, [level], unit="J")

    def coordinates(self) -> tuple[Any, ...]:
        """The level now, as a reference: the Param ``<name>.level``."""
        return (self._level,)

    def build(self, builder: Builder) -> None:
        """Add the level of each interval as a variable, and its balance as constraints."""
        trajectory = builder.trajectory
        if not isinstance(trajectory, ShootingTrajectory):
            raise ValueError(f"{self.name!r} needs a Shooting: the Params must step")
        compiled = builder.compiled
        count = len(trajectory.params)
        start = float(self._level.param.value[0])
        levels = builder.variables.add(f"level:{self.name}", count, 0.0, np.inf, start)
        rows = []
        for k in range(count):
            before = trajectory.now if k == 0 else trajectory.params[k - 1]
            q, v, tk = trajectory.q[k], trajectory.v[k], float(trajectory.t[k])
            z = trajectory.z[k]
            stored = builder.stored(compiled, q, v, z, trajectory.params[k], tk)
            was = builder.stored(compiled, q, v, z, before, tk)
            level = builder.value(self._level.param)
            if k and self.refill:
                level = levels[k - 1] + trajectory.dissipated[k - 1]
            elif k:
                level = levels[k - 1]
            rows.append(levels[k] - (level - (stored - was)))
        builder.constrain(self.name, ca.vertcat(*rows), 0.0, 0.0)
