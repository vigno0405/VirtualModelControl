"""A model from a function: the rotation and position of each site, written by the user."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import casadi as ca
import numpy as np

from ..core.params import Param, ParamSet
from ..core.space import Euclidean, Space
from .equations import Equations
from .kinematic import KinematicModel


def _matrix(x: Any, rows: int, cols: int) -> Any:
    """``x`` as a CasADi matrix of the given shape: CasADi values stay, lists (of CasADi values
    and numbers) and arrays are assembled."""
    if isinstance(x, (ca.SX, ca.MX, ca.DM)):
        return ca.reshape(x, rows, cols)
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):
            return ca.vertcat(*[ca.horzcat(*row) for row in x])
        return ca.reshape(ca.vertcat(*x), rows, cols)
    return ca.DM(np.asarray(x, dtype=float).reshape(rows, cols))


class FunctionModel(KinematicModel):
    """A kinematic model from ``frame(q, at, p)``, which returns the rotation (3, 3) and the
    position (3,) of the site or arc parameter ``at``, written with CasADi operations (smooth in
    ``q``, in the Params and in ``at``). ``p`` maps each Param's name to its value.

    ``space`` is a configuration space, or the number of joints of a flat one. ``params`` are
    the geometric numbers (each a ``Param``), ``sites`` the named points, ``q_unit`` the unit of
    ``q``. ``equations`` gives the robot's dynamics from a function (``Equations``) in place of
    inertances. A function cannot be written to a file, so a function model has no ``to_dict``.
    """

    def __init__(
        self,
        frame: Callable[[Any, Any, dict[str, Any]], tuple[Any, Any]],
        space: Space | int,
        params: Iterable[Param] = (),
        sites: Iterable[str] = (),
        q_unit: str = "rad",
        equations: Equations | None = None,
    ) -> None:
        self._frame = frame
        self.equations = equations
        self._space: Space = Euclidean(space) if isinstance(space, int) else space
        self._params = params if isinstance(params, ParamSet) else ParamSet(params)
        self._sites = tuple(sites)
        self.q_unit = q_unit

    @property
    def space(self) -> Space:
        """The configuration space."""
        return self._space

    @property
    def params(self) -> ParamSet:
        """The geometric numbers."""
        return self._params

    @property
    def sites(self) -> tuple[str, ...]:
        """The named points."""
        return self._sites

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """Rotation (3, 3) and position (3, 1) at ``at``, from the function."""
        R, position = self._frame(q, at, p)
        return _matrix(R, 3, 3), _matrix(position, 3, 1)
