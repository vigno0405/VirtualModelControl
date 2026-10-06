"""Integral regulation of positions: the goals of springs follow the error of their points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..models.kinematics import Kinematics
from .limits import admissible, live_matching


class PositionRegulation:
    """Moves the goals of springs to bring points of the robot to targets, by integral action.

    ``sites`` maps the name of each live goal Param to the site its spring pulls. A step adds
    ``gain`` times the position error to each goal, whatever holds the robot back.
    """

    def __init__(self, controller: Any, sites: Mapping[str, Any], gain: float = 0.05) -> None:
        compiled = controller.compiled
        for name in sites:
            live_matching(compiled, name)  # refuses a Param that is not live
        self.names, self.gain, self._sites = list(sites), gain, dict(sites)
        self._params = {n: compiled.params[n] for n in self.names}
        self._kin = Kinematics(compiled.system.robot, coordinates="motors")
        self._motors = compiled.n_motors[0]

    def step(self, controller: Any, targets: Mapping[str, ArrayLike]) -> float:
        """One step on ``controller`` (or a ``Tank`` around it), after its last ``step``.

        ``targets`` maps the same Param names to the wanted positions [m]; returns the jump of the
        controller's energy [J].
        """
        theta = controller.inputs()[: self._motors]
        live = controller.live_params()
        new = {}
        for name, site in self._sites.items():
            error = np.asarray(targets[name], dtype=float) - self._kin.position(theta, site)
            new[name] = admissible(
                self._params[name], np.ravel(live[name], order="F") + self.gain * error
            )
        return float(controller.set(new))
