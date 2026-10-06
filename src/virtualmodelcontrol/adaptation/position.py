"""Positions of the robot's points by the goals of springs: integral action and feedforward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..estimation.balance import balance
from ..models.kinematics import Kinematics
from .limits import NOISE, admissible, live_matching


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


class HoldingGoals:
    """The goals of springs that hold points of the robot at their targets, with no feedback.

    ``sites`` and ``robot`` are as for ``PositionRegulation`` and ``ContactForce``. The goals are
    put at the targets plus the smallest offset that lets the springs carry the robot's own
    stiffness and weight at its pose now: one Newton step on the static balance.
    """

    def __init__(self, controller: Any, sites: Mapping[str, Any], robot: Any = None) -> None:
        compiled = controller.compiled
        for name in sites:
            live_matching(compiled, name)  # refuses a Param that is not live
        self._sites = dict(sites)
        self._params = {n: compiled.params[n] for n in sites}
        self._kin = Kinematics(compiled.system.robot, coordinates="motors")
        n_angles, n_rates = compiled.n_motors
        self._motors = n_angles
        start = n_angles + n_rates + compiled.z0.size  # where the live Params start in x
        slices = compiled.live_slices()
        self._columns = {n: np.arange(slices[n].start, slices[n].stop) + start for n in sites}
        x, _, held = balance(controller, robot)
        goals = np.concatenate(list(self._columns.values())).tolist()
        self._terms = ca.Function("terms", [x], [held, ca.jacobian(held, x[goals])])

    def target(
        self, controller: Any, targets: Mapping[str, ArrayLike] | None = None
    ) -> dict[str, np.ndarray]:
        """The goals, flat, that hold the points at ``targets`` (default: where they are)."""
        x = np.array(controller.inputs(), dtype=float)
        here = {n: self._kin.position(x[: self._motors], s) for n, s in self._sites.items()}
        for name, point in here.items():
            x[self._columns[name]] = point  # a spring at rest: the force left is the robot's own
        held, B = (np.array(m) for m in self._terms(x))
        offset = -np.linalg.pinv(B, rcond=NOISE) @ held.ravel()
        out, k = {}, 0
        for name, columns in self._columns.items():
            wanted = here[name] if targets is None or name not in targets else targets[name]
            out[name] = np.ravel(wanted, order="F") + offset[k : k + columns.size]
            k += columns.size
        return out

    def step(self, controller: Any, targets: Mapping[str, ArrayLike] | None = None) -> float:
        """Set the goals on ``controller`` (or a ``Tank``); returns the energy jump [J]."""
        goals = self.target(controller, targets)
        return float(controller.set({n: admissible(self._params[n], g) for n, g in goals.items()}))
