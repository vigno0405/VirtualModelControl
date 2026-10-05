"""Torque steps for the step experiment: a controller that runs on any plant."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.signals import Signals

Pull = tuple[int | Sequence[int], float]
"""``(motors, torque)``: the motors that pull, by index, and the torque [N·m] they add."""


class Steps:
    """Torque steps on any plant, for ``fit_stiffness_damping``.

    Every motor gets ``baseline`` [N·m] (one value, or one per motor), and each pull in turn adds
    its torque to its motors for ``hold`` s. The run rests at the baseline for ``rest`` s before
    the first pull and after each one. The ``pulls`` train the fit and the ``held_out`` ones, run
    last, check it. The log holds ``train``: 1 while a training step runs, 0 on a held-out one.
    Start from rest at the baseline; ``duration`` is the time to run.
    """

    def __init__(
        self,
        baseline: ArrayLike,
        pulls: Iterable[Pull] = (),
        held_out: Iterable[Pull] = (),
        hold: float = 2.0,
        rest: float = 2.0,
    ) -> None:
        self.baseline = np.asarray(baseline, dtype=float)
        pulls, held_out = list(pulls), list(held_out)
        self.pulls, self.n_train = pulls + held_out, len(pulls)
        self.hold, self.rest = float(hold), float(rest)
        self.duration = self.rest + len(self.pulls) * (self.hold + self.rest)
        self.t0 = 0.0

    def reset(self, t: float, meas: Signals | None = None, z0: Any = None) -> None:
        """Start the script at time ``t`` [s]."""
        self.t0 = t

    def step(self, t: float, meas: Signals) -> Signals:
        """The baseline, plus the pull under way."""
        u = np.broadcast_to(self.baseline, np.shape(meas["motor_velocity"])).copy()
        k, into = divmod(t - self.t0 - self.rest + 1e-9, self.hold + self.rest)
        k = int(k)  # the pull this step belongs to; -1 before the first one
        if 0 <= k < len(self.pulls) and into < self.hold:
            motors, torque = self.pulls[k]
            u[np.atleast_1d(motors)] += torque
        train = min(k, len(self.pulls) - 1) < self.n_train
        return Signals(t, motor_torque=u, train=float(train))
