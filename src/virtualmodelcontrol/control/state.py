"""StateController: a controller that reads the robot's state, not its motors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np

from ..core.params import constants
from ..core.signals import Signals
from .controller import VMCController


def _from_signals(meas: Signals) -> tuple[np.ndarray, np.ndarray]:
    return meas["q"], meas["v"]


class StateController(VMCController):
    """A ``VMCController`` that evaluates the law at the robot's state (q, v), not at the state its
    motors give.

    ``read(meas)`` returns (q, v) of a measurement; by default ``meas["q"]`` and ``meas["v"]``,
    which a simulated plant reports and an estimator can supply for the coordinates no sensor
    sees. On a robot with motors on every coordinate it is the same controller. The laws and
    estimators that read a controller's inputs in the layout of the motors (``ForceTracking``,
    ``StiffnessTracking``, ``PositionRegulation``, ``HoldingGoals``, ``ContactForce``,
    ``TaskStiffness``) refuse it with a ``ValueError``.
    """

    def __init__(
        self,
        compiled: Any,
        output: list[Any] | None = None,
        read: Callable[[Signals], tuple[Any, Any]] = _from_signals,
    ) -> None:
        super().__init__(compiled, output)
        self.read = read
        space = compiled.system.robot.model.space
        actuation = compiled.system.actuation
        sizes = [space.nq, space.nv, compiled.z0.size, compiled.live_values().size, 1]
        x = ca.SX.sym("x", sum(sizes))
        q, v, z, p, t = ca.vertsplit(x, np.cumsum([0, *sizes]).tolist())
        args = [q, v, z, p, t]
        u, zdot = compiled.law(*args)
        V, T = compiled.energy(*args)
        port, dissipation, source = compiled.power(*args)
        elements = compiled.forces.call(args)
        pa = constants(actuation.params)
        for k in range(3, len(elements), 4):  # each element's torque, as motor torques
            elements[k] = actuation.allocate(elements[k], q, pa)
        self._fast = ca.Function("fast", [x], [ca.vertcat(u, zdot)])
        self._fast_energy = ca.Function("fast_energy", [x], [V + T])
        self._fast_power = ca.Function("fast_power", [x], [V, T, port, dissipation, source])
        self._fast_elements = ca.Function("fast_elements", [x], elements)

    def _pack(self, meas: Signals, t: float) -> np.ndarray:
        elapsed = t - (t if self.t0 is None else self.t0)
        q, v = self.read(meas)
        return np.concatenate([q, v, self.z, self.params, [elapsed]])


def require_motor_layout(controller: Any, what: str) -> None:
    """Refuse a ``StateController`` (also inside a ``Tank``): ``what`` reads the controller's
    inputs in the layout of the motors, and this controller's are the state (q, v)."""
    while hasattr(controller, "controller"):  # a Tank around the controller
        controller = controller.controller
    if isinstance(controller, StateController):
        raise ValueError(
            f"{what} reads the motors of a controller, and a StateController reads the state "
            "(q, v) of the robot. For a robot with fewer motors than coordinates, use "
            "control.underactuated.controller(compiled, 'frozen') without gravity=True, which "
            "reads the motors, or control.underactuated.DirectionalForce for a force"
        )
