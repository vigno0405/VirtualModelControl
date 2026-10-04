"""A controller's live Params as ROS parameters, changed between steps."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from rcl_interfaces.msg import SetParametersResult

from ..core.signals import Signals


class LiveParams:
    """Wraps ``controller`` and declares its live Params as parameters of ``node`` (floats or
    float lists, named as in ``controller.set``). A change, for example with ``ros2 param set``,
    reaches the controller at its next step; otherwise this behaves as the controller itself."""

    def __init__(self, node: Any, controller: Any) -> None:
        self.controller = controller
        self._pending: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        for name, where in controller.compiled.live_slices().items():
            value = np.atleast_1d(controller.params[where])
            node.declare_parameter(name, float(value[0]) if value.size == 1 else value.tolist())
        node.add_on_set_parameters_callback(self._on_set)

    def _on_set(self, params: list[Any]) -> SetParametersResult:
        with self._lock:
            for p in params:
                self._pending[p.name] = np.asarray(p.value, dtype=float)
        return SetParametersResult(successful=True)

    def step(self, t: float, meas: Signals) -> Signals:
        """Apply the changes received since the last step, then step the controller."""
        with self._lock:
            pending, self._pending = self._pending, {}
        if pending:
            self.controller.set(pending)
        return self.controller.step(t, meas)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)
