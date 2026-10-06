"""VMCController: runs a compiled virtual mechanism step by step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.signals import Signals
from .output import apply


class VMCController:
    """Runs a compiled system: motor angles and rates in, motor torques out.

    ``step`` reads ``motor_position`` [rad] and ``motor_velocity`` [rad/s] and returns
    ``motor_torque`` [N·m]. Virtual states are integrated by semi-implicit Euler over the measured
    dt. Time in the law counts from ``reset`` (or the first step), so ramps start with the
    controller. Live Params start at their values when compiled; ``set`` changes them here only.
    ``output`` lists optional output stages (``control.output``) applied in order; the returned
    Signals also hold ``law_torque``, the torque before them. An output stage with a ``reset``
    is reset with the controller.
    """

    def __init__(self, compiled: Any, output: list[Any] | None = None) -> None:
        self.compiled = compiled
        self.output = list(output or [])
        self._fast = compiled.fast
        self._fast_energy = compiled.fast_energy
        self._fast_power = compiled.fast_power
        self._fast_elements = compiled.fast_elements
        self.params = compiled.live_values()
        self._slices = compiled.live_slices()
        self.z = compiled.z0.copy()
        self.t: float | None = None
        self.t0: float | None = None
        self._x: np.ndarray | None = None

    def reset(self, t: float, meas: Signals | None = None, z0: ArrayLike | None = None) -> None:
        """Restart at time ``t`` [s] from the initial virtual state (or ``z0``)."""
        self.z = self.compiled.z0.copy() if z0 is None else np.array(z0, dtype=float)
        for stage in self.output:
            if hasattr(stage, "reset"):
                stage.reset()
        self.t = self.t0 = t
        self._x = None if meas is None else self._pack(meas, t)

    def step(self, t: float, meas: Signals) -> Signals:
        """One control step at time ``t`` [s]; returns the motor torques."""
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.t is None else t - self.t
        x = self._pack(meas, t)
        out = np.asarray(self._fast(x)).ravel()
        nu, nz = self.compiled.n_u, self.z.size // 2
        if nz:
            self.z[nz:] += dt * out[nu + nz :]
            self.z[:nz] += dt * self.z[nz:]
        self.t, self._x = t, x
        u = out[:nu]
        return Signals(t, motor_torque=apply(self.output, u, meas), law_torque=u)

    def set(self, values: Mapping[str, ArrayLike] | None = None, **kwargs: ArrayLike) -> float:
        """Change live Params at once; returns the exact jump of the controller's energy [J]."""
        new = self._changed(values, kwargs)
        jump = self._jump(new)
        self.params = new
        return jump

    def jump(self, values: Mapping[str, ArrayLike] | None = None, **kwargs: ArrayLike) -> float:
        """The jump of the controller's energy [J] that ``set`` would give, changing nothing."""
        return self._jump(self._changed(values, kwargs))

    def inputs(self) -> np.ndarray | None:
        """What the law reads now: the motor angles and rates and the virtual state of the last
        step, the live Params as they are, and the time, packed in one vector (``None`` before the
        first step)."""
        if self._x is None:
            return None
        x = self._x.copy()
        x[x.size - 1 - self.params.size : -1] = self.params
        return x

    def energy(self) -> float:
        """Energy of the controller at the last step [J] (stored plus virtual kinetic)."""
        return 0.0 if self._x is None else self._energy(self.params)

    def live_params(self) -> dict[str, np.ndarray]:
        """The controller's own values of its live Params, each in the Param's shape."""
        return {
            name: np.reshape(self.params[where], self.compiled.params[name].shape, order="F")
            for name, where in self.compiled.live_slices().items()
        }

    def balance(self) -> dict[str, float]:
        """The controller's energies [J] and powers [W] at the last step: ``stored`` (V),
        ``kinetic`` (T), ``port`` (τᵀv), ``dissipation`` and ``source``."""
        if self._x is None:
            return {}
        values = self._fast_power.call([self._x])
        names = ("stored", "kinetic", "port", "dissipation", "source")
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def elements(self) -> dict[str, dict[str, np.ndarray]]:
        """Each element at the last step: its coordinate ``y``, rate ``ydot``, ``force`` and
        ``torque``, its share of the motor torques before any output stage."""
        if self._x is None:
            return {}
        values = [np.array(v).ravel() for v in self._fast_elements.call([self._x])]
        quantities = ("y", "ydot", "force", "torque")
        return {
            name: dict(zip(quantities, values[4 * k : 4 * k + 4], strict=True))
            for k, name in enumerate(self.compiled.component_names)
        }

    def _changed(self, values: Mapping[str, ArrayLike] | None, kwargs: Mapping[str, Any]) -> Any:
        new = self.params.copy()
        for name, value in {**(values or {}), **kwargs}.items():
            if name not in self._slices:
                raise KeyError(
                    f"{name!r} is not a live Param here; compile with runtime=[{name!r}] to change "
                    "it while running"
                )
            new[self._slices[name]] = np.ravel(value, order="F")
        return new

    def _jump(self, new: np.ndarray) -> float:
        return 0.0 if self._x is None else self._energy(new) - self._energy(self.params)

    def _energy(self, params: np.ndarray) -> float:
        assert self._x is not None
        x = self._x.copy()
        n = self._x.size - 1 - params.size
        x[n:-1] = params
        return float(self._fast_energy(x))

    def _pack(self, meas: Signals, t: float) -> np.ndarray:
        elapsed = t - (t if self.t0 is None else self.t0)
        return np.concatenate(
            [meas["motor_position"], meas["motor_velocity"], self.z, self.params, [elapsed]]
        )
