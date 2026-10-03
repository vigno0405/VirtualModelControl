"""VMCController: runs a compiled virtual mechanism step by step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.signals import Signals


class VMCController:
    """Runs a compiled system: motor angles and rates in, motor torques out.

    ``step`` reads ``motor_position`` [rad] and ``motor_velocity`` [rad/s] and returns
    ``motor_torque`` [N·m]. Virtual states are integrated by semi-implicit Euler over the measured
    dt. Live Params start at their values when compiled; ``set`` changes them here only.
    """

    def __init__(self, compiled: Any) -> None:
        self.compiled = compiled
        self.params = compiled.live_values()
        self._slices = compiled.live_slices()
        self.z = compiled.z0.copy()
        self.t: float | None = None
        self._x: np.ndarray | None = None

    def reset(self, t: float, meas: Signals | None = None, z0: ArrayLike | None = None) -> None:
        """Restart at time ``t`` [s] from the initial virtual state (or ``z0``)."""
        self.z = self.compiled.z0.copy() if z0 is None else np.array(z0, dtype=float)
        self._x = None if meas is None else self._pack(meas, t)
        self.t = t

    def step(self, t: float, meas: Signals) -> Signals:
        """One control step at time ``t`` [s]; returns the motor torques."""
        dt = 0.0 if self.t is None else t - self.t
        x = self._pack(meas, t)
        out = np.asarray(self.compiled.fast(x)).ravel()
        nu, nz = self.compiled.n_u, self.z.size // 2
        if nz:
            self.z[nz:] += dt * out[nu + nz :]
            self.z[:nz] += dt * self.z[nz:]
        self.t, self._x = t, x
        return Signals(t, motor_torque=out[:nu])

    def set(self, values: Mapping[str, ArrayLike] | None = None, **kwargs: ArrayLike) -> float:
        """Change live Params at once; returns the exact jump of the controller's energy [J]."""
        new = self.params.copy()
        for name, value in {**(values or {}), **kwargs}.items():
            if name not in self._slices:
                raise KeyError(
                    f"{name!r} is not a live Param here; compile with runtime=[{name!r}] to change "
                    "it while running"
                )
            new[self._slices[name]] = np.ravel(value, order="F")
        jump = 0.0 if self._x is None else self._energy(new) - self._energy(self.params)
        self.params = new
        return jump

    def energy(self) -> float:
        """Energy of the controller at the last step [J] (stored plus virtual kinetic)."""
        return 0.0 if self._x is None else self._energy(self.params)

    def _energy(self, params: np.ndarray) -> float:
        assert self._x is not None
        x = self._x.copy()
        n = self._x.size - 1 - params.size
        x[n:-1] = params
        return float(self.compiled.fast_energy(x))

    def _pack(self, meas: Signals, t: float) -> np.ndarray:
        return np.concatenate(
            [meas["motor_position"], meas["motor_velocity"], self.z, self.params, [t]]
        )
