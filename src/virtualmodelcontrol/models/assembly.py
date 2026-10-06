"""Assemblies: several models mounted on one base (fingers on a palm, two arms on a frame)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import casadi as ca
import numpy as np

from ..core.params import Param, ParamSet, as_param
from ..core.registry import get, register
from ..core.space import Product
from ..math import exp_so3
from .actuation import Direct


@register("model", "assembly")
class Assembly:
    """Several models mounted at fixed poses on one base; q stacks the parts' coordinates.

    ``parts`` maps a name to ``(model, position, rotation[, parent])``: the mount position [m] and
    rotation vector [rad], ``design`` Params named ``{name}.mount.position`` and
    ``{name}.mount.rotation``, in the base frame or, with ``parent`` (``"other/site"``), in that
    frame of another part, so the part moves with it (a hand on an arm's flange). Sites are
    ``"name/site"``; for a continuous part, give ``at="name"`` and an arc parameter s.
    """

    def __init__(self, parts: Mapping[str, tuple[Any, ...]]) -> None:
        self.parts = {name: spec[0] for name, spec in parts.items()}
        self.parents = {name: (spec[3] if len(spec) > 3 else None) for name, spec in parts.items()}
        self.space = Product(*[m.space for m in self.parts.values()])
        self.params = ParamSet()
        self._local: dict[str, dict[str, str]] = {}
        self._mount: dict[str, tuple[Param, Param]] = {}
        free = (-np.inf, np.inf)
        for name, spec in parts.items():
            model, position, rotation = spec[0], spec[1], spec[2]
            self.params.merge(model.params, name)
            self._local[name] = {loc: self.params.name_of(par) for loc, par in model.params.items()}
            pos = as_param(
                position, f"{name}.mount.position", unit="m", scope="design", bounds=free
            )
            rot = as_param(
                rotation, f"{name}.mount.rotation", unit="rad", scope="design", bounds=free
            )
            self.params.add(pos, f"{name}.mount.position")
            self.params.add(rot, f"{name}.mount.rotation")
            self._mount[name] = (pos, rot)
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, model in self.parts.items():
            self._slices[name] = slice(offset, offset + model.space.nq)
            offset += model.space.nq
        self.sites = tuple(f"{n}/{s}" for n, m in self.parts.items() for s in m.sites)
        units = {getattr(m, "q_unit", "") for m in self.parts.values()}
        self.q_unit = units.pop() if len(units) == 1 else ""

    def part_view(self, name: str, p: dict[str, Any]) -> dict[str, Any]:
        """A part's Params under its own names."""
        return {loc: p[full] for loc, full in self._local[name].items()}

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """(R, position) of ``"part/site"``, or of ``(part, s)`` for a continuous part."""
        if isinstance(at, tuple):
            name, where = at
        else:
            name, where = at.split("/", 1)
        if name not in self.parts:
            raise KeyError(f"unknown part {name!r}; parts: {list(self.parts)}")
        R, pos = self.parts[name].frame(q[self._slices[name]], where, self.part_view(name, p))
        R_m, p_m = self.mount(q, name, p)
        return ca.mtimes(R_m, R), p_m + ca.mtimes(R_m, pos)

    def mount(self, q: Any, name: str, p: dict[str, Any]) -> tuple[Any, Any]:
        """Pose of a part's base in the assembly's base frame."""
        R_m = exp_so3(ca.reshape(p[f"{name}.mount.rotation"], 3, 1))
        p_m = ca.reshape(p[f"{name}.mount.position"], 3, 1)
        parent = self.parents[name]
        if parent is None:
            return R_m, p_m
        R_p, p_p = self.frame(q, parent, p)
        return ca.mtimes(R_p, R_m), p_p + ca.mtimes(R_p, p_m)

    def stacked_actuation(self, actuations: Mapping[str, Any] | None = None) -> StackedActuation:
        """One actuation for the whole assembly: each part's own (default ``Direct``)."""
        actuations = actuations or {}
        return StackedActuation(
            [(self.parts[n].space, actuations.get(n, Direct()), n) for n in self.parts]
        )

    def to_dict(self) -> dict[str, Any]:
        """Each part's model and mount pose."""
        return {
            "type": "assembly",
            "parts": {
                n: [
                    m.to_dict(),
                    self._mount[n][0].value.tolist(),
                    self._mount[n][1].value.tolist(),
                    self.parents[n],
                ]
                for n, m in self.parts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assembly:
        """Inverse of ``to_dict``."""
        parts = {
            n: (get("model", spec[0]["type"]).from_dict(spec[0]), *spec[1:])
            for n, spec in data["parts"].items()
        }
        return cls(parts)


class StackedActuation:
    """Each part of an assembly driven by its own actuation; motors are stacked in part order."""

    def __init__(self, parts: Sequence[tuple[Any, Any, str]]) -> None:
        self.parts = list(parts)
        self.params = ParamSet()
        for _space, actuation, name in self.parts:
            self.params.merge(actuation.params, name)
        self._sizes = [(s.nq, s.nv, *a.motor_sizes(s)) for s, a, _ in self.parts]

    def _view(self, i: int, p: dict[str, Any]) -> dict[str, Any]:
        name, actuation = self.parts[i][2], self.parts[i][1]
        return {loc: p[f"{name}.{loc}"] for loc in actuation.params}

    def _split(self, x: Any, which: int) -> list[Any]:
        out, offset = [], 0
        for sizes in self._sizes:
            out.append(x[offset : offset + sizes[which]])
            offset += sizes[which]
        return out

    def motor_sizes(self, space: Any) -> tuple[int, int]:
        """Total motor angles and rates."""
        return sum(s[2] for s in self._sizes), sum(s[3] for s in self._sizes)

    def _map(self, method: str, xs: list[Any], p: dict[str, Any], *extra: list[Any]) -> Any:
        out = []
        for i, (_space, actuation, _name) in enumerate(self.parts):
            args = [x[i] for x in (xs, *extra)]
            out.append(getattr(actuation, method)(*args, self._view(i, p)))
        return ca.vertcat(*out)

    def motor_angles(self, q: Any, p: dict[str, Any]) -> Any:
        """Each part's motor angles, stacked."""
        return self._map("motor_angles", self._split(q, 0), p)

    def motor_rates(self, q: Any, v: Any, p: dict[str, Any]) -> Any:
        """Each part's motor rates, stacked."""
        return self._map("motor_rates", self._split(q, 0), p, self._split(v, 1))

    def generalized_force(self, u: Any, q: Any, p: dict[str, Any]) -> Any:
        """Each part's generalized force, stacked."""
        return self._map("generalized_force", self._split(u, 3), p, self._split(q, 0))

    def allocate(self, tau: Any, q: Any, p: dict[str, Any]) -> Any:
        """Each part's motor torques, stacked."""
        return self._map("allocate", self._split(tau, 1), p, self._split(q, 0))

    def config_from_motors(self, theta: Any, p: dict[str, Any]) -> Any:
        """Each part's configuration, stacked."""
        return self._map("config_from_motors", self._split(theta, 2), p)

    def velocity_from_motors(self, q: Any, theta_dot: Any, p: dict[str, Any]) -> Any:
        """Each part's velocity, stacked."""
        return self._map("velocity_from_motors", self._split(q, 0), p, self._split(theta_dot, 3))
