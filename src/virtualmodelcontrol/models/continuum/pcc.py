"""Piecewise constant curvature (PCC) kinematics of a continuum robot with n segments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np

from ...core.params import ParamSet, as_param
from ...core.registry import register
from ...core.space import Euclidean
from ...core.units import M


def segment_frame(dx: Any, dy: Any, dl: Any, L0: Any, d: Any, s: Any, eps: float) -> tuple:
    """(R, t) of one PCC segment at local arc fraction s ∈ [0, 1].

    The segment is an arc of radius d (L0 + Dl) / D, D = √(Dx² + Dy² + ε), bent by s D / d
    about the axis (−Dy, Dx, 0) / D. Its tangent is R's z axis.
    """
    D = ca.sqrt(dx**2 + dy**2 + eps)
    theta = s * D / d
    c, sn = ca.cos(theta), ca.sin(theta)
    R = ca.vertcat(
        ca.horzcat(1 + dx**2 / D**2 * (c - 1), dx * dy / D**2 * (c - 1), dx / D * sn),
        ca.horzcat(dx * dy / D**2 * (c - 1), 1 + dy**2 / D**2 * (c - 1), dy / D * sn),
        ca.horzcat(-dx / D * sn, -dy / D * sn, c),
    )
    k = d * (L0 + dl) / D**2
    t = ca.vertcat(k * dx * (1 - c), k * dy * (1 - c), k * D * sn)
    return R, t


@register("model", "pcc")
class PCC:
    """Continuum robot of n PCC segments; q = Δ = [Dx, Dy, Dl] per segment, base to tip [m].

    Per segment i (from 1): ``seg{i}.L0`` rest length and ``seg{i}.d`` section radius [m],
    ``design`` Params. The arc parameter s ∈ [0, 1] is uniform in arc length (the breakpoints
    follow from the rest lengths). Sites: ``base``, ``seg{i}`` (end of segment i), ``tip``.
    """

    q_unit = M

    def __init__(self, L0: Sequence[Any], d: Any, *, eps: float = 1e-12) -> None:
        n = len(L0)
        d_list = list(d) if isinstance(d, (list, tuple, np.ndarray)) else [d] * n
        if len(d_list) != n:
            raise ValueError(f"{n} rest lengths but {len(d_list)} section radii")
        self.n_segments = n
        self.space = Euclidean(3 * n)
        self.eps = eps
        self.params = ParamSet()
        for i in range(n):
            for name, value in (("L0", L0[i]), ("d", d_list[i])):
                key = f"seg{i + 1}.{name}"
                param = as_param(value, key, unit=M, bounds=(0.0, np.inf), scope="design")
                self.params.add(param, key)
        self.sites = ("base", *(f"seg{i + 1}" for i in range(n)), "tip")

    def frame(self, q: Any, at: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """(R, position) at a site or at arc parameter s (clipped to [0, 1])."""
        n = self.n_segments
        L0 = [p[f"seg{i + 1}.L0"] for i in range(n)]
        d = [p[f"seg{i + 1}.d"] for i in range(n)]
        seg = [q[3 * i : 3 * i + 3] for i in range(n)]

        def local(i: int, s: Any) -> tuple:
            return segment_frame(seg[i][0], seg[i][1], seg[i][2], L0[i], d[i], s, self.eps)

        if isinstance(at, str):
            if at not in self.sites:
                raise KeyError(f"unknown site {at!r}; sites: {self.sites}")
            count = n if at == "tip" else 0 if at == "base" else int(at[3:])
            R, pos = ca.DM.eye(3), ca.DM.zeros(3, 1)
            for i in range(count):
                Ri, ti = local(i, 1.0)
                R, pos = ca.mtimes(R, Ri), pos + ca.mtimes(R, ti)
            return R, pos

        cum = [0]
        for i in range(n):
            cum.append(cum[-1] + L0[i])
        b = [c / cum[-1] for c in cum]  # breakpoints of s at the segment ends
        s = ca.fmin(ca.fmax(at, 0.0), 1.0)
        R_base, p_base = ca.DM.eye(3), ca.DM.zeros(3, 1)
        candidates = []
        for i in range(n):
            Ri, ti = local(i, (s - b[i]) / (b[i + 1] - b[i]))
            candidates.append((ca.mtimes(R_base, Ri), p_base + ca.mtimes(R_base, ti)))
            Re, te = local(i, 1.0)
            R_base, p_base = ca.mtimes(R_base, Re), p_base + ca.mtimes(R_base, te)
        R, pos = candidates[-1]
        for i in reversed(range(n - 1)):
            inside = s <= b[i + 1]
            R = ca.if_else(inside, candidates[i][0], R)
            pos = ca.if_else(inside, candidates[i][1], pos)
        return R, pos

    def breakpoints(self) -> np.ndarray:
        """Arc parameter at the base and at each segment end, at the current rest lengths."""
        L0 = np.array([float(self.params[f"seg{i + 1}.L0"].value) for i in range(self.n_segments)])
        return np.concatenate([[0.0], np.cumsum(L0)]) / L0.sum()

    def to_dict(self) -> dict[str, Any]:
        """Constructor arguments at the current Param values."""
        n = range(1, self.n_segments + 1)
        return {
            "type": "pcc",
            "L0": [float(self.params[f"seg{i}.L0"].value) for i in n],
            "d": [float(self.params[f"seg{i}.d"].value) for i in n],
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PCC:
        """Inverse of ``to_dict``."""
        return cls(data["L0"], data["d"], eps=data.get("eps", 1e-12))
