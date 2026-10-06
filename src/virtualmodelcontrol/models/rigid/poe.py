"""Serial chains by the product of exponentials, with all geometry as Params."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import casadi as ca
import numpy as np

from ...core.params import ParamSet, as_param
from ...core.registry import register
from ...core.space import Euclidean, Product, Quaternion
from ...core.units import M
from ...math import exp_so3, quat_rot, rot
from .spline import spline_point

JOINT_TYPES = ("revolute", "prismatic", "helical", "spherical", "free", "rail", "floating")
COORDINATES = {
    "revolute": 1,
    "prismatic": 1,
    "helical": 1,
    "spherical": 3,
    "free": 6,
    "rail": 1,
    "floating": 7,
}
"""How many entries of q each joint type owns (a floating joint has 6 velocities)."""


def chain_space(joints: Sequence[str]) -> Any:
    """The space of a chain: its joints in order, a floating one as a translation and a unit
    quaternion, the others as plain numbers."""
    spaces: list[Any] = []
    for kind in joints:
        flat = Euclidean(COORDINATES[kind])
        for part in [Euclidean(3), Quaternion()] if kind == "floating" else [flat]:
            if isinstance(part, Euclidean) and spaces and isinstance(spaces[-1], Euclidean):
                spaces[-1] = Euclidean(spaces[-1].nq + part.nq)
            else:
                spaces.append(part)
    return spaces[0] if len(spaces) == 1 else Product(*spaces)


@register("model", "poe")
class SerialChain:
    """Serial chain of joints, as a product of exponentials.

    A joint is ``"revolute"`` or ``"prismatic"`` (one coordinate), ``"helical"`` (one, a turn that
    also slides along its axis by ``pitch`` [m/rad]; give it as ``("helical", pitch)``),
    ``"spherical"`` (three, a rotation vector about a point), ``"free"`` (six, a translation and
    a rotation vector: a floating base), ``"floating"`` (seven, a translation and a unit
    quaternion, with the body's angular velocity as its three rates: a floating base that can turn
    any number of times) or ``"rail"`` (one, a slide along the natural cubic
    spline through the rows of ``j{i}.waypoints`` [m]; give it as ``("rail", waypoints)``; its
    coordinate is the spline parameter, 0 at the first waypoint and 1 at the last, and its
    waypoints are in the frame of the joint before it). Joint i (from 1) has ``j{i}.axis``
    (direction at q = 0, normalized internally; ``None`` for spherical, free, floating and rail) and
    ``j{i}.point`` [m], a point on its axis, or the point a rotation turns about. A site
    ``(after, position[, rotation])`` has ``{name}.position`` [m] and ``{name}.rotation``
    (rotation vector [rad]) at q = 0 and moves with the joints up to ``after`` (1-based). All are
    ``design`` Params.
    """

    def __init__(
        self,
        joints: Sequence[str | Sequence[Any]],
        axes: Sequence[Any],
        points: Sequence[Any],
        sites: Mapping[str, tuple[Any, ...]],
    ) -> None:
        specs = [(j, None) if isinstance(j, str) else (j[0], j[1]) for j in joints]
        if any(kind not in JOINT_TYPES for kind, _ in specs):
            raise ValueError(f"joint types must be in {JOINT_TYPES}, got {list(joints)}")
        if not len(joints) == len(axes) == len(points):
            raise ValueError("give one axis and one point per joint")
        self.joints = [kind for kind, _ in specs]
        sizes = [COORDINATES[kind] for kind in self.joints]
        self._start = np.cumsum([0, *sizes]).tolist()
        self.space = chain_space(self.joints)
        self.params = ParamSet()
        free = (-np.inf, np.inf)
        for i, ((kind, extra), axis, point) in enumerate(zip(specs, axes, points, strict=True)):
            if axis is not None:
                key = f"j{i + 1}.axis"
                self.params.add(as_param(axis, key, scope="design", bounds=free), key)
            key = f"j{i + 1}.point"
            self.params.add(as_param(point, key, unit=M, scope="design", bounds=free), key)
            if kind == "helical":
                key = f"j{i + 1}.pitch"
                self.params.add(
                    as_param(extra, key, unit="m/rad", scope="design", bounds=free), key
                )
            if kind == "rail":
                key = f"j{i + 1}.waypoints"
                if extra is None:
                    raise ValueError("give a rail as ('rail', waypoints)")
                waypoints = as_param(extra, key, unit=M, scope="design", bounds=free)
                if len(waypoints.shape) != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] < 2:
                    raise ValueError("a rail needs two or more waypoints, one row [x, y, z] each")
                self.params.add(waypoints, key)
        self.site_joint: dict[str, int] = {}
        for name, spec in sites.items():
            after, position = spec[0], spec[1]
            rotation = spec[2] if len(spec) > 2 else (0.0, 0.0, 0.0)
            key = f"{name}.position"
            self.params.add(as_param(position, key, unit=M, scope="design", bounds=free), key)
            key = f"{name}.rotation"
            self.params.add(as_param(rotation, key, unit="rad", scope="design", bounds=free), key)
            self.site_joint[name] = int(after)
        self.sites = tuple(sites)

    @property
    def q_unit(self) -> str:
        """Unit of the joint coordinates (rad, m if all prismatic, none if all rails)."""
        if all(kind == "prismatic" for kind in self.joints):
            return M
        return "" if all(kind == "rail" for kind in self.joints) else "rad"

    def frame(self, q: Any, at: str, p: dict[str, Any]) -> tuple[Any, Any]:
        """(R, position) of a site."""
        if at not in self.site_joint:
            raise KeyError(f"unknown site {at!r}; sites: {self.sites}")
        R, pos = ca.DM.eye(3), ca.DM.zeros(3, 1)
        for i in range(self.site_joint[at]):
            Ri, ti = self._joint(i, q, p)
            R, pos = ca.mtimes(R, Ri), pos + ca.mtimes(R, ti)
        R_site = exp_so3(ca.reshape(p[f"{at}.rotation"], 3, 1))
        return ca.mtimes(R, R_site), pos + ca.mtimes(R, ca.reshape(p[f"{at}.position"], 3, 1))

    def _joint(self, i: int, q: Any, p: dict[str, Any]) -> tuple[Any, Any]:
        """Rotation and translation of joint ``i`` (from 0) at its own coordinates."""
        kind, name = self.joints[i], f"j{i + 1}"
        mine = q[self._start[i] : self._start[i + 1]]
        eye, point = ca.DM.eye(3), ca.reshape(p[f"{name}.point"], 3, 1)
        if kind in ("spherical", "free", "floating"):
            if kind == "floating":
                R = quat_rot(mine[3:])
            else:
                R = exp_so3(mine[3:] if kind == "free" else mine)
            t = ca.mtimes(eye - R, point)  # the turn about ``point``
            if kind != "spherical":
                t = t + mine[:3]
            return R, t
        if kind == "rail":
            waypoints = self.params[f"{name}.waypoints"]
            rows = ca.reshape(p[f"{name}.waypoints"], waypoints.shape[0], 3)
            u = mine[0] * (waypoints.shape[0] - 1)
            return eye, spline_point(rows, u) - rows[0, :].T
        axis = ca.reshape(p[f"{name}.axis"], 3, 1)
        axis = axis / ca.norm_2(axis)
        if kind == "prismatic":
            return eye, axis * mine[0]
        R = rot(axis, mine[0])
        t = ca.mtimes(eye - R, point)
        if kind == "helical":
            t = t + axis * p[f"{name}.pitch"] * mine[0]
        return R, t

    def joint_points(self, q: Any, p: dict[str, Any]) -> list[Any]:
        """Positions of the joints' axis points at q, base to tip (for drawing)."""
        R, pos = ca.DM.eye(3), ca.DM.zeros(3, 1)
        out = []
        for i in range(len(self.joints)):
            out.append(pos + ca.mtimes(R, ca.reshape(p[f"j{i + 1}.point"], 3, 1)))
            Ri, ti = self._joint(i, q, p)
            R, pos = ca.mtimes(R, Ri), pos + ca.mtimes(R, ti)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Constructor arguments at the current Param values."""
        n = range(1, len(self.joints) + 1)
        return {
            "type": "poe",
            "joints": [
                [kind, self.params[f"j{i}.pitch"].value.item()]
                if kind == "helical"
                else [kind, self.params[f"j{i}.waypoints"].value.tolist()]
                if kind == "rail"
                else kind
                for i, kind in zip(n, self.joints, strict=True)
            ],
            "axes": [
                self.params[f"j{i}.axis"].value.tolist() if f"j{i}.axis" in self.params else None
                for i in n
            ],
            "points": [self.params[f"j{i}.point"].value.tolist() for i in n],
            "sites": {
                name: [
                    after,
                    self.params[f"{name}.position"].value.tolist(),
                    self.params[f"{name}.rotation"].value.tolist(),
                ]
                for name, after in self.site_joint.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerialChain:
        """Inverse of ``to_dict``."""
        sites = {name: tuple(spec) for name, spec in data["sites"].items()}
        return cls(data["joints"], data["axes"], data["points"], sites)

    @classmethod
    def from_dh(
        cls,
        d: Any,
        a: Any,
        alpha: Any,
        offset: Any = None,
        tool: str = "tool",
    ) -> SerialChain:
        """Revolute chain from a standard Denavit–Hartenberg table, with the last frame as a site.

        Frame i = frame i−1 · Rz(θ_i + offset_i) Tz(d_i) Tx(a_i) Rx(α_i) [m, rad]; joint i turns
        about the z axis of frame i−1.
        """
        from scipy.spatial.transform import Rotation

        d, a, alpha = (np.asarray(x, dtype=float) for x in (d, a, alpha))
        offset = np.zeros(len(d)) if offset is None else np.asarray(offset, dtype=float)
        T = np.eye(4)
        axes, points = [], []
        for i in range(len(d)):
            axes.append(T[:3, 2].copy())
            points.append(T[:3, 3].copy())
            ct, st = np.cos(offset[i]), np.sin(offset[i])
            ca_, sa = np.cos(alpha[i]), np.sin(alpha[i])
            A = np.array([
                [ct, -st * ca_, st * sa, a[i] * ct],
                [st, ct * ca_, -ct * sa, a[i] * st],
                [0.0, sa, ca_, d[i]],
                [0.0, 0.0, 0.0, 1.0],
            ])  # fmt: skip
            T = T @ A
        rot = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        sites = {tool: (len(d), T[:3, 3], rot)}
        return cls(["revolute"] * len(d), axes, points, sites)
