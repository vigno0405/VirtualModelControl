"""Serial chains by the product of exponentials, with all geometry as Params."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import casadi as ca
import numpy as np

from ...core.params import ParamSet, as_param
from ...core.registry import register
from ...core.space import Euclidean
from ...core.symbolic import exp_so3, rotation_from_vector
from ...core.units import M

JOINT_TYPES = ("revolute", "prismatic")


@register("model", "poe")
class SerialChain:
    """Serial chain of revolute and prismatic joints, as a product of exponentials.

    Joint i (from 1) has ``j{i}.axis`` (direction at q = 0, normalized internally) and, if revolute,
    ``j{i}.point`` [m], a point on its axis. A site ``(after, position[, rotation])`` has
    ``{name}.position`` [m] and ``{name}.rotation`` (rotation vector [rad]) at q = 0 and moves with
    the joints up to ``after`` (1-based). All are ``design`` Params.
    """

    def __init__(
        self,
        joints: Sequence[str],
        axes: Sequence[Any],
        points: Sequence[Any],
        sites: Mapping[str, tuple[Any, ...]],
    ) -> None:
        if any(j not in JOINT_TYPES for j in joints):
            raise ValueError(f"joint types must be in {JOINT_TYPES}, got {list(joints)}")
        if not len(joints) == len(axes) == len(points):
            raise ValueError("give one axis and one point per joint")
        self.joints = list(joints)
        self.space = Euclidean(len(joints))
        self.params = ParamSet()
        free = (-np.inf, np.inf)
        for i, (axis, point) in enumerate(zip(axes, points, strict=True)):
            key = f"j{i + 1}.axis"
            self.params.add(as_param(axis, key, scope="design", bounds=free), key)
            key = f"j{i + 1}.point"
            self.params.add(as_param(point, key, unit=M, scope="design", bounds=free), key)
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
        """Unit of the joint coordinates (rad if any joint is revolute)."""
        return "rad" if "revolute" in self.joints else M

    def frame(self, q: Any, at: str, p: dict[str, Any]) -> tuple[Any, Any]:
        """(R, position) of a site."""
        if at not in self.site_joint:
            raise KeyError(f"unknown site {at!r}; sites: {self.sites}")
        R, pos = ca.DM.eye(3), ca.DM.zeros(3, 1)
        for i in range(self.site_joint[at]):
            axis = ca.reshape(p[f"j{i + 1}.axis"], 3, 1)
            axis = axis / ca.norm_2(axis)
            if self.joints[i] == "revolute":
                Ri = exp_so3(axis, q[i])
                ti = ca.mtimes(ca.DM.eye(3) - Ri, ca.reshape(p[f"j{i + 1}.point"], 3, 1))
            else:
                Ri, ti = ca.DM.eye(3), axis * q[i]
            R, pos = ca.mtimes(R, Ri), pos + ca.mtimes(R, ti)
        R_site = rotation_from_vector(ca.reshape(p[f"{at}.rotation"], 3, 1))
        return ca.mtimes(R, R_site), pos + ca.mtimes(R, ca.reshape(p[f"{at}.position"], 3, 1))

    def joint_points(self, q: Any, p: dict[str, Any]) -> list[Any]:
        """Positions of the joints' axis points at q, base to tip (for drawing)."""
        R, pos = ca.DM.eye(3), ca.DM.zeros(3, 1)
        out = []
        for i in range(len(self.joints)):
            point = ca.reshape(p[f"j{i + 1}.point"], 3, 1)
            out.append(pos + ca.mtimes(R, point))
            axis = ca.reshape(p[f"j{i + 1}.axis"], 3, 1)
            axis = axis / ca.norm_2(axis)
            if self.joints[i] == "revolute":
                Ri = exp_so3(axis, q[i])
                ti = ca.mtimes(ca.DM.eye(3) - Ri, point)
            else:
                Ri, ti = ca.DM.eye(3), axis * q[i]
            R, pos = ca.mtimes(R, Ri), pos + ca.mtimes(R, ti)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Constructor arguments at the current Param values."""
        n = range(1, len(self.joints) + 1)
        return {
            "type": "poe",
            "joints": list(self.joints),
            "axes": [self.params[f"j{i}.axis"].value.tolist() for i in n],
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
