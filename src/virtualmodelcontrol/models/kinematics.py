"""Kinematics as numbers: positions, rotations, Jacobians and Hessians of any point of a robot."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..core.params import constants
from .actuation import Direct


def _vee(S: Any) -> Any:
    return ca.vertcat(S[2, 1], S[0, 2], S[1, 0])


def contact_map(J: Any, normal: ArrayLike | None = None) -> Any:
    """The matrix A of f = A τ: the force at a point from the torques on the coordinates its
    Jacobian ``J`` (3, n) is taken in, by the pseudo-inverse. With a ``normal`` (any length) the
    force is the one along it. Works on CasADi symbols and numbers."""
    if normal is None:
        return ca.solve(ca.mtimes(J, J.T), J)
    n = ca.DM(np.asarray(normal, dtype=float).ravel())
    j = ca.mtimes(J.T, n)
    return ca.mtimes(n, j.T) / ca.dot(j, j)


class Kinematics:
    """Positions, rotations, Jacobians and Hessians of a robot's points, exact by automatic
    differentiation, at the Params' current values.

    ``coordinates="q"`` differentiates with respect to the configuration (Δ for a soft arm);
    ``"motors"`` with respect to the motor angles, through the transmission's exact inverse.
    ``at`` is a site name, an arc parameter s, or ``(part, s)``; ``offset`` [m] is a point in
    that frame.
    """

    def __init__(self, robot: Any, coordinates: str = "q") -> None:
        if coordinates not in ("q", "motors"):
            raise ValueError("coordinates must be 'q' or 'motors'")
        self.model = robot.model if hasattr(robot, "components") else robot
        actuation = getattr(robot, "actuation", None)
        self.actuation = Direct() if actuation is None else actuation
        self.coordinates = coordinates
        self._cache: dict[Any, ca.Function] = {}

    def functions(self, at: Any, offset: ArrayLike | None = None) -> ca.Function:
        """CasADi function x → (p, R, J, J_ω, H) for a point; H stacks ∂²p_k/∂x² row-block-wise."""
        off = np.zeros(3) if offset is None else np.asarray(offset, dtype=float).ravel()
        values = np.concatenate([self.model.params.vector(), self.actuation.params.vector()])
        key = (at if not isinstance(at, np.ndarray) else float(at), tuple(off), values.tobytes())
        if key not in self._cache:
            self._cache[key] = self._build(at, off)
        return self._cache[key]

    def _build(self, at: Any, offset: np.ndarray) -> ca.Function:
        space = self.model.space
        pm = constants(self.model.params)
        if self.coordinates == "motors":
            n = self.actuation.motor_sizes(space)[0]
            x = ca.SX.sym("theta", n)
            q = self.actuation.config_from_motors(x, constants(self.actuation.params))
            G = ca.SX.eye(n)
        else:
            x = ca.SX.sym("q", space.nq)
            q = x
            G = space.velocity_map(x)
        R, p = self.model.frame(q, at, pm)
        p = p + ca.mtimes(R, ca.DM(offset))
        J = ca.mtimes(ca.jacobian(p, x), G)
        dR = [ca.jacobian(ca.vec(R), x[i]) for i in range(x.numel())]
        Jw = ca.horzcat(*[_vee(ca.mtimes(ca.reshape(d, 3, 3), R.T)) for d in dR])
        Jw = ca.mtimes(Jw, G)
        H = ca.vertcat(*[ca.hessian(p[k], x)[0] for k in range(3)])
        return ca.Function("kinematics", [x], [p, R, J, Jw, H], ["x"], ["p", "R", "J", "Jw", "H"])

    def _eval(self, x: ArrayLike, at: Any, offset: ArrayLike | None, k: int) -> np.ndarray:
        return np.array(self.functions(at, offset)(np.asarray(x, dtype=float))[k])

    def position(self, x: ArrayLike, at: Any, offset: ArrayLike | None = None) -> np.ndarray:
        """Position [m], (3,)."""
        return self._eval(x, at, offset, 0).ravel()

    def rotation(self, x: ArrayLike, at: Any, offset: ArrayLike | None = None) -> np.ndarray:
        """Rotation of the frame, (3, 3)."""
        return self._eval(x, at, offset, 1)

    def jacobian(self, x: ArrayLike, at: Any, offset: ArrayLike | None = None) -> np.ndarray:
        """Position Jacobian ∂p/∂x, (3, n) (with respect to velocities on curved spaces)."""
        return self._eval(x, at, offset, 2)

    def angular_jacobian(
        self, x: ArrayLike, at: Any, offset: ArrayLike | None = None
    ) -> np.ndarray:
        """Angular-velocity Jacobian in the base frame, (3, n): ω = J_ω ẋ."""
        return self._eval(x, at, offset, 3)

    def hessian(self, x: ArrayLike, at: Any, offset: ArrayLike | None = None) -> np.ndarray:
        """Position Hessian, (3, n, n): H[k, i, j] = ∂²p_k/∂x_i∂x_j."""
        H = self._eval(x, at, offset, 4)
        n = H.shape[1]
        return H.reshape(3, n, n)
