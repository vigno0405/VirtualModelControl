"""A natural cubic spline through points that are Params, written in CasADi operations."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


def curvature_map(m: int) -> np.ndarray:
    """The matrix E with E @ y the second derivatives of the natural spline through m values y,
    one unit of the parameter apart (zero at both ends)."""
    E = np.zeros((m, m))
    if m > 2:
        n = m - 2
        A = np.diag(np.full(n, 2 / 3)) + np.diag(np.full(n - 1, 1 / 6), 1)
        A = A + np.diag(np.full(n - 1, 1 / 6), -1)
        D = np.zeros((n, m))
        for i in range(n):
            D[i, i : i + 3] = [1.0, -2.0, 1.0]
        E[1:-1] = np.linalg.solve(A, D)
    return E


def spline_point(points: Any, u: Any) -> Any:
    """The point (3, 1) at ``u``, in [0, m - 1], of the natural cubic spline through the rows of
    ``points`` (m, 3), which are one unit of ``u`` apart; past the ends the end cubics go on."""
    m = points.shape[0]
    second = ca.mtimes(ca.DM(curvature_map(m)), points)

    def piece(k: int) -> Any:
        b = u - k
        a = 1 - b
        row = a * points[k, :] + b * points[k + 1, :]
        return (row + ((a**3 - a) * second[k, :] + (b**3 - b) * second[k + 1, :]) / 6).T

    out = piece(m - 2)
    for k in reversed(range(m - 2)):
        out = ca.if_else(u <= k + 1, piece(k), out)
    return out
