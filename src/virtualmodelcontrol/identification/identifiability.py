"""What logged runs can tell apart: the velocity filter's window and the unknowns the data miss."""

from __future__ import annotations

import numpy as np


def blind_directions(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The scale of each unknown (the norm of its column of the Jacobian J, 1 for a zero column)
    and the directions of the scaled unknowns that J does not see, one per column. The columns are
    scaled to one so that unknowns of different units are compared alike."""
    norms = np.linalg.norm(J, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    scaled = J / norms
    w, V = np.linalg.eigh(scaled.T @ scaled)
    return norms, V[:, w <= 1e-12 * max(w[-1], np.finfo(float).tiny)]


def check_smoothing(smoothing: int) -> None:
    """The Savitzky-Golay window is an odd number of samples: an even one shifts the derivative
    by half a sample."""
    if smoothing < 5 or smoothing % 2 == 0:
        raise ValueError(f"smoothing is an odd number of samples, at least 5, not {smoothing}")


def undetermined(J: np.ndarray) -> np.ndarray:
    """The unknowns whose value the data cannot fix: those that can move along a direction that
    J does not see."""
    return np.linalg.norm(blind_directions(J)[1], axis=1) > 1e-6
