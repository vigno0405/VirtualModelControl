"""Projection of a stiffness onto the matrices a passive spring can have."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def project_psd(K: ArrayLike) -> np.ndarray:
    """The symmetric positive semidefinite matrix nearest to ``K``: symmetrize it, then raise its
    negative eigenvalues to 0. A spring with such a stiffness never stores negative energy."""
    K = np.asarray(K, dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (K + K.T))
    return (vectors * np.maximum(values, 0.0)) @ vectors.T
