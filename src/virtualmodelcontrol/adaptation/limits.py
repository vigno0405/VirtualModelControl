"""What the laws share: which live Params they move, and the values those may take."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..control.projection import project_psd

TINY = 1e-12
"""Below this a step size or a force change counts as zero, so that no division blows up."""

NOISE = 1e-10
"""Singular values of a map below this share of the largest are rounding noise, not a direction."""


def live_matching(compiled: Any, params: str | Sequence[str]) -> list[str]:
    """The live Params of ``compiled`` matching the glob patterns, in the order of its vector."""
    patterns = [params] if isinstance(params, str) else list(params)
    names = [n for n in compiled.live if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not names:
        raise ValueError(f"no live Param matches {patterns}; compile with runtime=[...]")
    return names


def admissible(param: Any, value: np.ndarray) -> np.ndarray:
    """``value`` (flat, column by column) within the bounds of ``param``; a square matrix is also
    made symmetric and positive semidefinite."""
    lo, hi = (
        np.ravel(np.broadcast_to(np.asarray(b, float), param.shape), order="F")
        for b in param.bounds
    )
    value = np.clip(value, lo, hi)
    if len(param.shape) == 2 and param.shape[0] == param.shape[1]:
        value = np.ravel(project_psd(np.reshape(value, param.shape, order="F")), order="F")
    return value
