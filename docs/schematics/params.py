"""Parameter tables computed from a robot's Params, with the meaning of each written beside it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from IPython.display import Markdown


def _value(value: np.ndarray) -> str:
    v = np.asarray(value, dtype=float)
    if v.ndim == 0 or v.size == 1:
        return f"{float(v.ravel()[0]):.4g}"
    if v.ndim == 1 and v.size <= 3:
        return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
    if v.ndim == 1:
        return f"{v.size} values, {v.min():.4g} to {v.max():.4g}"
    return f"{v.shape[0]} × {v.shape[1]} matrix" if v.ndim == 2 else f"{v.size} values"


def table(robot: Any, meanings: Mapping[str, str], degrees: tuple[str, ...] = ()) -> Markdown:
    """A Markdown table (name, value, unit, meaning) of the named Params; values from the robot.

    Params named in ``degrees`` are shown in degrees as well as their stored radians.
    """
    params = robot.params
    lines = ["| Param | Value | Unit | Meaning |", "|---|---|---|---|"]
    for name, meaning in meanings.items():
        param = params[name]
        value = _value(param.value)
        if name in degrees:
            value += " (" + _value(np.degrees(param.value)) + " °)"
        unit = param.unit.replace("*", "·").replace("^2", "²").replace("^3", "³") or "1"
        lines.append(f"| `{name}` | {value} | {unit} | {meaning} |")
    return Markdown("\n".join(lines))
