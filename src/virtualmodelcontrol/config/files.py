"""Configuration files: YAML in and out, with numbers such as 1e-3 read as numbers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class _Loader(yaml.SafeLoader):
    """The safe loader, reading floats as YAML 1.2 does: 1e-3 is a number, not a string."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The configuration in a YAML file."""
    data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_Loader)  # a safe loader
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a configuration is a mapping of sections, got {data!r}")
    return data


def plain(value: Any) -> Any:
    """``value`` with tuples as lists and numpy arrays and numbers as Python's, for YAML."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def dumps(data: dict[str, Any]) -> str:
    """A configuration as YAML text, short lists on one line."""
    return yaml.dump(
        plain(data),
        Dumper=yaml.SafeDumper,
        sort_keys=False,
        default_flow_style=None,
        allow_unicode=True,
        width=100,
    )


def write(data: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a configuration as YAML; returns the path."""
    path = Path(path)
    path.write_text(dumps(data), encoding="utf-8")
    return path
