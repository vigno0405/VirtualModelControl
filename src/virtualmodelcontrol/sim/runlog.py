"""Run logs: the recorded steps of a run and what the run was, saved and loaded in one file."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
"""Version of the saved format; ``RunLog.load`` reads this one and older ones."""


@dataclass
class RunLog:
    """Recorded signals of a run, one row per step; ``arrays`` stacks them.

    ``info`` holds the statistics of a real-time run; ``meta`` says what the run was (the library's
    version, the start time, the Params at the start, a hardware profile, a configuration).
    """

    rows: dict[str, list[np.ndarray]] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def append(self, **values: Any) -> None:
        """Add one row of named values."""
        for name, value in values.items():
            # A copy: plants and controllers may update their arrays in place.
            self.rows.setdefault(name, []).append(np.array(value, dtype=float, ndmin=1))

    def step(self, **values: Any) -> None:
        """Add one step's values. A name left out at this step gets NaN, and a name new at this
        step gets NaN for the steps before, so every signal keeps one row per step."""
        n = max((len(rows) for rows in self.rows.values()), default=0)
        for name, value in values.items():
            row = np.array(value, dtype=float, ndmin=1)
            rows = self.rows.setdefault(name, [])
            rows.extend(np.full_like(row, np.nan) for _ in range(n - len(rows)))
            rows.append(row)
        for rows in self.rows.values():
            if len(rows) == n:
                rows.append(np.full_like(rows[-1], np.nan))

    def arrays(self) -> dict[str, np.ndarray]:
        """Each signal as an (n_steps, ...) array."""
        return {name: np.array(rows) for name, rows in self.rows.items()}

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Write the log as a compressed ``.npz`` and return its path. The file is written whole or
        not at all (through a temporary file), and an existing one is kept unless ``overwrite``."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; save with overwrite=True to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name[: -len(".npz")] + ".part.npz")
        contents: dict[str, Any] = {
            **self.arrays(),
            "_schema": np.array(SCHEMA),
            "_meta": np.array(json.dumps(self.meta, default=_plain)),
            "_info": np.array(json.dumps(self.info, default=_plain)),
        }
        try:
            np.savez_compressed(part, **contents)
            os.replace(part, path)
        except BaseException:  # an interrupted or failed write leaves nothing behind
            part.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RunLog:
        """A log written by ``save``."""
        with np.load(path, allow_pickle=False) as data:
            if "_schema" not in data.files:
                raise ValueError(f"{path} is not a run log written by RunLog.save")
            schema = int(data["_schema"])
            if schema > SCHEMA:
                raise ValueError(
                    f"{path} has schema {schema}, newer than this library's {SCHEMA}: update it"
                )
            rows = {name: list(data[name]) for name in data.files if not name.startswith("_")}
            meta, info = json.loads(str(data["_meta"])), json.loads(str(data["_info"]))
        return cls(rows, info, meta)

    def to_csv(self, path: str | os.PathLike[str]) -> Path:
        """Write the steps as CSV, one line per step and one column per entry: ``name`` for a
        single value, ``name_i`` (``name_i_j``) for the entries of a vector (matrix)."""
        columns, blocks = [], []
        for name, values in self.arrays().items():
            shape = values.shape[1:]
            if shape in ((), (1,)):
                columns.append(name)
            else:
                columns += [f"{name}_{'_'.join(map(str, i))}" for i in np.ndindex(shape)]
            blocks.append(values.reshape(len(values), -1))
        path = Path(path)
        np.savetxt(
            path,
            np.hstack(blocks),
            fmt="%.17g",
            delimiter=",",
            header=",".join(columns),
            comments="",
        )
        return path


def compare(
    a: RunLog, b: RunLog, names: Iterable[str] | None = None
) -> dict[str, dict[str, float]]:
    """How far run ``b`` is from run ``a``, signal by signal: the largest and the root-mean-square
    difference, with ``b`` interpolated to ``a``'s times where they overlap (NaN steps left out);
    by default every signal both runs hold."""
    rows_a, rows_b = a.arrays(), b.arrays()
    ta, tb = rows_a["t"].ravel(), rows_b["t"].ravel()
    if names is None:
        names = [name for name in rows_a if name in rows_b and name not in ("t", "dt")]
    inside = (ta >= tb[0]) & (ta <= tb[-1])
    out = {}
    for name in names:
        x = rows_a[name].reshape(len(ta), -1)
        y = rows_b[name].reshape(len(tb), -1)
        at_a = np.column_stack([np.interp(ta, tb, column) for column in y.T])
        gap = (x - at_a)[inside]
        rms = float(np.sqrt(np.nanmean(gap**2)))
        out[name] = {"max": float(np.nanmax(np.abs(gap))), "rms": rms}
    return out


def library_version() -> str:
    """The library's version, as ``vmc.__version__`` gives it."""
    try:
        from .._version import __version__
    except ImportError:  # a source tree that was never installed
        return "0.0.0+unknown"
    return str(__version__)


def _plain(value: Any) -> Any:
    """numpy values as JSON takes them."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"cannot store {type(value).__name__} in a run log's metadata")
