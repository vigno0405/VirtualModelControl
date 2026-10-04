"""Lab figure style (Computer Modern, NPG palette) and saving every figure as PDF and SVG."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
from cycler import cycler

PALETTE = [
    "#4DBBD5",  # blue
    "#E64B35",  # red
    "#00A087",  # teal
    "#3C5488",  # navy
    "#F39B7F",  # coral
    "#8491B4",  # lavender
    "#91D1C2",  # light teal
    "#DC0000",  # bright red
    "#7E6148",  # brown
    "#949494",  # grey
]
"""Colours in plotting order (NPG palette)."""


def rc(usetex: bool | None = None, font_size: float = 18.0) -> dict[str, Any]:
    """Matplotlib settings of the lab style; LaTeX text when ``usetex`` (default: if installed)."""
    if usetex is None:
        usetex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    small = round(font_size * 0.72)
    return {
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.family": "serif",
        # Without LaTeX, matplotlib's own Computer Modern (cmr10); DejaVu fills missing glyphs.
        "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,  # tick labels as math: cmr10 has no minus sign
        "axes.unicode_minus": False,  # LaTeX renders "-" in math mode as a minus sign
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": small,
        "legend.title_fontsize": small,
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "axes.linewidth": 1.5,
        "axes.grid": False,
        "grid.linewidth": 1.0,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=PALETTE),
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }


def use_style(usetex: bool | None = None, font_size: float = 18.0) -> None:
    """Apply the lab style to every following figure."""
    mpl.rcParams.update(cast(Any, rc(usetex, font_size)))


@contextmanager
def style(usetex: bool | None = None, font_size: float = 18.0) -> Iterator[None]:
    """The lab style inside a ``with`` block only."""
    with mpl.rc_context(cast(Any, rc(usetex, font_size))):
        yield


def save(fig: Any, path: str | Path) -> tuple[Path, Path]:
    """Save ``fig`` as PDF and SVG next to each other; returns both paths."""
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, svg = stem.with_suffix(".pdf"), stem.with_suffix(".svg")
    fig.savefig(pdf)
    fig.savefig(svg)
    return pdf, svg
