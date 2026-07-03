"""Centralized figure saving.

Every downstream plot is written in **both** PNG and SVG. Instead of dropping
the two formats side by side, the figure tree is mirrored into two top-level
folders under the figure root:

    <root>/png/<same-subtree>/<name>.png
    <root>/svg/<same-subtree>/<name>.svg

where ``<root>`` is the nearest ancestor directory whose name is a known figure
root (see :data:`FIGURE_ROOTS`). For example::

    .../plots/umap/umap_by_cluster.png
        -> .../plots/png/umap/umap_by_cluster.png
        -> .../plots/svg/umap/umap_by_cluster.svg

Only figures flow through here; sibling non-figure outputs (CSV/JSON) are left
untouched wherever their writers put them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib import rc_context
from matplotlib.figure import Figure

# Directory names that anchor a png/svg split. A figure written anywhere below
# one of these gets mirrored into ``<root>/png/...`` and ``<root>/svg/...``.
# The rightmost (deepest) match wins, so a stray "plots" higher in an absolute
# path never shadows the real figure root.
FIGURE_ROOTS = ("plots", "batch_diagnostics", "analysis_of_tracking_data")

DEFAULT_FORMATS = ("png", "svg")

# The SVGs are the paper-submission asset. SVG geometry is resolution
# independent, but rasterized sub-elements (imshow heatmaps, ethograms,
# rasterized scatter) bake at ``savefig`` dpi — floor it so those stay
# print-grade (>=300 dpi) regardless of the caller's on-screen dpi.
SVG_MIN_DPI = 300

# Per-format rcParams applied only while writing that format. ``svg.fonttype:
# none`` keeps text as selectable/editable ``<text>`` (real fonts, smaller
# files) instead of outlined paths — matching the repo's camera-ready style.
_FORMAT_RC: dict[str, dict[str, object]] = {
    "svg": {"svg.fonttype": "none"},
}


def _split_at_root(outfile: Path) -> tuple[Path, Path] | None:
    """Split ``outfile`` into ``(root, rel)`` at the deepest figure-root marker.

    ``rel`` includes the filename. Returns ``None`` when no marker is present.
    """
    parts = outfile.parts
    # Search parent directories only (exclude the filename itself), deepest first.
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in FIGURE_ROOTS:
            return Path(*parts[: i + 1]), Path(*parts[i + 1 :])
    return None


def save_figure(
    fig: Figure,
    outfile: Path | str,
    *,
    dpi: int = 140,
    formats: Iterable[str] = DEFAULT_FORMATS,
    **savefig_kwargs,
) -> None:
    """Save ``fig`` as every format in ``formats`` under the png/svg split.

    ``outfile`` is the historical single-format path (typically ``*.png``); its
    suffix is swapped per format. When ``outfile`` lives under a figure root the
    output is redirected into ``<root>/<fmt>/...``; otherwise the formats are
    written side by side next to ``outfile`` as a fallback.

    Extra keyword arguments (e.g. ``bbox_inches="tight"``) are forwarded to
    ``Figure.savefig`` unchanged.
    """
    outfile = Path(outfile)
    split = _split_at_root(outfile)
    for fmt in formats:
        if split is not None:
            root, rel = split
            dest = root / fmt / rel.with_suffix(f".{fmt}")
        else:
            dest = outfile.with_suffix(f".{fmt}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fmt_dpi = max(dpi, SVG_MIN_DPI) if fmt == "svg" else dpi
        with rc_context(_FORMAT_RC.get(fmt, {})):
            fig.savefig(dest, dpi=fmt_dpi, **savefig_kwargs)
