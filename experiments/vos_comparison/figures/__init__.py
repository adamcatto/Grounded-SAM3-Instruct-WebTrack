"""
Conference-ready figures and LaTeX tables for the VOS comparison.

Consumes the artifacts written by ``analyze.py`` (``timelines.npz``, ``results.json``,
``video_summary.csv``, ``object_summary.csv``, ``collapse_stats.csv``,
``aggregate_stats.csv``) and writes an organized tree:

    <out_dir>/figures/
        curves/      mean, min, objects_overlay, objects_facets  (per-video small multiples)
        aggregate/   band_{mean,min,objects}, survival_{mean,min,objects}
        summary/     box_<metric>, summary_overview              (boxplots by reduction)
        bars/        per_video_summary
    <out_dir>/tables/
        table_aggregate.tex, table_per_video.tex, table_collapse_by_reduction.tex

This package is backend-free (only matplotlib + numpy + the CSV/npz/json files), so it
can render anywhere without the SAM/mask stack.
"""

from __future__ import annotations

from pathlib import Path

from . import _common as C
from . import aggregate, bars, curves, summary, tables


def make_figures(out_dir: Path, *, threshold: float | None = None) -> None:
    out_dir = Path(out_dir).resolve()
    C.apply_paper_style()

    results = C.load_results(out_dir)
    tl = C.load_timelines(out_dir)
    meta = C.video_meta(out_dir)
    tau = threshold if threshold is not None else results.get("threshold", 0.5)

    fig_root = out_dir / "figures"
    tab_dir = out_dir / "tables"
    print(f"[figures] writing to {fig_root} and {tab_dir}")

    if not tl:
        print("[figures] WARNING: no timelines.npz found — run `analyze` first. "
              "Skipping curve/aggregate figures.")
    else:
        curves.curves_mean(out_dir, tl, meta, tau, fig_root / "curves")
        curves.curves_min(out_dir, tl, meta, tau, fig_root / "curves")
        curves.curves_objects_overlay(out_dir, tl, meta, tau, fig_root / "curves")
        curves.curves_objects_facets(out_dir, tl, meta, tau, fig_root / "curves")
        aggregate.make_aggregate(tl, tau, fig_root / "aggregate")

    summary.make_summary(out_dir, fig_root / "summary")
    bars.make_bars(out_dir, meta, tau, fig_root / "bars")
    tables.make_tables(out_dir, results, tab_dir)


__all__ = ["make_figures"]
