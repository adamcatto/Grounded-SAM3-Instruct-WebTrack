"""
Conference-ready figures and LaTeX tables for the VOS comparison.

Reads ``results.json`` + ``timelines.npz`` produced by ``evaluation.py`` and
writes vector (PDF) + raster (PNG) figures plus booktabs LaTeX tables into
``<out_dir>/figures/`` and ``<out_dir>/tables/``.

Figures:
  fig_iou_curves         small-multiples of per-video IoU-vs-frame, collapse marked
  fig_aggregate_band     mean IoU ± IQR vs. normalized video position (all videos)
  fig_survival           fraction of videos still tracking (IoU >= tau) vs. position
  fig_summary_bars       per-video mean IoU and time-to-collapse

Tables:
  table_per_video.tex    one row per video
  table_aggregate.tex    headline numbers for the paper
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Two-condition palette: single-shot (what we test) vs the anchor baseline line.
C_SINGLE = "#C0392B"   # red — single-shot degradation
C_ANCHOR = "#27AE60"   # green — anchor ground-truth reference (IoU == 1)
C_BAND = "#5B8DD9"     # blue — aggregate band
C_COLLAPSE = "#E67E22"  # orange — collapse marker


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
        }
    )


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"[figures]   {name}.pdf / .png")


def _load(out_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    results = json.loads((out_dir / "results.json").read_text())
    tl = np.load(out_dir / "timelines.npz")
    timelines = {k: tl[k] for k in tl.files}
    return results, timelines


def _video_arrays(timelines: dict[str, np.ndarray], vid: str) -> tuple[np.ndarray, np.ndarray]:
    return timelines[f"{vid}__frames"], timelines[f"{vid}__mean_iou"]


def _normalized_resample(
    frames: np.ndarray, vals: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate a per-frame series onto a common [0,1] normalized-position grid."""
    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return np.full(grid.size, np.nan)
    f = frames[finite].astype(np.float64)
    v = vals[finite]
    pos = (f - f.min()) / max(1e-9, (f.max() - f.min()))
    return np.interp(grid, pos, v, left=v[0], right=v[-1])


# ── Figure: per-video IoU curves (small multiples) ─────────────────────────
def fig_iou_curves(results, timelines, threshold, out_dir: Path) -> None:
    videos = results["videos"]
    n = len(videos)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False, sharey=True
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for i, vs in enumerate(videos):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        frames, vals = _video_arrays(timelines, vs["video_id"])
        ax.plot(frames, vals, color=C_SINGLE, label="single-shot")
        ax.axhline(1.0, color=C_ANCHOR, lw=1.0, ls="--", alpha=0.7, label="anchor (GT)")
        ax.axhline(threshold, color="0.4", lw=0.8, ls=":", label=f"τ={threshold:g}")
        if vs.get("collapse_frame") is not None:
            ax.axvline(vs["collapse_frame"], color=C_COLLAPSE, lw=1.2, ls="-",
                       label="collapse")
        ax.set_title(f"{vs['video_name'][:28]}\n(mean IoU={vs['mean_iou']:.2f})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("frame")
        if i % ncols == 0:
            ax.set_ylabel("IoU vs. anchor GT")

    # Single shared legend.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Single-shot VOS quality over time (per video)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    _save(fig, out_dir, "fig_iou_curves")


# ── Figure: aggregate IoU band vs normalized position ──────────────────────
def fig_aggregate_band(results, timelines, threshold, out_dir: Path) -> None:
    videos = results["videos"]
    if not videos:
        return
    grid = np.linspace(0.0, 1.0, 200)
    stacked = []
    for vs in videos:
        frames, vals = _video_arrays(timelines, vs["video_id"])
        stacked.append(_normalized_resample(frames, vals, grid))
    M = np.vstack(stacked)

    mean = np.nanmean(M, axis=0)
    q1 = np.nanpercentile(M, 25, axis=0)
    q3 = np.nanpercentile(M, 75, axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(1.0, color=C_ANCHOR, lw=1.2, ls="--", alpha=0.8,
               label="anchor labeling (ground truth)")
    ax.fill_between(grid, q1, q3, color=C_BAND, alpha=0.25, label="IQR across videos")
    ax.plot(grid, mean, color=C_BAND, label="single-shot mean IoU")
    ax.axhline(threshold, color="0.4", lw=0.9, ls=":", label=f"collapse threshold τ={threshold:g}")
    ax.set_xlabel("normalized video position (0 = first frame, 1 = last)")
    ax.set_ylabel("IoU vs. anchor ground truth")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Single-shot tracking degrades over the video; anchors hold IoU at 1")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, "fig_aggregate_band")


# ── Figure: survival curve ─────────────────────────────────────────────────
def fig_survival(results, timelines, threshold, out_dir: Path) -> None:
    videos = results["videos"]
    if not videos:
        return
    grid = np.linspace(0.0, 1.0, 200)
    alive = np.zeros(grid.size, dtype=np.float64)
    counted = np.zeros(grid.size, dtype=np.float64)
    for vs in videos:
        frames, vals = _video_arrays(timelines, vs["video_id"])
        r = _normalized_resample(frames, vals, grid)
        finite = np.isfinite(r)
        counted += finite
        alive += np.where(finite & (r >= threshold), 1.0, 0.0)
    frac = np.divide(alive, counted, out=np.full_like(alive, np.nan), where=counted > 0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(grid, 100 * frac, color=C_SINGLE, label="single-shot")
    ax.axhline(100, color=C_ANCHOR, lw=1.2, ls="--", alpha=0.8, label="anchor labeling")
    ax.set_xlabel("normalized video position")
    ax.set_ylabel(f"% of videos with IoU ≥ {threshold:g}")
    ax.set_ylim(-2, 103)
    ax.set_title("Tracking survival: fraction of videos still matching ground truth")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, "fig_survival")


# ── Figure: summary bars ───────────────────────────────────────────────────
def fig_summary_bars(results, timelines, threshold, out_dir: Path) -> None:
    videos = results["videos"]
    if not videos:
        return
    order = sorted(videos, key=lambda v: v["mean_iou"])
    names = [v["video_name"][:22] for v in order]
    means = [v["mean_iou"] for v in order]
    coll = [
        (v["collapse_fraction"] * 100 if v["collapse_fraction"] is not None else 100.0)
        for v in order
    ]
    y = np.arange(len(order))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, max(3.0, 0.45 * len(order) + 1.5)))
    ax1.barh(y, means, color=C_SINGLE, alpha=0.85)
    ax1.axvline(threshold, color="0.4", ls=":", lw=0.9, label=f"τ={threshold:g}")
    ax1.set_yticks(y, names)
    ax1.set_xlabel("mean IoU vs. anchor GT")
    ax1.set_xlim(0, 1)
    ax1.set_title("Per-video mean IoU")
    ax1.legend(loc="lower right")

    ax2.barh(y, coll, color=C_COLLAPSE, alpha=0.85)
    ax2.set_yticks(y, names)
    ax2.set_xlabel("collapse point (% of video; 100 = never)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Time to collapse")
    fig.suptitle("Single-shot VOS: where and how badly tracking fails")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, "fig_summary_bars")


# ── LaTeX tables ───────────────────────────────────────────────────────────
def _fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:.{nd}f}"


def table_per_video(results, out_dir: Path) -> None:
    rows = []
    for v in sorted(results["videos"], key=lambda d: d["video_name"].lower()):
        cf = v["collapse_frame"]
        rows.append(
            " & ".join(
                [
                    _latex_escape(v["video_name"]),
                    str(v["num_scored_frames"]),
                    _fmt(v["mean_iou"]),
                    _fmt(v["median_iou"]),
                    _fmt(v["frac_frames_above_threshold"], pct=True),
                    (str(cf) if cf is not None else "--"),
                    (_fmt(v["collapse_fraction"], pct=True) if v["collapse_fraction"] is not None else "--"),
                ]
            )
            + r" \\"
        )
    agg = results["aggregate"]
    body = "\n".join(rows)
    tex = rf"""% Auto-generated by experiments/vos_comparison/figures.py
\begin{{table}}[t]
\centering
\caption{{Single-shot VOS vs. anchor-frame labeling (ground truth). IoU is the
Jaccard index against the anchor-based track. Collapse = first frame after which
IoU stays below $\tau={_fmt(results['threshold'])}$ for the remainder of the video.}}
\label{{tab:vos-per-video}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Video & Frames & Mean IoU & Median IoU & \% IoU$\geq\tau$ & Collapse frame & Collapse \% \\
\midrule
{body}
\midrule
\textbf{{All ({agg['n_videos']})}} & -- & {_fmt(agg['mean_iou'])} & -- & {_fmt(agg['mean_frac_above_threshold'], pct=True)} & -- & {(_fmt(agg['median_collapse_fraction'], pct=True) if agg.get('median_collapse_fraction') is not None else '--')} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_per_video.tex").write_text(tex)
    print("[figures]   table_per_video.tex")


def table_aggregate(results, out_dir: Path) -> None:
    agg = results["aggregate"]
    if agg.get("median_collapse_fraction") is not None:
        median_str = _fmt(agg["median_collapse_fraction"], pct=True) + r"\% of video"
    else:
        median_str = "--"
    tex = rf"""% Auto-generated by experiments/vos_comparison/figures.py
\begin{{table}}[t]
\centering
\caption{{Aggregate single-shot VOS performance against anchor-frame ground
truth over {agg['n_videos']} videos. Anchor labeling holds IoU at 1.0 by
construction; single-shot tracking degrades and collapses on
{_fmt(agg['frac_videos_collapsed'], pct=True)}\% of videos.}}
\label{{tab:vos-aggregate}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Single-shot \\
\midrule
Videos & {agg['n_videos']} \\
Mean IoU & {_fmt(agg['mean_iou'])} $\pm$ {_fmt(agg['std_iou'])} \\
Mean IoU AUC (vs. position) & {_fmt(agg['mean_iou_auc'])} \\
Mean \% frames IoU$\geq\tau$ & {_fmt(agg['mean_frac_above_threshold'], pct=True)} \\
Videos collapsed & {agg['n_collapsed']} / {agg['n_videos']} ({_fmt(agg['frac_videos_collapsed'], pct=True)}\%) \\
Median collapse point & {median_str} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_aggregate.tex").write_text(tex)
    print("[figures]   table_aggregate.tex")


def _latex_escape(s: str) -> str:
    repl = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
    return "".join(repl.get(c, c) for c in str(s))


def make_figures(out_dir: Path, *, threshold: float | None = None) -> None:
    out_dir = out_dir.resolve()
    results, timelines = _load(out_dir)
    tau = threshold if threshold is not None else results.get("threshold", 0.5)
    _apply_paper_style()

    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    print(f"[figures] writing to {fig_dir} and {tab_dir}")
    fig_iou_curves(results, timelines, tau, fig_dir)
    fig_aggregate_band(results, timelines, tau, fig_dir)
    fig_survival(results, timelines, tau, fig_dir)
    fig_summary_bars(results, timelines, tau, fig_dir)
    table_per_video(results, tab_dir)
    table_aggregate(results, tab_dir)
