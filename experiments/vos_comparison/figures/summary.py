"""Summary-statistic boxplots across videos, grouped by reduction (mean/min/object)."""

from __future__ import annotations

import numpy as np

from . import _common as C

# (metric column, axis label, as_percent)
METRICS = [
    ("mean_iou", "mean IoU", False),
    ("collapsed_frac", "fraction of video collapsed\n(rolling IoU<τ for ≥ min_run)", True),
    ("frac_below_thresh", "fraction of scored frames\nwith IoU < τ (any frame)", True),
    ("frac_zero", "fraction of scored frames\nwith IoU = 0 (any frame)", True),
    ("zero_collapsed_frac", "fraction of video at\nsustained zero IoU", True),
    ("collapsed_longest_run", "longest collapse run (frames)", False),
]
_ORDER = ["mean", "min", "object"]


def _values_by_reduction(rows, metric):
    out = {r: [] for r in _ORDER}
    for row in rows:
        red = row.get("reduction")
        if red in out:
            v = C.f(row.get(metric))
            if np.isfinite(v):
                out[red].append(v)
    return out


def _boxplot(ax, rows, metric, label, as_percent):
    vals = _values_by_reduction(rows, metric)
    data, labels, colors = [], [], []
    for red in _ORDER:
        xs = vals[red]
        if as_percent:
            xs = [100 * x for x in xs]
        data.append(xs if xs else [np.nan])
        labels.append(f"{red}\n(n={len(vals[red])})")
        colors.append(C.REDUCTION_COLOR[red])
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel(label + (" (%)" if as_percent else ""))
    if as_percent:
        ax.set_ylim(-2, 102)


def make_summary(out_dir, fig_dir):
    rows = C.collapse_stats(out_dir)
    if not rows:
        return
    # individual boxplots
    for metric, label, pct in METRICS:
        fig, ax = C.plt.subplots(figsize=(5.0, 4.0))
        _boxplot(ax, rows, metric, label, pct)
        ax.set_title(f"{label.splitlines()[0]} by reduction")
        fig.tight_layout()
        C.save(fig, fig_dir, f"box_{metric}")

    # one combined overview panel
    n = len(METRICS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = C.plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    for k, (metric, label, pct) in enumerate(METRICS):
        _boxplot(axes[k // ncols][k % ncols], rows, metric, label, pct)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Single-shot VOS failure statistics by reduction "
                 "(mean / min over objects / per-object)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    C.save(fig, fig_dir, "summary_overview")
