"""Select and plot representative per-video IoU curves by failure mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import _common as C

CATEGORIES = ("gradual", "catastrophic", "resilient")
CAT_TITLES = {
    "gradual": "Gradual drift",
    "catastrophic": "Catastrophic collapse",
    "resilient": "Resilient tracking",
}


def _video_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for r in C.collapse_stats(out_dir):
        if r.get("reduction") != "mean":
            continue
        rows.append({
            "video_id": r["video_id"],
            "name": r.get("video_name") or r["video_id"],
            "mean_iou": C.f(r.get("mean_iou")),
            "collapsed_frac": C.f(r.get("collapsed_frac")),
            "collapse_start": C.i(r.get("collapse_start_frame")),
        })
    return rows


def select_representative_videos(out_dir: Path) -> dict[str, str]:
    """Return {category: video_id} picking one exemplar per failure pattern."""
    rows = _video_rows(out_dir)
    picks: dict[str, dict[str, Any]] = {}

    resilient = [r for r in rows if r["mean_iou"] > 0.8 and r["collapse_start"] is None]
    catastrophic = [r for r in rows if r["mean_iou"] < 0.3]
    gradual = [
        r for r in rows
        if 0.3 <= r["mean_iou"] <= 0.6
        and r["collapsed_frac"] is not None
        and np.isfinite(r["collapsed_frac"])
        and 0.3 <= r["collapsed_frac"] <= 0.7
    ]

    if gradual:
        picks["gradual"] = sorted(gradual, key=lambda r: abs(r["mean_iou"] - 0.45))[0]
    if catastrophic:
        picks["catastrophic"] = sorted(catastrophic, key=lambda r: r["mean_iou"])[0]
    if resilient:
        picks["resilient"] = sorted(resilient, key=lambda r: -r["mean_iou"])[0]

    return {k: v["video_id"] for k, v in picks.items()}


def _collapse_start(out_dir: Path, vid: str) -> float | None:
    for r in C.collapse_stats(out_dir):
        if r.get("reduction") == "mean" and r.get("video_id") == vid:
            return C.i(r.get("collapse_start_frame"))
    return None


def make_representative_panel(out_dir: Path, out_base: Path, *, tau: float | None = None) -> dict[str, str]:
    """Plot 1×3 IoU curves; save to out_base.{png,pdf,svg}. Returns video_id picks."""
    out_dir = Path(out_dir)
    tl = C.load_timelines(out_dir)
    if not tl:
        raise FileNotFoundError(f"timelines.npz not found under {out_dir}")

    results = C.load_results(out_dir)
    tau = tau if tau is not None else float(results.get("threshold", 0.3))
    picks = select_representative_videos(out_dir)

    fig, axes = C.plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        ax.set_title(CAT_TITLES[cat], fontsize=9, loc="left")
        ax.set_xlabel("frame")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(1.0, color=C.C_ANCHOR, lw=1.0, ls="--", alpha=0.75)
        ax.axhline(tau, color="0.4", lw=0.8, ls=":")
        vid = picks.get(cat)
        if not vid:
            ax.text(0.5, 0.5, "no exemplar", ha="center", va="center")
            continue
        frames, vals = C.reduction_series(tl, vid, "mean")
        ax.plot(frames, vals, color=C.C_SINGLE, lw=1.2)
        cstart = _collapse_start(out_dir, vid)
        if cstart is not None:
            ax.axvline(cstart, color=C.C_COLLAPSE, lw=1.0, ls="-", alpha=0.9)
        short = str(vid)[:18]
        ax.text(0.02, 0.05, short, transform=ax.transAxes, fontsize=7, va="bottom")
    axes[0].set_ylabel("IoU vs. anchor GT")
    fig.tight_layout()
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"))
    C.plt.close(fig)
    print(f"[figures]   representative panel → {out_base.name}.{{png,pdf,svg}}")
    return picks
