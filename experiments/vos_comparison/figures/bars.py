"""Per-video summary bars: mean IoU and time-to-collapse (mean reduction)."""

from __future__ import annotations

import numpy as np

from . import _common as C


def make_bars(out_dir, meta, tau, fig_dir):
    vids = list(meta.keys())
    if not vids:
        return
    order = sorted(vids, key=lambda v: (meta[v]["mean_iou"]
                                        if np.isfinite(meta[v]["mean_iou"]) else 1.0))
    names = [str(meta[v]["name"])[:22] for v in order]
    means = [meta[v]["mean_iou"] for v in order]
    coll = [(meta[v]["collapse_fraction"] * 100
             if np.isfinite(meta[v]["collapse_fraction"]) else 100.0) for v in order]
    y = np.arange(len(order))

    fig, (ax1, ax2) = C.plt.subplots(
        1, 2, figsize=(10.0, max(3.0, 0.42 * len(order) + 1.5)))
    ax1.barh(y, means, color=C.C_SINGLE, alpha=0.85)
    ax1.axvline(tau, color="0.4", ls=":", lw=0.9, label=f"τ={tau:g}")
    ax1.set_yticks(y, names)
    ax1.set_xlabel("mean IoU vs. anchor GT")
    ax1.set_xlim(0, 1)
    ax1.set_title("Per-video mean IoU")
    ax1.legend(loc="lower right")

    ax2.barh(y, coll, color=C.C_COLLAPSE, alpha=0.85)
    ax2.set_yticks(y, names)
    ax2.set_xlabel("collapse point (% of video; 100 = never)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Time to collapse")
    fig.suptitle("Single-shot VOS: where and how badly tracking fails")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    C.save(fig, fig_dir, "per_video_summary")
