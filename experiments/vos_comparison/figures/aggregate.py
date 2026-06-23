"""Aggregate IoU band (mean ± IQR) and survival curves vs. normalized video position.

Each is rendered for three signals:
  * mean    — one series per video (mean over objects)
  * min     — one series per video (worst object)
  * objects — one series per (video, object), pooled
"""

from __future__ import annotations

import numpy as np

from . import _common as C

_GRID = np.linspace(0.0, 1.0, 200)


def _collect(tl, kind: str) -> list[np.ndarray]:
    """Resampled [0,1] series for the requested signal."""
    out = []
    for vid in C.video_ids(tl):
        if kind in ("mean", "min"):
            frames, vals = C.reduction_series(tl, vid, kind)
            out.append(C.normalized_resample(frames, vals, _GRID))
        else:  # objects
            frames, objs = C.object_series(tl, vid)
            for _oid, _nm, vals in objs:
                out.append(C.normalized_resample(frames, vals, _GRID))
    return out


def band(tl, tau, kind, fig_dir):
    series = _collect(tl, kind)
    if not series:
        return
    M = np.vstack(series)
    mean = np.nanmean(M, axis=0)
    q1 = np.nanpercentile(M, 25, axis=0)
    q3 = np.nanpercentile(M, 75, axis=0)
    unit = "videos" if kind in ("mean", "min") else "object tracks"

    fig, ax = C.plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(1.0, color=C.C_ANCHOR, lw=1.2, ls="--", alpha=0.8, label="anchor labeling (GT)")
    ax.fill_between(_GRID, q1, q3, color=C.C_BAND, alpha=0.25, label=f"IQR across {unit}")
    ax.plot(_GRID, mean, color=C.C_BAND, label=f"single-shot {kind} IoU")
    ax.axhline(tau, color="0.4", lw=0.9, ls=":", label=f"τ={tau:g}")
    ax.set_xlabel("normalized video position (0 = first frame, 1 = last)")
    ax.set_ylabel("IoU vs. anchor ground truth")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Single-shot IoU degrades over the video — {kind} signal ({M.shape[0]} {unit})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    C.save(fig, fig_dir, f"band_{kind}")


def survival(tl, tau, kind, fig_dir):
    series = _collect(tl, kind)
    if not series:
        return
    M = np.vstack(series)
    finite = np.isfinite(M)
    counted = finite.sum(axis=0)
    alive = (finite & (M >= tau)).sum(axis=0)
    frac = np.divide(alive, counted, out=np.full(counted.shape, np.nan, dtype=float),
                     where=counted > 0)
    unit = "videos" if kind in ("mean", "min") else "object tracks"

    fig, ax = C.plt.subplots(figsize=(7.0, 4.2))
    ax.plot(_GRID, 100 * frac, color=C.C_SINGLE, label=f"single-shot ({kind})")
    ax.axhline(100, color=C.C_ANCHOR, lw=1.2, ls="--", alpha=0.8, label="anchor labeling")
    ax.set_xlabel("normalized video position")
    ax.set_ylabel(f"% of {unit} with IoU ≥ {tau:g}")
    ax.set_ylim(-2, 103)
    ax.set_title(f"Tracking survival — {kind} signal")
    ax.legend(loc="lower left")
    fig.tight_layout()
    C.save(fig, fig_dir, f"survival_{kind}")


def make_aggregate(tl, tau, fig_dir):
    for kind in ("mean", "min", "objects"):
        band(tl, tau, kind, fig_dir)
        survival(tl, tau, kind, fig_dir)
