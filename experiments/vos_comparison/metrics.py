"""IoU / Jaccard metrics and sustained-collapse detection for VOS comparison."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .common import npz_key_base_id

# np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed under the old name.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two binary masks.

    Empty-vs-empty returns 1.0 (both correctly predict "nothing here"), matching
    the DAVIS J convention; empty-vs-nonempty returns 0.0.
    """
    am = np.squeeze(np.asarray(a)).astype(bool)
    bm = np.squeeze(np.asarray(b)).astype(bool)
    if am.shape != bm.shape:
        # Defensive: resize should never differ, but guard against off-by-one.
        h = min(am.shape[0], bm.shape[0])
        w = min(am.shape[1], bm.shape[1])
        am, bm = am[:h, :w], bm[:h, :w]
    inter = np.logical_and(am, bm).sum(dtype=np.int64)
    union = np.logical_or(am, bm).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _merge_instances(masks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse instance keys ('1', '1_2', …) into per-base-object unions."""
    merged: dict[str, np.ndarray] = {}
    for key, m in masks.items():
        base = npz_key_base_id(str(key))
        mm = np.squeeze(np.asarray(m)).astype(bool)
        if base in merged:
            merged[base] = np.logical_or(merged[base], mm)
        else:
            merged[base] = mm
    return merged


def per_object_iou(
    gt_masks: Mapping[str, np.ndarray],
    pred_masks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """IoU per ground-truth object (instances merged by base id).

    Only objects present in the GT frame are scored. A GT object with no
    predicted counterpart scores 0.0 (full miss).
    """
    gt = _merge_instances(gt_masks)
    pred = _merge_instances(pred_masks)
    out: dict[str, float] = {}
    for base, gmask in gt.items():
        if not gmask.any():
            continue  # object absent in GT this frame — nothing to score
        pmask = pred.get(base)
        if pmask is None:
            out[base] = 0.0
        else:
            out[base] = mask_iou(gmask, pmask)
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean ignoring NaNs; returns array same length as x."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    window = max(1, int(window))
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = x[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(seg.mean())
    return out


def detect_collapse(
    scores: np.ndarray,
    *,
    threshold: float,
    window: int,
    persist_frac: float = 0.8,
    min_run: int | None = None,
) -> dict:
    """Find the frame where IoU degenerates for a large chunk of the rest of the video.

    The collapse frame is the earliest index ``t0`` such that, looking from ``t0``
    to the end, at least ``persist_frac`` of the (smoothed) frames sit below
    ``threshold`` and the collapsed tail is at least ``min_run`` frames long. This
    captures a sustained drop while ignoring brief dips that recover.

    Returns a dict with keys: ``collapsed`` (bool), ``collapse_frame`` (offset
    into ``scores`` or None), ``last_good_frame`` (offset or None).
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        return {"collapsed": False, "collapse_frame": None, "last_good_frame": None}

    if min_run is None:
        min_run = max(window, n // 20)  # >= 5% of the video, at least one window

    smoothed = rolling_mean(s, window)
    # NaN (object absent) is treated as "below" so missing tracks count as lost.
    below = ~(smoothed >= threshold)

    collapse_frame: int | None = None
    for t0 in range(n):
        tail = below[t0:]
        if tail.size < min_run:
            break  # remaining tail too short to qualify as a sustained collapse
        if tail.mean() >= persist_frac:
            collapse_frame = t0
            break

    # Last frame whose smoothed score is still good.
    good_idx = np.where(smoothed >= threshold)[0]
    last_good = int(good_idx[-1]) if good_idx.size else None

    return {
        "collapsed": collapse_frame is not None,
        "collapse_frame": collapse_frame,
        "last_good_frame": last_good,
    }


def consecutive_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a 1-D boolean array as (start_index, length)."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([np.int8(0)], m, [np.int8(0)])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def duration_stats(
    frame_indices: np.ndarray,
    series: np.ndarray,
    *,
    threshold: float,
    window: int,
    min_run: int = 150,
    zero_eps: float = 1e-6,
) -> dict:
    """How long an IoU series spends below threshold / at zero, plus sustained collapse.

    Computed for one per-frame series (a reduction over objects, or a single object).
    "any" = every scored frame meeting the condition (no run constraint); "sustained"
    = frames inside runs of >= ``min_run`` consecutive frames on the *rolling-average*
    series (the "amount of time the video is collapsed"). Fractions for the raw/rolling
    counts are over scored frames; collapse fractions are over the full series length.
    NaN (unscored / object absent) frames are excluded from the counts.
    """
    fi = np.asarray(frame_indices)
    v = np.asarray(series, dtype=np.float64)
    n = int(v.size)
    valid = np.isfinite(v)
    nval = int(valid.sum())

    below_raw = valid & (v < threshold)
    zero_raw = valid & (v <= zero_eps)

    roll = rolling_mean(v, window)
    rvalid = np.isfinite(roll)
    nrval = int(rvalid.sum())
    below_roll = rvalid & (roll < threshold)
    zero_roll = rvalid & (roll <= zero_eps)

    def _sustained(mask: np.ndarray) -> tuple[int, int, int, int | None]:
        runs = [(s, ln) for (s, ln) in consecutive_runs(mask) if ln >= min_run]
        total = int(sum(ln for _, ln in runs))
        longest = int(max((ln for _, ln in runs), default=0))
        first = int(fi[runs[0][0]]) if runs else None
        return total, longest, len(runs), first

    sb_total, sb_long, sb_n, sb_first = _sustained(below_roll)
    sz_total, sz_long, sz_n, sz_first = _sustained(zero_roll)
    fin = v[valid]

    return {
        "n_frames": n,
        "n_scored": nval,
        "mean_iou": float(fin.mean()) if nval else float("nan"),
        "median_iou": float(np.median(fin)) if nval else float("nan"),
        "min_run": int(min_run),
        # "any value" conditions on the raw per-frame series (over scored frames)
        "frames_below_thresh": int(below_raw.sum()),
        "frac_below_thresh": float(below_raw.sum() / nval) if nval else float("nan"),
        "frames_zero": int(zero_raw.sum()),
        "frac_zero": float(zero_raw.sum() / nval) if nval else float("nan"),
        # same conditions on the rolling-average series (over rolling-valid frames)
        "roll_frames_below_thresh": int(below_roll.sum()),
        "roll_frac_below_thresh": float(below_roll.sum() / nrval) if nrval else float("nan"),
        "roll_frames_zero": int(zero_roll.sum()),
        "roll_frac_zero": float(zero_roll.sum() / nrval) if nrval else float("nan"),
        # sustained collapse: rolling avg below threshold for >= min_run consecutive frames
        "collapsed_frames": sb_total,
        "collapsed_frac": float(sb_total / n) if n else float("nan"),
        "collapsed_longest_run": sb_long,
        "collapsed_n_runs": sb_n,
        "collapse_start_frame": sb_first,
        # sustained zero: rolling avg ~0 for >= min_run consecutive frames
        "zero_collapsed_frames": sz_total,
        "zero_collapsed_frac": float(sz_total / n) if n else float("nan"),
        "zero_collapsed_longest_run": sz_long,
        "zero_collapsed_n_runs": sz_n,
        "zero_collapse_start_frame": sz_first,
    }


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def summarize_series(
    frame_indices: np.ndarray,
    per_frame_iou: np.ndarray,
    *,
    threshold: float,
    window: int,
    persist_frac: float = 0.8,
) -> dict:
    """Per-video summary statistics over the frame-mean IoU series."""
    frame_indices = np.asarray(frame_indices)
    vals = np.asarray(per_frame_iou, dtype=np.float64)
    finite = _finite(vals)
    n = vals.size

    collapse = detect_collapse(
        vals, threshold=threshold, window=window, persist_frac=persist_frac
    )

    cf = collapse["collapse_frame"]
    if cf is not None:
        pre = _finite(vals[:cf])
        post = _finite(vals[cf:])
        collapse_frame_real = int(frame_indices[cf])
    else:
        pre = finite
        post = np.array([], dtype=np.float64)
        collapse_frame_real = None

    # Area under the IoU-vs-normalized-position curve (trapezoid over valid frames).
    if finite.size >= 2:
        xs = np.linspace(0.0, 1.0, finite.size)
        auc = float(_trapezoid(finite, xs))
    else:
        auc = float(finite.mean()) if finite.size else float("nan")

    lg = collapse["last_good_frame"]
    return {
        "num_frames": int(n),
        "num_scored_frames": int(finite.size),
        "mean_iou": float(finite.mean()) if finite.size else float("nan"),
        "median_iou": float(np.median(finite)) if finite.size else float("nan"),
        "iou_auc": auc,
        "frac_frames_above_threshold": (
            float((finite >= threshold).mean()) if finite.size else float("nan")
        ),
        "collapsed": bool(collapse["collapsed"]),
        "collapse_frame": collapse_frame_real,
        "collapse_fraction": (
            float(cf / n) if (cf is not None and n) else None
        ),
        "frames_after_collapse": (int(n - cf) if cf is not None else 0),
        "last_good_frame": (int(frame_indices[lg]) if lg is not None else None),
        "pre_collapse_mean_iou": float(pre.mean()) if pre.size else float("nan"),
        "post_collapse_mean_iou": float(post.mean()) if post.size else float("nan"),
    }
