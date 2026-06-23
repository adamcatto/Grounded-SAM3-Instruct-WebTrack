"""
Derive every VOS-comparison analysis from ``per_object_iou.csv`` — no mask decoding.

``evaluation.py`` is the expensive measurement stage: it decodes masks and emits the
per-(video, frame, object) IoU in ``per_object_iou.csv``. Everything downstream is a
cheap re-analysis of that table, so it lives here and can be re-run (with new
thresholds, windows, or metrics) without touching the GPU/mask pipeline.

For each video we build three families of per-frame IoU series:
  * ``mean``   — mean over the objects present in the frame (default video signal)
  * ``min``    — min  over the objects present in the frame (worst-object signal)
  * ``object`` — each object's own IoU timeline

and compute, per series, sustained-collapse / time-below-threshold / time-at-zero
statistics (see ``metrics.duration_stats``), plus the legacy per-video and per-object
collapse summaries. Cross-video aggregates and the ``timelines.npz`` consumed by the
figures package are written too.

Outputs (under ``<out_dir>/``):
  * ``video_summary.csv``    — one row per video   (mean-reduction collapse summary)
  * ``object_summary.csv``   — one row per (video, object)
  * ``collapse_stats.csv``   — one row per (video, reduction); the full duration stats
  * ``aggregate_stats.csv``  — one row per (reduction, metric); cross-video summary
  * ``timelines.npz``        — per-video frame axis + mean/min/per-object IoU series
  * ``results.json``         — config + headline aggregate + per-video summaries
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import DEFAULT_IOU_THRESHOLD
from .evaluation import (
    _OBJECT_FIELDS,
    _VIDEO_FIELDS,
    _aggregate,
    _default_window,
    _json_default,
)
from .metrics import duration_stats, summarize_series
from .project_builder import load_manifest

DEFAULT_MIN_RUN = 150

# collapse_stats.csv column order
_REDUCTION_META = ["video_id", "video_name", "reduction", "object_id", "object_name"]
_DUR_FIELDS = [
    "n_frames", "n_scored", "mean_iou", "median_iou", "min_run",
    "frames_below_thresh", "frac_below_thresh", "frames_zero", "frac_zero",
    "roll_frames_below_thresh", "roll_frac_below_thresh",
    "roll_frames_zero", "roll_frac_zero",
    "collapsed_frames", "collapsed_frac", "collapsed_longest_run",
    "collapsed_n_runs", "collapse_start_frame",
    "zero_collapsed_frames", "zero_collapsed_frac", "zero_collapsed_longest_run",
    "zero_collapsed_n_runs", "zero_collapse_start_frame",
]
_COLLAPSE_FIELDS = _REDUCTION_META + _DUR_FIELDS

# aggregate_stats.csv: which collapse_stats metrics to summarize across videos
_AGG_METRICS = [
    "mean_iou", "median_iou",
    "frac_below_thresh", "frac_zero", "roll_frac_below_thresh", "roll_frac_zero",
    "collapsed_frames", "collapsed_frac", "collapsed_longest_run", "collapsed_n_runs",
    "zero_collapsed_frames", "zero_collapsed_frac",
    "zero_collapsed_longest_run", "zero_collapsed_n_runs",
]
_AGG_FIELDS = ["reduction", "metric", "n", "mean", "std", "median", "q1", "q3", "min", "max"]

REDUCTIONS = ["mean", "min", "object"]


def _try_tqdm(iterable, **kw):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kw)
    except Exception:
        return iterable


def _obj_sort_key(oid: str):
    try:
        return (0, int(oid))
    except (TypeError, ValueError):
        return (1, str(oid))


def _load_per_object(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Group per_object_iou.csv into {video_id: {name, objects: {oid: {name, frames, ious}}}}."""
    videos: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            vid = r["video_id"]
            oid = r["object_id"]
            v = videos.setdefault(vid, {"name": r.get("video_name") or vid, "objects": {}})
            o = v["objects"].setdefault(oid, {"name": r.get("object_name") or oid,
                                              "frames": [], "ious": []})
            o["frames"].append(int(r["frame"]))
            o["ious"].append(float(r["iou"]))
    return videos


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _summary_stats(values: list[float]) -> dict[str, Any]:
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def analyze_project(
    project_dir: Path,
    *,
    out_dir: Path | None = None,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    window: int | None = None,
    persist_frac: float = 0.8,
    min_run: int = DEFAULT_MIN_RUN,
    progress: bool = True,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    manifest = load_manifest(project_dir)
    out_dir = (out_dir or (project_dir / "vos_comparison_results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_object_csv = out_dir / "per_object_iou.csv"
    if not per_object_csv.is_file():
        raise FileNotFoundError(
            f"{per_object_csv} not found — run `evaluate` (and `merge` if sharded) first."
        )

    data = _load_per_object(per_object_csv)
    # Manifest gives the canonical frame range + ordering.
    meta = {
        str(v["single_shot_video_id"]): (int(v["first_frame"]), int(v["num_frames"]), v["name"])
        for v in manifest["videos"]
    }

    video_summaries: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    timelines: dict[str, np.ndarray] = {}
    ordered_vids: list[str] = []

    for ventry in _try_tqdm(manifest["videos"], desc="Analyze", unit="vid", disable=not progress):
        vid = str(ventry["single_shot_video_id"])
        if vid not in data:
            continue
        first, num_frames, name = meta[vid]
        axis = np.arange(first, num_frames, dtype=np.int64)
        if axis.size == 0:
            continue
        pos = {int(fr): i for i, fr in enumerate(axis)}

        objs = data[vid]["objects"]
        oids = sorted(objs.keys(), key=_obj_sort_key)
        obj_arrays: dict[str, np.ndarray] = {}
        for oid in oids:
            arr = np.full(axis.size, np.nan, dtype=np.float64)
            for fr, iou in zip(objs[oid]["frames"], objs[oid]["ious"]):
                i = pos.get(int(fr))
                if i is not None:
                    arr[i] = iou
            obj_arrays[oid] = arr

        stack = np.vstack([obj_arrays[o] for o in oids])  # (n_objects, T)
        with warnings.catch_warnings():  # all-NaN columns are expected (no object present)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_series = np.nanmean(stack, axis=0)
            min_series = np.nanmin(stack, axis=0)

        win = window if window is not None else _default_window(axis.size)

        # ── legacy per-video (mean reduction) + per-object collapse summaries ──
        vsum = summarize_series(axis, mean_series, threshold=threshold,
                                window=win, persist_frac=persist_frac)
        vsum.update({
            "video_id": vid, "source_video_id": ventry["source_video_id"],
            "video_name": name, "first_frame": first, "window": win,
        })
        video_summaries.append(vsum)

        for oid in oids:
            oname = objs[oid]["name"]
            osum = summarize_series(axis, obj_arrays[oid], threshold=threshold,
                                    window=win, persist_frac=persist_frac)
            object_summaries.append({
                "video_id": vid, "video_name": name, "object_id": oid, "object_name": oname,
                "mean_iou": osum["mean_iou"], "median_iou": osum["median_iou"],
                "frac_frames_above_threshold": osum["frac_frames_above_threshold"],
                "collapsed": osum["collapsed"], "collapse_frame": osum["collapse_frame"],
                "collapse_fraction": osum["collapse_fraction"],
            })

        # ── duration / sustained-collapse stats per reduction ────────────────
        def _row(reduction, series, object_id="", object_name=""):
            d = duration_stats(axis, series, threshold=threshold, window=win, min_run=min_run)
            d.update({"video_id": vid, "video_name": name, "reduction": reduction,
                      "object_id": object_id, "object_name": object_name})
            return d

        collapse_rows.append(_row("mean", mean_series))
        collapse_rows.append(_row("min", min_series))
        for oid in oids:
            collapse_rows.append(_row("object", obj_arrays[oid], oid, objs[oid]["name"]))

        # ── timelines for the figures package ────────────────────────────────
        ordered_vids.append(vid)
        timelines[f"{vid}__frames"] = axis
        timelines[f"{vid}__mean"] = mean_series
        timelines[f"{vid}__min"] = min_series
        timelines[f"{vid}__objids"] = np.array(oids, dtype=object).astype("U")
        timelines[f"{vid}__objnames"] = np.array([objs[o]["name"] for o in oids],
                                                 dtype=object).astype("U")
        for oid in oids:
            timelines[f"{vid}__obj__{oid}"] = obj_arrays[oid]

    # ── cross-video aggregate_stats (per reduction × metric) ─────────────────
    by_reduction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in collapse_rows:
        by_reduction[r["reduction"]].append(r)
    aggregate_rows: list[dict[str, Any]] = []
    for reduction in REDUCTIONS:
        rows = by_reduction.get(reduction, [])
        for metric in _AGG_METRICS:
            st = _summary_stats([r.get(metric) for r in rows])
            aggregate_rows.append({"reduction": reduction, "metric": metric, **st})

    # ── write everything ─────────────────────────────────────────────────────
    _write_csv(out_dir / "video_summary.csv", video_summaries, _VIDEO_FIELDS)
    _write_csv(out_dir / "object_summary.csv", object_summaries, _OBJECT_FIELDS)
    _write_csv(out_dir / "collapse_stats.csv", collapse_rows, _COLLAPSE_FIELDS)
    _write_csv(out_dir / "aggregate_stats.csv", aggregate_rows, _AGG_FIELDS)
    if timelines:
        timelines["__video_ids__"] = np.array(ordered_vids, dtype=object).astype("U")
        np.savez_compressed(out_dir / "timelines.npz", **timelines)

    agg = _aggregate(video_summaries, threshold)
    results = {
        "source_project": manifest.get("source_project_name"),
        "single_shot_project_id": manifest.get("single_shot_project_id"),
        "threshold": threshold,
        "persist_frac": persist_frac,
        "min_run": min_run,
        "reductions": REDUCTIONS,
        "n_videos": len(video_summaries),
        "aggregate": agg,
        "videos": video_summaries,
        "objects": object_summaries,
        "aggregate_stats": aggregate_rows,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))

    print(f"[analyze] {len(video_summaries)} video(s), "
          f"{sum(1 for r in collapse_rows if r['reduction'] == 'object')} object series → {out_dir}")
    if agg.get("n_videos"):
        print(f"[analyze] mean-reduction: mean IoU={agg['mean_iou']:.3f}, "
              f"{agg['n_collapsed']}/{agg['n_videos']} videos collapsed.")
        for reduction in REDUCTIONS:
            rows = by_reduction.get(reduction, [])
            cf = _summary_stats([r["collapsed_frac"] for r in rows])
            if cf["n"]:
                print(f"[analyze]   {reduction:6s}: median time-collapsed = "
                      f"{100 * cf['median']:.1f}% of video (n={cf['n']})")
    return results
