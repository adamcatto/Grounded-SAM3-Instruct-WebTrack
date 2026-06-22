"""
Score single-shot tracks against anchor-frame tracks (ground truth).

For every video in the manifest we walk frame-by-frame over the propagated
range, load the predicted (single-shot) masks and the ground-truth (anchor)
masks from their respective ``masks.sqlite`` stores, and compute per-object IoU.
The per-frame mean IoU series drives collapse detection and summary stats.

Outputs (under ``<out_dir>/``):
  * ``per_frame_iou.csv``  — long: video, frame, mean_iou, n_objects
  * ``per_object_iou.csv`` — long: video, frame, object_id, object_name, iou
  * ``video_summary.csv``  — one row per video (mean/median IoU, collapse, …)
  * ``object_summary.csv`` — one row per (video, object)
  * ``results.json``       — config + all summaries
  * ``timelines.npz``      — frame index + mean-IoU arrays per video (for figures)

Results are checkpointed after **every** video: the CSV rows are appended and
``results.json`` / ``timelines.npz`` are rewritten, so partial output is visible
while the job runs and survives a crash/timeout. Re-running resumes by skipping
videos already present in ``video_summary.csv`` (pass ``resume=False`` to redo).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import DEFAULT_IOU_THRESHOLD
from .common import mask_storage
from .metrics import per_object_iou, summarize_series
from .project_builder import load_manifest

# ── CSV column orders ──────────────────────────────────────────────────────
_PER_FRAME_FIELDS = ["video_id", "video_name", "frame", "mean_iou", "n_objects"]
_PER_OBJECT_FIELDS = ["video_id", "video_name", "frame", "object_id", "object_name", "iou"]
_VIDEO_FIELDS = [
    "video_id", "video_name", "source_video_id", "first_frame", "num_frames",
    "num_scored_frames", "window", "mean_iou", "median_iou", "iou_auc",
    "frac_frames_above_threshold", "collapsed", "collapse_frame",
    "collapse_fraction", "frames_after_collapse", "last_good_frame",
    "pre_collapse_mean_iou", "post_collapse_mean_iou",
]
_OBJECT_FIELDS = [
    "video_id", "video_name", "object_id", "object_name", "mean_iou",
    "median_iou", "frac_frames_above_threshold", "collapsed",
    "collapse_frame", "collapse_fraction",
]
# Typed reload (for resume) — which video_summary columns are int / float / bool.
_VS_INT = {"first_frame", "num_frames", "num_scored_frames", "window",
           "collapse_frame", "frames_after_collapse", "last_good_frame"}
_VS_FLOAT = {"mean_iou", "median_iou", "iou_auc", "frac_frames_above_threshold",
             "collapse_fraction", "pre_collapse_mean_iou", "post_collapse_mean_iou"}
_VS_BOOL = {"collapsed"}
_OS_FLOAT = {"mean_iou", "median_iou", "frac_frames_above_threshold", "collapse_fraction"}
_OS_INT = {"collapse_frame"}
_OS_BOOL = {"collapsed"}


def _try_tqdm(iterable, **kw):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kw)
    except Exception:
        return iterable


def _object_name_lookup(source_video_dir: Path, source_video_id: str) -> dict[str, str]:
    """Object id -> name from the source project's config, for one video."""
    # source_video_dir = <source_project>/videos/<vid>_slug
    cfg_path = source_video_dir.parent.parent / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return {}
    vm = (cfg.get("videos") or {}).get(source_video_id) or {}
    objs = vm.get("objects") or {}
    return {str(oid): str(o.get("name") or oid) for oid, o in objs.items()}


def _default_window(num_frames: int) -> int:
    """Smoothing window for collapse detection: ~1% of the video, clamped."""
    return int(np.clip(round(num_frames * 0.01), 15, 300))


def _read_typed_csv(path: Path, int_fields: set, float_fields: set,
                    bool_fields: set) -> list[dict[str, Any]]:
    """Re-load a checkpointed CSV, restoring int/float/bool/None types."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            d: dict[str, Any] = {}
            for k, v in r.items():
                if k in bool_fields:
                    d[k] = (v == "True")
                elif v in ("", "None"):
                    d[k] = None
                elif k in int_fields:
                    d[k] = int(float(v))
                elif k in float_fields:
                    d[k] = float(v)
                else:
                    d[k] = v
            rows.append(d)
    return rows


def _open_csv(path: Path, fieldnames: list[str], append: bool):
    """Open a CSV for writing; write a header only when starting fresh."""
    new = not (append and path.is_file())
    f = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if new:
        writer.writeheader()
        f.flush()
    return f, writer


def evaluate_project(
    project_dir: Path,
    *,
    out_dir: Path | None = None,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    window: int | None = None,
    persist_frac: float = 0.8,
    progress: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    manifest = load_manifest(project_dir)
    out_dir = (out_dir or (project_dir / "vos_comparison_results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_summary_csv = out_dir / "video_summary.csv"
    object_summary_csv = out_dir / "object_summary.csv"
    per_frame_csv = out_dir / "per_frame_iou.csv"
    per_object_csv = out_dir / "per_object_iou.csv"
    timelines_npz = out_dir / "timelines.npz"
    results_json = out_dir / "results.json"

    # ── Resume: load already-scored videos so we can skip + keep aggregates ──
    done_ids: set[str] = set()
    video_summaries: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    timelines: dict[str, np.ndarray] = {}
    if resume:
        video_summaries = _read_typed_csv(video_summary_csv, _VS_INT, _VS_FLOAT, _VS_BOOL)
        object_summaries = _read_typed_csv(object_summary_csv, _OS_INT, _OS_FLOAT, _OS_BOOL)
        done_ids = {str(v["video_id"]) for v in video_summaries}
        if timelines_npz.is_file():
            with np.load(timelines_npz) as z:
                timelines = {k: z[k] for k in z.files}
        if done_ids:
            print(f"[evaluate] resuming: {len(done_ids)} video(s) already scored, will skip them")

    # Persistent append-mode writers (header written once, on fresh start).
    vf_f, vf_w = _open_csv(video_summary_csv, _VIDEO_FIELDS, append=resume)
    of_f, of_w = _open_csv(object_summary_csv, _OBJECT_FIELDS, append=resume)
    pf_f, pf_w = _open_csv(per_frame_csv, _PER_FRAME_FIELDS, append=resume)
    po_f, po_w = _open_csv(per_object_csv, _PER_OBJECT_FIELDS, append=resume)

    def _checkpoint() -> dict[str, Any]:
        """Rewrite results.json + timelines.npz from current in-memory state."""
        agg = _aggregate(video_summaries, threshold)
        res = {
            "source_project": manifest.get("source_project_name"),
            "single_shot_project_id": manifest.get("single_shot_project_id"),
            "threshold": threshold,
            "persist_frac": persist_frac,
            "n_videos": len(video_summaries),
            "aggregate": agg,
            "videos": video_summaries,
            "objects": object_summaries,
        }
        tmp = results_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=2, default=_json_default))
        tmp.replace(results_json)
        if timelines:
            tmp_npz = timelines_npz.with_suffix(".npz.tmp")
            np.savez_compressed(tmp_npz, **timelines)
            tmp_npz.replace(timelines_npz)
        return res

    results: dict[str, Any] = {}
    try:
        for ventry in _try_tqdm(manifest["videos"], desc="Videos", unit="vid", disable=not progress):
            sid = str(ventry["single_shot_video_id"])
            if sid in done_ids:
                continue
            name = ventry["name"]
            first = int(ventry["first_frame"])
            num_frames = int(ventry["num_frames"])
            gt_dir = Path(ventry["source_video_dir"])
            pred_dir = Path(ventry["single_shot_video_dir"])
            gt_store = mask_storage(gt_dir)
            pred_store = mask_storage(pred_dir)
            obj_names = _object_name_lookup(gt_dir, ventry["source_video_id"])

            frame_indices = list(range(first, num_frames))
            win = window if window is not None else _default_window(len(frame_indices))

            mean_iou = np.full(len(frame_indices), np.nan, dtype=np.float64)
            obj_series: dict[str, np.ndarray] = {}
            frame_rows: list[dict[str, Any]] = []
            object_rows: list[dict[str, Any]] = []

            frame_iter = _try_tqdm(
                list(enumerate(frame_indices)),
                desc=f"  {name[:24]}", unit="frm", leave=False, disable=not progress,
            )
            for row, fi in frame_iter:
                if not gt_store.has_masks(fi):
                    continue  # no ground truth at this frame — cannot score
                gt_masks = gt_store.load_masks_dense(fi)
                pred_masks = pred_store.load_masks_dense(fi) if pred_store.has_masks(fi) else {}
                ious = per_object_iou(gt_masks, pred_masks)
                if not ious:
                    continue
                for oid, val in ious.items():
                    if oid not in obj_series:
                        obj_series[oid] = np.full(len(frame_indices), np.nan, dtype=np.float64)
                    obj_series[oid][row] = val
                    object_rows.append({
                        "video_id": sid, "video_name": name, "frame": fi,
                        "object_id": oid, "object_name": obj_names.get(oid, oid), "iou": val,
                    })
                frame_mean = float(np.mean(list(ious.values())))
                mean_iou[row] = frame_mean
                frame_rows.append({
                    "video_id": sid, "video_name": name, "frame": fi,
                    "mean_iou": frame_mean, "n_objects": len(ious),
                })

            fidx_arr = np.asarray(frame_indices, dtype=np.int64)
            summary = summarize_series(
                fidx_arr, mean_iou, threshold=threshold, window=win, persist_frac=persist_frac
            )
            summary.update({
                "video_id": sid, "source_video_id": ventry["source_video_id"],
                "video_name": name, "first_frame": first, "window": win,
            })

            obj_sum_rows: list[dict[str, Any]] = []
            for oid, series in obj_series.items():
                osum = summarize_series(
                    fidx_arr, series, threshold=threshold, window=win, persist_frac=persist_frac
                )
                obj_sum_rows.append({
                    "video_id": sid, "video_name": name, "object_id": oid,
                    "object_name": obj_names.get(oid, oid), "mean_iou": osum["mean_iou"],
                    "median_iou": osum["median_iou"],
                    "frac_frames_above_threshold": osum["frac_frames_above_threshold"],
                    "collapsed": osum["collapsed"], "collapse_frame": osum["collapse_frame"],
                    "collapse_fraction": osum["collapse_fraction"],
                })

            # ── Commit this video: append CSV rows, update memory, checkpoint ──
            pf_w.writerows(frame_rows); pf_f.flush()
            po_w.writerows(object_rows); po_f.flush()
            vf_w.writerow(summary); vf_f.flush()
            of_w.writerows(obj_sum_rows); of_f.flush()

            video_summaries.append(summary)
            object_summaries.extend(obj_sum_rows)
            timelines[f"{sid}__frames"] = fidx_arr
            timelines[f"{sid}__mean_iou"] = mean_iou
            done_ids.add(sid)
            results = _checkpoint()
    finally:
        for fh in (vf_f, of_f, pf_f, po_f):
            fh.close()

    if not results:  # nothing new processed (all skipped) — still emit a checkpoint
        results = _checkpoint()

    agg = results["aggregate"]
    print(f"[evaluate] {len(video_summaries)} video(s) scored → {out_dir}")
    if agg.get("n_videos"):
        print(f"[evaluate] aggregate mean IoU = {agg['mean_iou']:.3f}; "
              f"{agg['n_collapsed']}/{agg['n_videos']} videos collapsed "
              f"(median collapse at {agg['median_collapse_fraction_pct']}% of video).")
    return results


def _aggregate(video_summaries: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not video_summaries:
        return {"n_videos": 0}
    means = np.array([v["mean_iou"] for v in video_summaries], dtype=np.float64)
    aucs = np.array([v["iou_auc"] for v in video_summaries], dtype=np.float64)
    fracs = np.array([v["frac_frames_above_threshold"] for v in video_summaries], dtype=np.float64)
    collapsed = [v for v in video_summaries if v["collapsed"]]
    coll_fracs = np.array(
        [v["collapse_fraction"] for v in collapsed if v["collapse_fraction"] is not None],
        dtype=np.float64,
    )
    median_cf = float(np.median(coll_fracs)) if coll_fracs.size else None
    return {
        "n_videos": len(video_summaries),
        "n_collapsed": len(collapsed),
        "frac_videos_collapsed": len(collapsed) / len(video_summaries),
        "mean_iou": float(np.nanmean(means)),
        "std_iou": float(np.nanstd(means)),
        "mean_iou_auc": float(np.nanmean(aucs)),
        "mean_frac_above_threshold": float(np.nanmean(fracs)),
        "median_collapse_fraction": median_cf,
        "median_collapse_fraction_pct": (round(100 * median_cf, 1) if median_cf is not None else None),
    }


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
