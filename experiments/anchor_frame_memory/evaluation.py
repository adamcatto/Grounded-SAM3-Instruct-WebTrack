"""Evaluate recent queue memory against anchor-frame memory tracking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import DEFAULT_IOU_THRESHOLD, RESULTS_DIRNAME
from .common import load_config, mask_storage, object_names
from .project_builder import load_manifest
from experiments.vos_comparison.metrics import mask_iou, summarize_series


PER_FRAME_FIELDS = [
    "video_id", "video_name", "frame", "mean_iou", "min_iou",
    "n_anchor_objects", "n_queue_objects", "n_missing_objects",
]
PER_OBJECT_FIELDS = [
    "video_id", "video_name", "frame", "object_id", "object_name",
    "anchor_present", "queue_present", "anchor_area", "queue_area", "iou",
    "entered_after_absence",
]
OBJECT_SUMMARY_FIELDS = [
    "video_id", "video_name", "object_id", "object_name",
    "n_present_anchor_frames", "n_present_queue_frames", "n_missing_frames",
    "mean_iou", "median_iou", "frac_frames_above_threshold",
    "frac_present_frames_missed", "reentry_events", "missed_reentry_events",
    "mean_iou_on_reentry_window", "missed_frames_after_reentry",
]
VIDEO_SUMMARY_FIELDS = [
    "video_id", "video_name", "num_frames", "num_scored_frames",
    "mean_iou", "median_iou", "mean_min_iou", "frac_frames_above_threshold",
    "total_missing_object_frames", "total_reentry_events", "missed_reentry_events",
]
REENTRY_FIELDS = [
    "video_id", "video_name", "object_id", "object_name", "frame",
    "absence_length", "window", "mean_iou_window", "missed_frames_window",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _base_id(key: str) -> str:
    return str(key).split("_")[0]


def _merge_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, mask in masks.items():
        base = _base_id(key)
        mm = np.squeeze(np.asarray(mask)).astype(bool)
        if base in out:
            out[base] = np.logical_or(out[base], mm)
        else:
            out[base] = mm
    return out


def _safe_stats(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(np.median(arr))


def _entry_flags(present: np.ndarray, min_absence_frames: int) -> tuple[np.ndarray, dict[int, int]]:
    flags = np.zeros(present.size, dtype=bool)
    absence_lengths: dict[int, int] = {}
    absent_run = 0
    seen_present = False
    for i, p in enumerate(present):
        if p:
            if seen_present and absent_run >= min_absence_frames:
                flags[i] = True
                absence_lengths[i] = absent_run
            seen_present = True
            absent_run = 0
        else:
            absent_run += 1
    return flags, absence_lengths


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def evaluate_project(
    queue_project_dir: Path,
    *,
    out_dir: Path | None = None,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    min_absence_frames: int = 30,
    reentry_window: int = 90,
) -> dict[str, Any]:
    queue_project_dir = queue_project_dir.resolve()
    manifest = load_manifest(queue_project_dir)
    source_dir = Path(manifest["source_project_dir"]).resolve()
    source_cfg = load_config(source_dir)
    out_dir = (out_dir or queue_project_dir / RESULTS_DIRNAME).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows: list[dict[str, Any]] = []
    per_object_rows: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    timelines: dict[str, np.ndarray] = {}
    ordered_videos: list[str] = []

    for v in manifest["videos"]:
        vid = str(v["video_id"])
        name = str(v["video_name"])
        start = int(v.get("start_frame") or 0)
        num_frames = int(v["num_frames"])
        frames = np.arange(start, num_frames, dtype=np.int64)
        anchor_store = mask_storage(Path(v["source_video_dir"]))
        queue_store = mask_storage(Path(v["queue_video_dir"]))
        names = object_names(source_cfg, str(v["source_video_id"]))
        expected_oids = sorted(names.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

        obj_iou = {oid: np.full(frames.size, np.nan, dtype=np.float64) for oid in expected_oids}
        obj_anchor_present = {oid: np.zeros(frames.size, dtype=bool) for oid in expected_oids}
        obj_queue_present = {oid: np.zeros(frames.size, dtype=bool) for oid in expected_oids}
        obj_anchor_area = {oid: np.zeros(frames.size, dtype=np.int64) for oid in expected_oids}
        obj_queue_area = {oid: np.zeros(frames.size, dtype=np.int64) for oid in expected_oids}
        mean_series = np.full(frames.size, np.nan, dtype=np.float64)
        min_series = np.full(frames.size, np.nan, dtype=np.float64)

        for row, frame in enumerate(frames):
            if not anchor_store.has_masks(int(frame)):
                continue
            anchor = _merge_masks(anchor_store.load_masks_dense(int(frame)))
            queue = _merge_masks(queue_store.load_masks_dense(int(frame))) if queue_store.has_masks(int(frame)) else {}
            frame_ious: list[float] = []
            present_anchor = 0
            present_queue = 0
            missing = 0
            oids = sorted(set(expected_oids) | set(anchor.keys()), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
            for oid in oids:
                am = anchor.get(oid)
                qm = queue.get(oid)
                a_area = int(am.sum()) if am is not None else 0
                q_area = int(qm.sum()) if qm is not None else 0
                a_present = a_area > 0
                q_present = q_area > 0
                if a_present:
                    present_anchor += 1
                    if oid not in obj_iou:
                        obj_iou[oid] = np.full(frames.size, np.nan, dtype=np.float64)
                        obj_anchor_present[oid] = np.zeros(frames.size, dtype=bool)
                        obj_queue_present[oid] = np.zeros(frames.size, dtype=bool)
                        obj_anchor_area[oid] = np.zeros(frames.size, dtype=np.int64)
                        obj_queue_area[oid] = np.zeros(frames.size, dtype=np.int64)
                    val = mask_iou(am, qm) if qm is not None else 0.0
                    obj_iou[oid][row] = val
                    frame_ious.append(val)
                    if not q_present:
                        missing += 1
                if q_present:
                    present_queue += 1
                if oid in obj_iou:
                    obj_anchor_present[oid][row] = a_present
                    obj_queue_present[oid][row] = q_present
                    obj_anchor_area[oid][row] = a_area
                    obj_queue_area[oid][row] = q_area

            if frame_ious:
                mean_series[row] = float(np.mean(frame_ious))
                min_series[row] = float(np.min(frame_ious))
                per_frame_rows.append({
                    "video_id": vid, "video_name": name, "frame": int(frame),
                    "mean_iou": mean_series[row], "min_iou": min_series[row],
                    "n_anchor_objects": present_anchor, "n_queue_objects": present_queue,
                    "n_missing_objects": missing,
                })

        total_reentries = 0
        missed_reentries = 0
        total_missing_object_frames = 0
        for oid in sorted(obj_iou.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            flags, absence_lengths = _entry_flags(obj_anchor_present[oid], min_absence_frames)
            total_reentries += int(flags.sum())
            vals = obj_iou[oid]
            present = obj_anchor_present[oid]
            queue_present = obj_queue_present[oid]
            missing_mask = present & ~queue_present
            total_missing_object_frames += int(missing_mask.sum())
            missed_obj_reentries = 0
            missed_after = 0
            window_iou_vals: list[float] = []
            for idx in np.flatnonzero(flags):
                hi = min(frames.size, idx + reentry_window)
                win_present = present[idx:hi]
                win_queue = queue_present[idx:hi]
                win_iou = vals[idx:hi]
                miss_count = int((win_present & ~win_queue).sum())
                missed_after += miss_count
                finite = win_iou[np.isfinite(win_iou)]
                mean_win = float(finite.mean()) if finite.size else float("nan")
                window_iou_vals.append(mean_win)
                if miss_count > 0:
                    missed_obj_reentries += 1
                reentry_rows.append({
                    "video_id": vid, "video_name": name, "object_id": oid,
                    "object_name": names.get(oid, oid), "frame": int(frames[idx]),
                    "absence_length": int(absence_lengths.get(int(idx), 0)),
                    "window": int(hi - idx), "mean_iou_window": mean_win,
                    "missed_frames_window": miss_count,
                })
            missed_reentries += missed_obj_reentries

            for row, frame in enumerate(frames):
                if not present[row]:
                    continue
                per_object_rows.append({
                    "video_id": vid, "video_name": name, "frame": int(frame),
                    "object_id": oid, "object_name": names.get(oid, oid),
                    "anchor_present": bool(present[row]),
                    "queue_present": bool(queue_present[row]),
                    "anchor_area": int(obj_anchor_area[oid][row]),
                    "queue_area": int(obj_queue_area[oid][row]),
                    "iou": float(vals[row]) if np.isfinite(vals[row]) else "",
                    "entered_after_absence": bool(flags[row]),
                })
            mean_iou, med_iou = _safe_stats(vals[present].tolist())
            above = (vals[present] >= threshold)
            object_summaries.append({
                "video_id": vid, "video_name": name, "object_id": oid,
                "object_name": names.get(oid, oid),
                "n_present_anchor_frames": int(present.sum()),
                "n_present_queue_frames": int((present & queue_present).sum()),
                "n_missing_frames": int(missing_mask.sum()),
                "mean_iou": mean_iou,
                "median_iou": med_iou,
                "frac_frames_above_threshold": float(above.mean()) if above.size else float("nan"),
                "frac_present_frames_missed": float(missing_mask.sum() / present.sum()) if present.sum() else float("nan"),
                "reentry_events": int(flags.sum()),
                "missed_reentry_events": int(missed_obj_reentries),
                "mean_iou_on_reentry_window": _safe_stats(window_iou_vals)[0],
                "missed_frames_after_reentry": int(missed_after),
            })

        finite_mean = mean_series[np.isfinite(mean_series)]
        finite_min = min_series[np.isfinite(min_series)]
        video_summaries.append({
            "video_id": vid, "video_name": name, "num_frames": num_frames,
            "num_scored_frames": int(finite_mean.size),
            "mean_iou": float(finite_mean.mean()) if finite_mean.size else float("nan"),
            "median_iou": float(np.median(finite_mean)) if finite_mean.size else float("nan"),
            "mean_min_iou": float(finite_min.mean()) if finite_min.size else float("nan"),
            "frac_frames_above_threshold": float((finite_mean >= threshold).mean()) if finite_mean.size else float("nan"),
            "total_missing_object_frames": int(total_missing_object_frames),
            "total_reentry_events": int(total_reentries),
            "missed_reentry_events": int(missed_reentries),
        })

        ordered_videos.append(vid)
        timelines[f"{vid}__frames"] = frames
        timelines[f"{vid}__mean"] = mean_series
        timelines[f"{vid}__min"] = min_series
        timelines[f"{vid}__objids"] = np.asarray(sorted(obj_iou.keys()), dtype="U")
        timelines[f"{vid}__objnames"] = np.asarray([names.get(o, o) for o in sorted(obj_iou.keys())], dtype="U")
        for oid, arr in obj_iou.items():
            timelines[f"{vid}__obj__{oid}"] = arr
            timelines[f"{vid}__present__{oid}"] = obj_anchor_present[oid].astype(np.uint8)
            timelines[f"{vid}__queue_present__{oid}"] = obj_queue_present[oid].astype(np.uint8)

    _write_csv(out_dir / "per_frame_iou.csv", per_frame_rows, PER_FRAME_FIELDS)
    _write_csv(out_dir / "per_object_iou.csv", per_object_rows, PER_OBJECT_FIELDS)
    _write_csv(out_dir / "object_summary.csv", object_summaries, OBJECT_SUMMARY_FIELDS)
    _write_csv(out_dir / "video_summary.csv", video_summaries, VIDEO_SUMMARY_FIELDS)
    _write_csv(out_dir / "reentry_events.csv", reentry_rows, REENTRY_FIELDS)
    if timelines:
        timelines["__video_ids__"] = np.asarray(ordered_videos, dtype="U")
        np.savez_compressed(out_dir / "timelines.npz", **timelines)

    agg = _aggregate(video_summaries, object_summaries)
    results = {
        "experiment": "anchor_frame_memory",
        "source_project": manifest.get("source_project_name"),
        "source_project_dir": manifest.get("source_project_dir"),
        "queue_project_id": manifest.get("queue_project_id"),
        "queue_project_dir": manifest.get("queue_project_dir"),
        "threshold": threshold,
        "min_absence_frames": min_absence_frames,
        "reentry_window": reentry_window,
        "aggregate": agg,
        "videos": video_summaries,
        "objects": object_summaries,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"[evaluate] wrote {out_dir}")
    print(f"[evaluate] mean IoU={agg.get('mean_iou'):.3f}; missing object-frames={agg.get('total_missing_object_frames')}")
    return results


def _aggregate(video_rows: list[dict[str, Any]], object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not video_rows:
        return {"n_videos": 0}
    means = np.asarray([r["mean_iou"] for r in video_rows], dtype=np.float64)
    obj_means = np.asarray([r["mean_iou"] for r in object_rows], dtype=np.float64) if object_rows else np.asarray([])
    return {
        "n_videos": len(video_rows),
        "n_objects": len(object_rows),
        "mean_iou": float(np.nanmean(means)),
        "median_video_iou": float(np.nanmedian(means)),
        "mean_object_iou": float(np.nanmean(obj_means)) if obj_means.size else float("nan"),
        "total_missing_object_frames": int(sum(r["total_missing_object_frames"] for r in video_rows)),
        "total_reentry_events": int(sum(r["total_reentry_events"] for r in video_rows)),
        "missed_reentry_events": int(sum(r["missed_reentry_events"] for r in video_rows)),
    }
