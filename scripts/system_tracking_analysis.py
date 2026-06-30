#!/usr/bin/env python3
"""System-level tracking analysis for one Grounded-SAM3 video directory.

Outputs CSV tables and PNG/PDF/SVG figures covering labeling efficiency,
tracking coverage, and saved-frame throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FIG_FORMATS = ("png", "pdf", "svg")
NEAR_BBOX_GAP_THRESHOLD_NORM = 0.02
ZERO_BBOX_GAP_EPS = 1e-12


@dataclass(frozen=True)
class VideoContext:
    video_dir: Path
    output_dir: Path
    project_dir: Path | None
    video_id: str
    video_meta: dict[str, Any]


def _video_id_from_dir(video_dir: Path) -> str:
    return video_dir.name.split("_", 1)[0]


def _find_project_dir(video_dir: Path) -> Path | None:
    for parent in [video_dir.parent, *video_dir.parents]:
        cfg = parent / "config.json"
        if cfg.is_file():
            return parent
    return None


def _load_video_meta(project_dir: Path | None, video_id: str, video_dir: Path) -> dict[str, Any]:
    if project_dir is None:
        return {}
    cfg_path = project_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    config = json.loads(cfg_path.read_text())
    videos = config.get("videos") or {}
    if video_id in videos and isinstance(videos[video_id], dict):
        return dict(videos[video_id])
    for key, meta in videos.items():
        if not isinstance(meta, dict):
            continue
        if str(key).startswith(video_id) or video_dir.name.startswith(str(key)):
            return dict(meta)
    return {}


def resolve_context(video_ref: str, output_dir: str | None) -> VideoContext:
    path = Path(video_ref).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Video directory not found: {path}")
    project_dir = _find_project_dir(path)
    video_id = _video_id_from_dir(path)
    video_meta = _load_video_meta(project_dir, video_id, path)
    out = Path(output_dir).expanduser().resolve() if output_dir else path / "system_analysis"
    out.mkdir(parents=True, exist_ok=True)
    return VideoContext(
        video_dir=path,
        output_dir=out,
        project_dir=project_dir,
        video_id=video_id,
        video_meta=video_meta,
    )


def read_sqlite_frames(sqlite_path: Path) -> pd.DataFrame:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Missing masks.sqlite: {sqlite_path}")
    with sqlite3.connect(str(sqlite_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT frame_idx, length(seg_blob) AS seg_blob_bytes, bbox_json, updated_at
            FROM frame_segmentation
            ORDER BY frame_idx
            """,
            conn,
        )
    if df.empty:
        return df
    df["updated_at"] = pd.to_datetime(df["updated_at"], format="%Y-%m-%d %H:%M:%S")
    df["object_count"] = df["bbox_json"].map(_bbox_object_count)
    return df.drop(columns=["bbox_json"])


def _bbox_object_count(raw: str) -> int:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return 0
    return len(value) if isinstance(value, dict) else 0


def _bbox_pair_metrics(bboxes: dict[str, Any], width: float, height: float) -> dict[str, Any]:
    boxes: list[tuple[str, float, float, float, float]] = []
    for oid, raw in bboxes.items():
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            x, y, w, h = [float(v) for v in raw[:4]]
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((str(oid), x, y, w, h))

    if len(boxes) < 2 or width <= 0 or height <= 0:
        return {
            "bbox_pair_count": 0,
            "nearest_bbox_pair": "",
            "min_bbox_center_distance_px": math.nan,
            "min_bbox_center_distance_norm": math.nan,
            "min_bbox_edge_gap_px": math.nan,
            "min_bbox_edge_gap_norm": math.nan,
        }

    diag = math.hypot(width, height)
    best: dict[str, Any] | None = None
    pair_count = 0
    for i, (oid_a, ax, ay, aw, ah) in enumerate(boxes):
        for oid_b, bx, by, bw, bh in boxes[i + 1:]:
            pair_count += 1
            acx, acy = ax + aw / 2.0, ay + ah / 2.0
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            center_px = math.hypot((acx - bcx) * width, (acy - bcy) * height)

            gap_x_norm = max(ax - (bx + bw), bx - (ax + aw), 0.0)
            gap_y_norm = max(ay - (by + bh), by - (ay + ah), 0.0)
            edge_gap_px = math.hypot(gap_x_norm * width, gap_y_norm * height)
            row = {
                "bbox_pair_count": pair_count,
                "nearest_bbox_pair": f"{oid_a}:{oid_b}",
                "min_bbox_center_distance_px": center_px,
                "min_bbox_center_distance_norm": center_px / diag if diag > 0 else math.nan,
                "min_bbox_edge_gap_px": edge_gap_px,
                "min_bbox_edge_gap_norm": edge_gap_px / diag if diag > 0 else math.nan,
            }
            if best is None or row["min_bbox_edge_gap_px"] < best["min_bbox_edge_gap_px"]:
                best = row

    if best is None:
        return {
            "bbox_pair_count": 0,
            "nearest_bbox_pair": "",
            "min_bbox_center_distance_px": math.nan,
            "min_bbox_center_distance_norm": math.nan,
            "min_bbox_edge_gap_px": math.nan,
            "min_bbox_edge_gap_norm": math.nan,
        }
    best["bbox_pair_count"] = pair_count
    return best


def read_bbox_separation(sqlite_path: Path, frame_indices: list[int], width: float, height: float) -> dict[int, dict[str, Any]]:
    if not sqlite_path.is_file() or not frame_indices:
        return {}
    unique_frames = sorted(set(int(f) for f in frame_indices))
    placeholders = ",".join("?" for _ in unique_frames)
    query = f"SELECT frame_idx, bbox_json FROM frame_segmentation WHERE frame_idx IN ({placeholders})"
    out: dict[int, dict[str, Any]] = {}
    with sqlite3.connect(str(sqlite_path)) as conn:
        for frame_idx, raw in conn.execute(query, unique_frames):
            try:
                bboxes = json.loads(raw or "{}")
            except json.JSONDecodeError:
                bboxes = {}
            out[int(frame_idx)] = _bbox_pair_metrics(bboxes if isinstance(bboxes, dict) else {}, width, height)
    return out


def read_progress_frames(path: Path) -> list[int]:
    if not path.is_file():
        return []
    frames: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frames.append(int(line))
        except ValueError:
            continue
    return frames


def prompt_labeled_frames(video_meta: dict[str, Any]) -> set[int]:
    frames: set[int] = set()
    for obj_prompts in (video_meta.get("point_prompts") or {}).values():
        if not isinstance(obj_prompts, dict):
            continue
        for frame_key in obj_prompts:
            try:
                frames.add(int(frame_key))
            except (TypeError, ValueError):
                continue
    return frames


def annotated_anchor_frames(video_meta: dict[str, Any]) -> list[int]:
    anchors: set[int] = set()
    for value in video_meta.get("annotated_anchors") or []:
        try:
            anchors.add(int(value))
        except (TypeError, ValueError):
            continue
    anchors |= prompt_labeled_frames(video_meta)
    return sorted(anchors)


def prompt_counts_by_frame(video_meta: dict[str, Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for obj_prompts in (video_meta.get("point_prompts") or {}).values():
        if not isinstance(obj_prompts, dict):
            continue
        for frame_key, prompts in obj_prompts.items():
            try:
                frame_idx = int(frame_key)
            except (TypeError, ValueError):
                continue
            if isinstance(prompts, list):
                counts[frame_idx] += len(prompts)
            elif prompts:
                counts[frame_idx] += 1
    return counts


def labeling_timing_rows(video_meta: dict[str, Any]) -> list[dict[str, Any]]:
    timing = video_meta.get("anchor_labeling_timing") or {}
    frames = timing.get("frames") or {}
    prompt_counts = prompt_counts_by_frame(video_meta)
    rows: list[dict[str, Any]] = []
    for frame_key, info in frames.items():
        if not isinstance(info, dict):
            continue
        try:
            frame_idx = int(frame_key)
        except (TypeError, ValueError):
            continue
        entered = info.get("entered_frontend_ms")
        committed = info.get("committed_ms")
        duration = info.get("duration_ms")
        try:
            entered_i = int(entered) if entered is not None else None
            committed_i = int(committed) if committed is not None else None
            duration_i = int(duration) if duration is not None else (
                committed_i - entered_i if entered_i is not None and committed_i is not None else None
            )
        except (TypeError, ValueError):
            continue
        if duration_i is None:
            continue
        rows.append(
            {
                "frame_idx": frame_idx,
                "entered_frontend_ms": entered_i,
                "committed_ms": committed_i,
                "duration_ms": duration_i,
                "duration_s": duration_i / 1000.0,
                "prompt_count": int(prompt_counts.get(frame_idx, 0)),
            }
        )
    rows.sort(key=lambda r: (r["entered_frontend_ms"] if r["entered_frontend_ms"] is not None else 10**30, r["frame_idx"]))
    first_entered = next((r["entered_frontend_ms"] for r in rows if r["entered_frontend_ms"] is not None), None)
    for i, row in enumerate(rows, start=1):
        row["anchor_number"] = i
        if first_entered is not None and row["entered_frontend_ms"] is not None:
            row["elapsed_start_s"] = (row["entered_frontend_ms"] - first_entered) / 1000.0
        else:
            row["elapsed_start_s"] = math.nan
        if first_entered is not None and row["committed_ms"] is not None:
            row["elapsed_commit_s"] = (row["committed_ms"] - first_entered) / 1000.0
        else:
            row["elapsed_commit_s"] = math.nan
        row["cumulative_anchors_committed"] = i
    return rows


def infer_anchor_interval(anchors: list[int]) -> float:
    if len(anchors) < 2:
        return math.nan
    diffs = np.diff(np.asarray(anchors, dtype=float))
    positive = diffs[diffs > 0]
    return float(np.median(positive)) if positive.size else math.nan


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for fmt in FIG_FORMATS:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _duration_group_stats(durations: np.ndarray, gaps: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(durations) & np.isfinite(gaps)
    d = durations[valid]
    g = gaps[valid]

    def values(mask: np.ndarray) -> np.ndarray:
        return d[mask]

    def count(mask: np.ndarray) -> int:
        return int(np.sum(mask))

    def median(mask: np.ndarray) -> float:
        return float(np.median(d[mask])) if np.any(mask) else math.nan

    def mean(mask: np.ndarray) -> float:
        vals = values(mask)
        return float(np.mean(vals)) if vals.size else math.nan

    def std(mask: np.ndarray) -> float:
        vals = values(mask)
        return float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else math.nan)

    overlap = g <= ZERO_BBOX_GAP_EPS
    nonzero = g > ZERO_BBOX_GAP_EPS
    close = g < NEAR_BBOX_GAP_THRESHOLD_NORM
    separated = g > NEAR_BBOX_GAP_THRESHOLD_NORM
    overlap_nonzero = _compare_samples(values(overlap), values(nonzero))
    close_separated = _compare_samples(values(close), values(separated))
    return {
        "labeling_overlap_bbox_count": count(overlap),
        "labeling_nonzero_bbox_gap_count": count(nonzero),
        "labeling_close_bbox_gap_count_lt_0p02": count(close),
        "labeling_separated_bbox_gap_count_gt_0p02": count(separated),
        "labeling_duration_mean_overlap_bbox_s": mean(overlap),
        "labeling_duration_std_overlap_bbox_s": std(overlap),
        "labeling_duration_mean_nonzero_bbox_gap_s": mean(nonzero),
        "labeling_duration_std_nonzero_bbox_gap_s": std(nonzero),
        "labeling_duration_mean_close_bbox_gap_lt_0p02_s": mean(close),
        "labeling_duration_std_close_bbox_gap_lt_0p02_s": std(close),
        "labeling_duration_mean_separated_bbox_gap_gt_0p02_s": mean(separated),
        "labeling_duration_std_separated_bbox_gap_gt_0p02_s": std(separated),
        "labeling_duration_median_overlap_bbox_s": median(overlap),
        "labeling_duration_median_nonzero_bbox_gap_s": median(nonzero),
        "labeling_duration_median_close_bbox_gap_lt_0p02_s": median(close),
        "labeling_duration_median_separated_bbox_gap_gt_0p02_s": median(separated),
        "labeling_duration_overlap_vs_nonzero_welch_p": overlap_nonzero["welch_p"],
        "labeling_duration_overlap_vs_nonzero_mannwhitney_p": overlap_nonzero["mannwhitney_p"],
        "labeling_duration_close_lt_0p02_vs_gt_0p02_welch_p": close_separated["welch_p"],
        "labeling_duration_close_lt_0p02_vs_gt_0p02_mannwhitney_p": close_separated["mannwhitney_p"],
    }


def _compare_samples(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"welch_p": math.nan, "mannwhitney_p": math.nan}
    return {
        "welch_p": float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue),
        "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
    }


def _format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, p: float, h_frac: float = 0.035) -> None:
    ymin, ymax = ax.get_ylim()
    h = max((ymax - ymin) * h_frac, 0.5)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + h, _format_p_value(p), ha="center", va="bottom", fontsize=8)
    if y + h > ymax:
        ax.set_ylim(ymin, y + h * 1.25)


def build_summary(ctx: VideoContext, frames: pd.DataFrame, progress_frames: list[int]) -> dict[str, Any]:
    meta = ctx.video_meta
    num_frames = int(meta.get("num_frames") or (int(frames["frame_idx"].max()) + 1 if not frames.empty else 0))
    fps = float(meta.get("fps") or 0.0)
    anchors = annotated_anchor_frames(meta)
    prompt_frames = prompt_labeled_frames(meta)
    prompt_counts = prompt_counts_by_frame(meta)
    timing_rows = labeling_timing_rows(meta)
    timing_durations = np.asarray([r["duration_s"] for r in timing_rows], dtype=float)
    bbox_sep = read_bbox_separation(
        ctx.video_dir / "masks.sqlite",
        [int(r["frame_idx"]) for r in timing_rows],
        float(meta.get("width") or 0.0),
        float(meta.get("height") or 0.0),
    )
    edge_gaps = np.asarray(
        [
            bbox_sep.get(int(r["frame_idx"]), {}).get("min_bbox_edge_gap_norm", math.nan)
            for r in timing_rows
        ],
        dtype=float,
    )
    center_distances = np.asarray(
        [
            bbox_sep.get(int(r["frame_idx"]), {}).get("min_bbox_center_distance_norm", math.nan)
            for r in timing_rows
        ],
        dtype=float,
    )
    saved_frames = int(len(frames))
    min_saved = int(frames["frame_idx"].min()) if saved_frames else None
    max_saved = int(frames["frame_idx"].max()) if saved_frames else None
    tracked_span_frames = (max_saved - min_saved + 1) if min_saved is not None and max_saved is not None else 0
    elapsed_s = 0.0
    if saved_frames and frames["updated_at"].nunique() > 1:
        elapsed_s = float((frames["updated_at"].max() - frames["updated_at"].min()).total_seconds())
    throughput = saved_frames / elapsed_s if elapsed_s > 0 else math.nan
    throughput_values = np.asarray([], dtype=float)
    if saved_frames:
        by_time = (
            frames.groupby("updated_at")
            .agg(frames_saved=("frame_idx", "count"))
            .reset_index()
            .sort_values("updated_at")
        )
        by_time["elapsed_s"] = (by_time["updated_at"] - by_time["updated_at"].iloc[0]).dt.total_seconds()
        dt = by_time["elapsed_s"].diff()
        inst = (by_time["frames_saved"] / dt.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        throughput_values = inst.to_numpy(dtype=float)
    if throughput_values.size:
        throughput_q25, throughput_median, throughput_q75 = np.percentile(throughput_values, [25, 50, 75])
        throughput_iqr = throughput_q75 - throughput_q25
        throughput_low = throughput_q25 - 1.5 * throughput_iqr
        throughput_high = throughput_q75 + 1.5 * throughput_iqr
        throughput_outliers = int(np.sum((throughput_values < throughput_low) | (throughput_values > throughput_high)))
    else:
        throughput_q25 = throughput_median = throughput_q75 = math.nan
        throughput_outliers = 0
    video_duration_s = num_frames / fps if fps > 0 and num_frames else math.nan
    tracked_duration_s = tracked_span_frames / fps if fps > 0 and tracked_span_frames else math.nan
    return {
        "video_id": ctx.video_id,
        "video_name": meta.get("name") or ctx.video_dir.name,
        "num_frames": num_frames,
        "fps": fps,
        "video_duration_s": video_duration_s,
        "saved_tracking_frames": saved_frames,
        "saved_frame_min": min_saved,
        "saved_frame_max": max_saved,
        "tracked_span_frames": tracked_span_frames,
        "tracking_coverage_of_video": saved_frames / num_frames if num_frames else math.nan,
        "tracking_coverage_of_saved_span": saved_frames / tracked_span_frames if tracked_span_frames else math.nan,
        "progress_file_frame_count": len(set(progress_frames)),
        "anchor_frame_count": len(anchors),
        "prompt_labeled_frame_count": len(prompt_frames),
        "total_prompt_count": int(sum(prompt_counts.values())),
        "median_anchor_interval_frames": infer_anchor_interval(anchors),
        "frames_per_anchor_label": num_frames / len(anchors) if anchors else math.nan,
        "saved_tracking_frames_per_anchor_label": saved_frames / len(anchors) if anchors else math.nan,
        "anchor_label_fraction_of_video_frames": len(anchors) / num_frames if num_frames else math.nan,
        "labeling_timed_anchor_count": int(timing_durations.size),
        "labeling_sum_active_s": float(np.sum(timing_durations)) if timing_durations.size else math.nan,
        "labeling_duration_q25_s": float(np.percentile(timing_durations, 25)) if timing_durations.size else math.nan,
        "labeling_duration_median_s": float(np.percentile(timing_durations, 50)) if timing_durations.size else math.nan,
        "labeling_duration_q75_s": float(np.percentile(timing_durations, 75)) if timing_durations.size else math.nan,
        "labeling_duration_mean_s": float(np.mean(timing_durations)) if timing_durations.size else math.nan,
        "labeling_duration_max_s": float(np.max(timing_durations)) if timing_durations.size else math.nan,
        "labeling_mask_distance_count": int(np.sum(np.isfinite(edge_gaps))),
        "labeling_min_bbox_edge_gap_q25_norm": float(np.nanpercentile(edge_gaps, 25)) if np.any(np.isfinite(edge_gaps)) else math.nan,
        "labeling_min_bbox_edge_gap_median_norm": float(np.nanpercentile(edge_gaps, 50)) if np.any(np.isfinite(edge_gaps)) else math.nan,
        "labeling_min_bbox_edge_gap_q75_norm": float(np.nanpercentile(edge_gaps, 75)) if np.any(np.isfinite(edge_gaps)) else math.nan,
        "labeling_min_bbox_center_distance_median_norm": float(np.nanpercentile(center_distances, 50)) if np.any(np.isfinite(center_distances)) else math.nan,
        "labeling_duration_vs_edge_gap_spearman": float(pd.Series(timing_durations).corr(pd.Series(edge_gaps), method="spearman")) if timing_durations.size and np.sum(np.isfinite(edge_gaps)) >= 2 else math.nan,
        "labeling_duration_vs_center_distance_spearman": float(pd.Series(timing_durations).corr(pd.Series(center_distances), method="spearman")) if timing_durations.size and np.sum(np.isfinite(center_distances)) >= 2 else math.nan,
        **_duration_group_stats(timing_durations, edge_gaps),
        "sqlite_first_update": frames["updated_at"].min().isoformat() if saved_frames else "",
        "sqlite_last_update": frames["updated_at"].max().isoformat() if saved_frames else "",
        "sqlite_elapsed_s": elapsed_s,
        "mean_saved_frame_throughput_fps": throughput,
        "save_burst_throughput_q25_fps": float(throughput_q25),
        "save_burst_throughput_median_fps": float(throughput_median),
        "save_burst_throughput_q75_fps": float(throughput_q75),
        "save_burst_throughput_iqr_outlier_count": throughput_outliers,
        "realtime_factor_vs_video_fps": throughput / fps if fps > 0 and not math.isnan(throughput) else math.nan,
        "processing_seconds_per_video_second": (elapsed_s / tracked_duration_s) if tracked_duration_s and elapsed_s else math.nan,
    }


def write_summary_csv(summary: dict[str, Any], output_dir: Path) -> None:
    rows = []
    units = {
        "fps": "frames/s",
        "video_duration_s": "s",
        "sqlite_elapsed_s": "s",
        "mean_saved_frame_throughput_fps": "frames/s",
        "save_burst_throughput_q25_fps": "frames/s",
        "save_burst_throughput_median_fps": "frames/s",
        "save_burst_throughput_q75_fps": "frames/s",
        "labeling_sum_active_s": "s",
        "labeling_duration_q25_s": "s",
        "labeling_duration_median_s": "s",
        "labeling_duration_q75_s": "s",
        "labeling_duration_mean_s": "s",
        "labeling_duration_max_s": "s",
        "labeling_duration_median_overlap_bbox_s": "s",
        "labeling_duration_median_nonzero_bbox_gap_s": "s",
        "labeling_duration_median_close_bbox_gap_lt_0p02_s": "s",
        "labeling_duration_median_separated_bbox_gap_gt_0p02_s": "s",
        "labeling_duration_mean_overlap_bbox_s": "s",
        "labeling_duration_std_overlap_bbox_s": "s",
        "labeling_duration_mean_nonzero_bbox_gap_s": "s",
        "labeling_duration_std_nonzero_bbox_gap_s": "s",
        "labeling_duration_mean_close_bbox_gap_lt_0p02_s": "s",
        "labeling_duration_std_close_bbox_gap_lt_0p02_s": "s",
        "labeling_duration_mean_separated_bbox_gap_gt_0p02_s": "s",
        "labeling_duration_std_separated_bbox_gap_gt_0p02_s": "s",
        "labeling_min_bbox_edge_gap_q25_norm": "frame diagonal fraction",
        "labeling_min_bbox_edge_gap_median_norm": "frame diagonal fraction",
        "labeling_min_bbox_edge_gap_q75_norm": "frame diagonal fraction",
        "labeling_min_bbox_center_distance_median_norm": "frame diagonal fraction",
        "processing_seconds_per_video_second": "s/s",
        "median_anchor_interval_frames": "frames",
    }
    for key, value in summary.items():
        rows.append({"metric": key, "value": value, "unit": units.get(key, "")})
    save_csv(output_dir / "summary_metrics.csv", rows, ["metric", "value", "unit"])


def write_anchor_csv(ctx: VideoContext, output_dir: Path) -> pd.DataFrame:
    anchors = annotated_anchor_frames(ctx.video_meta)
    prompt_counts = prompt_counts_by_frame(ctx.video_meta)
    fps = float(ctx.video_meta.get("fps") or 0.0)
    rows = []
    prev = None
    for i, frame_idx in enumerate(anchors, start=1):
        rows.append(
            {
                "anchor_number": i,
                "frame_idx": frame_idx,
                "time_s": frame_idx / fps if fps > 0 else math.nan,
                "prompt_count": int(prompt_counts.get(frame_idx, 0)),
                "gap_from_previous_anchor_frames": frame_idx - prev if prev is not None else "",
            }
        )
        prev = frame_idx
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "anchor_frames.csv", index=False)
    return df


def write_labeling_timing_csv(ctx: VideoContext, output_dir: Path) -> pd.DataFrame:
    rows = labeling_timing_rows(ctx.video_meta)
    df = pd.DataFrame(rows)
    if not df.empty:
        sep = read_bbox_separation(
            ctx.video_dir / "masks.sqlite",
            df["frame_idx"].astype(int).tolist(),
            float(ctx.video_meta.get("width") or 0.0),
            float(ctx.video_meta.get("height") or 0.0),
        )
        for col in [
            "bbox_pair_count",
            "nearest_bbox_pair",
            "min_bbox_center_distance_px",
            "min_bbox_center_distance_norm",
            "min_bbox_edge_gap_px",
            "min_bbox_edge_gap_norm",
        ]:
            df[col] = [sep.get(int(f), {}).get(col, math.nan if col != "nearest_bbox_pair" else "") for f in df["frame_idx"]]
    df.to_csv(output_dir / "labeling_timing.csv", index=False)
    return df


def write_throughput_csv(frames: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if frames.empty:
        df = pd.DataFrame()
        df.to_csv(output_dir / "throughput_timeseries.csv", index=False)
        return df
    grouped = (
        frames.groupby("updated_at")
        .agg(
            frames_saved=("frame_idx", "count"),
            min_frame_idx=("frame_idx", "min"),
            max_frame_idx=("frame_idx", "max"),
            mean_seg_blob_bytes=("seg_blob_bytes", "mean"),
            mean_object_count=("object_count", "mean"),
        )
        .reset_index()
        .sort_values("updated_at")
    )
    start = grouped["updated_at"].iloc[0]
    grouped["elapsed_s"] = (grouped["updated_at"] - start).dt.total_seconds()
    grouped["cumulative_frames_saved"] = grouped["frames_saved"].cumsum()
    dt = grouped["elapsed_s"].diff()
    grouped["instantaneous_saved_fps"] = grouped["frames_saved"] / dt.replace(0, np.nan)
    grouped["cumulative_saved_fps"] = grouped["cumulative_frames_saved"] / grouped["elapsed_s"].replace(0, np.nan)
    valid = grouped["instantaneous_saved_fps"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        grouped["throughput_q25_fps"] = math.nan
        grouped["throughput_median_fps"] = math.nan
        grouped["throughput_q75_fps"] = math.nan
    else:
        grouped["throughput_q25_fps"] = float(np.percentile(valid, 25))
        grouped["throughput_median_fps"] = float(np.percentile(valid, 50))
        grouped["throughput_q75_fps"] = float(np.percentile(valid, 75))
    grouped.to_csv(output_dir / "throughput_timeseries.csv", index=False)
    return grouped


def write_throughput_distribution_csv(throughput: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if throughput.empty:
        df = pd.DataFrame()
        df.to_csv(output_dir / "throughput_distribution.csv", index=False)
        return df
    values = throughput["instantaneous_saved_fps"].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        df = pd.DataFrame()
        df.to_csv(output_dir / "throughput_distribution.csv", index=False)
        return df
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df = throughput.loc[values.index, [
        "updated_at",
        "elapsed_s",
        "frames_saved",
        "min_frame_idx",
        "max_frame_idx",
        "instantaneous_saved_fps",
    ]].copy()
    df["throughput_q25_fps"] = float(q1)
    df["throughput_median_fps"] = float(med)
    df["throughput_q75_fps"] = float(q3)
    df["iqr_outlier"] = (df["instantaneous_saved_fps"] < low) | (df["instantaneous_saved_fps"] > high)
    df.to_csv(output_dir / "throughput_distribution.csv", index=False)
    return df


def write_coverage_csv(ctx: VideoContext, frames: pd.DataFrame, output_dir: Path, bin_size: int = 1000) -> pd.DataFrame:
    num_frames = int(ctx.video_meta.get("num_frames") or (int(frames["frame_idx"].max()) + 1 if not frames.empty else 0))
    anchors = set(annotated_anchor_frames(ctx.video_meta))
    if num_frames <= 0:
        df = pd.DataFrame()
        df.to_csv(output_dir / "frame_coverage_by_interval.csv", index=False)
        return df
    saved = set(frames["frame_idx"].astype(int).tolist()) if not frames.empty else set()
    rows = []
    for start in range(0, num_frames, bin_size):
        end = min(start + bin_size - 1, num_frames - 1)
        denom = end - start + 1
        saved_count = sum(1 for f in saved if start <= f <= end)
        anchor_count = sum(1 for f in anchors if start <= f <= end)
        rows.append(
            {
                "interval_start_frame": start,
                "interval_end_frame": end,
                "interval_frame_count": denom,
                "saved_tracking_frames": saved_count,
                "coverage_fraction": saved_count / denom if denom else math.nan,
                "anchor_frame_count": anchor_count,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "frame_coverage_by_interval.csv", index=False)
    return df


def plot_labeling_efficiency(summary: dict[str, Any], labeling_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), gridspec_kw={"height_ratios": [1.0, 1.05]})
    axes = axes.ravel()

    if not labeling_df.empty:
        durations = labeling_df["duration_s"].to_numpy(dtype=float)
        q25, med, q75 = np.percentile(durations, [25, 50, 75])

        ax = axes[0]
        ax.boxplot(durations, vert=True, widths=0.35, showfliers=True, patch_artist=True,
                   boxprops={"facecolor": "#D7E6F5", "color": "#4C78A8"},
                   medianprops={"color": "#E15759", "linewidth": 2})
        jitter = np.linspace(-0.08, 0.08, len(durations)) if len(durations) > 1 else np.array([0.0])
        ax.scatter(np.ones_like(durations) + jitter, durations, s=22, color="#4C78A8", alpha=0.7, zorder=3)
        ax.set_xticks([1])
        ax.set_xticklabels(["Anchor frames"])
        ax.set_ylabel("Labeling duration (s)")
        ax.set_title(f"Per-anchor duration: median {med:.1f}s (IQR {q25:.1f}-{q75:.1f}s)")
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[1]
        if "min_bbox_edge_gap_norm" in labeling_df:
            _plot_distance_group_comparison(ax, labeling_df)
        else:
            ax.text(0.5, 0.5, "No nearest bbox gap data", ha="center", va="center")
            ax.set_axis_off()

        ax = axes[2]
        ax.plot(labeling_df["anchor_number"], labeling_df["duration_s"], color="#4C78A8", linewidth=1.25)
        ax.scatter(labeling_df["anchor_number"], labeling_df["duration_s"], s=18, color="#4C78A8")
        ax.axhline(med, color="#E15759", linestyle="--", linewidth=1.2, label="median")
        ax.fill_between(
            [labeling_df["anchor_number"].min(), labeling_df["anchor_number"].max()],
            q25,
            q75,
            color="#E15759",
            alpha=0.12,
            label="IQR",
        )
        ax.set_xlabel("Anchor number")
        ax.set_ylabel("Labeling duration (s)")
        ax.set_title("Duration timecourse")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        ax = axes[3]
        bins = min(18, max(6, int(np.sqrt(len(durations)))))
        ax.hist(durations, bins=bins, color="#F28E2B", alpha=0.78, edgecolor="white")
        ax.axvline(q25, color="#555555", linestyle=":", linewidth=1.2, label="25th/75th")
        ax.axvline(med, color="#E15759", linestyle="-", linewidth=1.8, label="median")
        ax.axvline(q75, color="#555555", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Labeling duration (s)")
        ax.set_ylabel("Anchor count")
        ax.set_title("Duration distribution")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No anchor_labeling_timing data found", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle(
        f"Human anchor labeling: {summary['anchor_frame_count']:,} anchors for {summary['num_frames']:,} video frames",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "labeling_efficiency")


def _plot_distance_group_comparison(ax: plt.Axes, labeling_df: pd.DataFrame) -> None:
    gaps = pd.to_numeric(labeling_df["min_bbox_edge_gap_norm"], errors="coerce")
    durations = pd.to_numeric(labeling_df["duration_s"], errors="coerce")
    valid = gaps.notna() & durations.notna()
    gaps = gaps[valid]
    durations = durations[valid]

    groups = [
        ("0", durations[gaps <= ZERO_BBOX_GAP_EPS].to_numpy(dtype=float), "#4C78A8"),
        (">0", durations[gaps > ZERO_BBOX_GAP_EPS].to_numpy(dtype=float), "#4C78A8"),
        ("<0.02", durations[gaps < NEAR_BBOX_GAP_THRESHOLD_NORM].to_numpy(dtype=float), "#59A14F"),
        (">0.02", durations[gaps > NEAR_BBOX_GAP_THRESHOLD_NORM].to_numpy(dtype=float), "#59A14F"),
    ]
    data = [vals for _, vals, _ in groups if vals.size]
    positions = [i + 1 for i, (_, vals, _) in enumerate(groups) if vals.size]
    labels = [f"{label}\n(n={len(vals)})" for label, vals, _ in groups if vals.size]
    colors = [color for _, vals, color in groups if vals.size]
    if not data:
        ax.text(0.5, 0.5, "No nearest bbox gap data", ha="center", va="center")
        ax.set_axis_off()
        return

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": "#E15759", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    for pos, vals, color in zip(positions, data, colors):
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=16, color=color, alpha=0.55, zorder=3)
    ax.axvline(2.5, color="#999999", linewidth=1, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nearest bbox edge gap / frame diagonal")
    ax.set_ylabel("Labeling duration (s)")
    ax.set_title("Duration by bbox overlap and close-separation bins")
    ax.grid(True, axis="y", alpha=0.25)
    y_base = max(np.nanmax(vals) for vals in data if vals.size)
    y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1.0)
    if len(groups[0][1]) and len(groups[1][1]):
        _add_sig_bracket(ax, 1, 2, y_base + y_span * 0.05, _compare_samples(groups[0][1], groups[1][1])["mannwhitney_p"])
    if len(groups[2][1]) and len(groups[3][1]):
        _add_sig_bracket(ax, 3, 4, y_base + y_span * 0.16, _compare_samples(groups[2][1], groups[3][1])["mannwhitney_p"])


def plot_throughput(throughput: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    if not throughput.empty:
        axes[0].plot(
            throughput["updated_at"],
            throughput["cumulative_frames_saved"],
            color="#4C78A8",
            linewidth=2,
        )
        axes[1].bar(
            throughput["updated_at"],
            throughput["frames_saved"],
            width=1 / (24 * 60 * 1.5),
            color="#59A14F",
            alpha=0.75,
        )
    axes[0].set_ylabel("Cumulative saved frames")
    axes[0].set_title(
        f"Mean saved-frame throughput: {summary['mean_saved_frame_throughput_fps']:.2f} frames/s"
    )
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel("Frames saved per timestamp")
    axes[1].set_xlabel("SQLite update time")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, output_dir, "frame_throughput")


def plot_throughput_distribution(throughput_dist: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    if not throughput_dist.empty:
        values = throughput_dist["instantaneous_saved_fps"].to_numpy(dtype=float)
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        outliers = throughput_dist[throughput_dist["iqr_outlier"]]

        ax = axes[0]
        parts = ax.violinplot(values, positions=[1], showmeans=False, showmedians=False, widths=0.65)
        for body in parts["bodies"]:
            body.set_facecolor("#76B7B2")
            body.set_edgecolor("#3B7A78")
            body.set_alpha(0.65)
        ax.boxplot(values, positions=[1], widths=0.18, showfliers=True,
                   medianprops={"color": "#E15759", "linewidth": 2})
        ax.scatter(np.ones_like(values) + np.linspace(-0.08, 0.08, len(values)), values,
                   s=12, color="#4C78A8", alpha=0.35)
        if not outliers.empty:
            ax.scatter(np.ones(len(outliers)) + 0.18, outliers["instantaneous_saved_fps"],
                       s=28, facecolors="none", edgecolors="#E15759", linewidths=1.1, label="IQR outlier")
            ax.legend(frameon=False)
        ax.set_xticks([1])
        ax.set_xticklabels(["Save bursts"])
        ax.set_ylabel("Throughput (saved frames/s)")
        ax.set_title(f"Throughput quartiles: {q25:.1f}, {med:.1f}, {q75:.1f} fps")
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[1]
        bins = min(40, max(8, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color="#59A14F", alpha=0.8, edgecolor="white")
        ax.axvline(q25, color="#555555", linestyle=":", linewidth=1.2, label="25th/75th")
        ax.axvline(med, color="#E15759", linewidth=1.8, label="median")
        ax.axvline(q75, color="#555555", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Throughput (saved frames/s)")
        ax.set_ylabel("Save-burst count")
        ax.set_title("Histogram with quartiles")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No nonzero timestamp intervals found", ha="center", va="center")
            ax.set_axis_off()
    fig.tight_layout()
    save_fig(fig, output_dir, "throughput_distribution")


def plot_coverage(coverage: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    if not coverage.empty:
        mids = (coverage["interval_start_frame"] + coverage["interval_end_frame"]) / 2
        ax1.bar(
            mids,
            coverage["coverage_fraction"],
            width=coverage["interval_frame_count"],
            color="#4C78A8",
            alpha=0.75,
            align="center",
            label="Tracking coverage",
        )
        ax2 = ax1.twinx()
        ax2.step(
            mids,
            coverage["anchor_frame_count"],
            where="mid",
            color="#E15759",
            linewidth=1.5,
            label="Anchor labels",
        )
        ax2.set_ylabel("Anchor labels per interval")
        ax2.set_ylim(bottom=0)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Coverage fraction")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Tracking coverage by 1,000-frame interval")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, output_dir, "tracking_coverage")


def write_readme(ctx: VideoContext, summary: dict[str, Any]) -> None:
    text = f"""# System Tracking Analysis

Video: {summary['video_name']}
Video id: {summary['video_id']}

Definitions:
- Anchor labels are frames in `annotated_anchors` plus any frame with point prompts in `config.json`.
- Labeling timing uses `anchor_labeling_timing` in `config.json`, recorded by the frontend from landing on an anchor frame to committing it.
- Mask separation uses the nearest pair of persisted object bounding boxes on each timed anchor frame; edge gap and center distance are normalized by frame diagonal.
- Saved tracking frames are rows in `masks.sqlite/frame_segmentation`.
- Throughput uses SQLite `updated_at` timestamps, so it measures persisted saved-frame rate, not GPU kernel timing. Distribution plots use nonzero save-burst intervals between distinct SQLite timestamps.
- Coverage is saved tracking frames divided by video frames, with an additional span-normalized coverage metric.

Key metrics:
- Anchor labels: {summary['anchor_frame_count']:,}
- Median labeling duration: {summary['labeling_duration_median_s']:.3f} s/anchor
- Median nearest bbox edge gap: {summary['labeling_min_bbox_edge_gap_median_norm']:.5f} frame diagonals
- Duration vs nearest bbox edge gap Spearman: {summary['labeling_duration_vs_edge_gap_spearman']:.3f}
- Saved tracking frames: {summary['saved_tracking_frames']:,}
- Saved tracking frames per anchor label: {summary['saved_tracking_frames_per_anchor_label']:.2f}
- Mean saved-frame throughput: {summary['mean_saved_frame_throughput_fps']:.3f} frames/s
- Real-time factor vs video FPS: {summary['realtime_factor_vs_video_fps']:.3f}
"""
    (ctx.output_dir / "README.md").write_text(text)


def run(video_ref: str, output_dir: str | None = None) -> Path:
    ctx = resolve_context(video_ref, output_dir)
    frames = read_sqlite_frames(ctx.video_dir / "masks.sqlite")
    progress_frames = read_progress_frames(ctx.video_dir / "propagation_progress.txt")

    summary = build_summary(ctx, frames, progress_frames)
    write_summary_csv(summary, ctx.output_dir)
    anchor_df = write_anchor_csv(ctx, ctx.output_dir)
    labeling_df = write_labeling_timing_csv(ctx, ctx.output_dir)
    throughput_df = write_throughput_csv(frames, ctx.output_dir)
    throughput_dist_df = write_throughput_distribution_csv(throughput_df, ctx.output_dir)
    coverage_df = write_coverage_csv(ctx, frames, ctx.output_dir)

    plot_labeling_efficiency(summary, labeling_df, ctx.output_dir)
    plot_throughput(throughput_df, summary, ctx.output_dir)
    plot_throughput_distribution(throughput_dist_df, ctx.output_dir)
    plot_coverage(coverage_df, ctx.output_dir)
    write_readme(ctx, summary)
    return ctx.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", help="Tracked video directory under <project>/videos/")
    parser.add_argument("--output-dir", help="Output directory. Default: <video_dir>/system_analysis/")
    args = parser.parse_args()
    out = run(args.video_dir, args.output_dir)
    print(out)


if __name__ == "__main__":
    main()
