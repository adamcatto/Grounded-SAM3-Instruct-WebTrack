"""Figures for the anchor-frame memory ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import RESULTS_DIRNAME


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    if x in (None, "", "None"):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"[figures] {out_dir / name}.{{png,pdf,svg}}")


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 500, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def make_figures(out_dir: Path) -> None:
    out_dir = Path(out_dir).resolve()
    fig_dir = out_dir / "figures"
    _style()
    results = json.loads((out_dir / "results.json").read_text())
    with np.load(out_dir / "timelines.npz") as z:
        tl = {k: z[k] for k in z.files}
    video_rows = _read_csv(out_dir / "video_summary.csv")
    object_rows = _read_csv(out_dir / "object_summary.csv")
    reentry_rows = _read_csv(out_dir / "reentry_events.csv")
    per_frame = _read_csv(out_dir / "per_frame_iou.csv")

    _plot_timecourse(tl, video_rows, reentry_rows, results, fig_dir)
    _plot_object_facets(tl, object_rows, reentry_rows, results, fig_dir)
    _plot_summary_bars(object_rows, video_rows, fig_dir)
    _plot_missing_timeline(per_frame, fig_dir)
    _plot_reentry_windows(reentry_rows, fig_dir)


def _video_ids(tl: dict[str, np.ndarray]) -> list[str]:
    if "__video_ids__" in tl:
        return [str(x) for x in tl["__video_ids__"]]
    return sorted(k[: -len("__frames")] for k in tl if k.endswith("__frames"))


def _plot_timecourse(tl, video_rows, reentry_rows, results, fig_dir):
    meta = {r["video_id"]: r for r in video_rows}
    events = {}
    for r in reentry_rows:
        events.setdefault(r["video_id"], []).append(int(float(r["frame"])))
    tau = float(results.get("threshold", 0.5))
    vids = _video_ids(tl)
    if not vids:
        return
    fig, axes = plt.subplots(len(vids), 1, figsize=(12, max(3.2, 2.8 * len(vids))), squeeze=False, sharex=False)
    for ax, vid in zip(axes[:, 0], vids):
        frames = tl[f"{vid}__frames"]
        ax.plot(frames, tl[f"{vid}__mean"], color="#1F77B4", label="mean IoU")
        ax.plot(frames, tl[f"{vid}__min"], color="#D62728", lw=1.0, alpha=0.85, label="worst-object IoU")
        ax.axhline(tau, color="0.25", ls=":", lw=1.0, label=f"IoU {tau:g}")
        for frame in events.get(vid, []):
            ax.axvline(frame, color="#2CA02C", alpha=0.35, lw=0.9)
        title = meta.get(vid, {}).get("video_name", vid)
        ax.set_title(f"{title}: queue memory vs anchor-frame memory")
        ax.set_ylabel("IoU")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="lower left", ncol=4)
    axes[-1, 0].set_xlabel("frame")
    fig.suptitle("Tracking quality over time; green ticks mark anchor-GT object re-entry after absence", y=1.0)
    fig.tight_layout()
    _save(fig, fig_dir, "timecourse_mean_min_iou")


def _plot_object_facets(tl, object_rows, reentry_rows, results, fig_dir):
    event_frames = {(r["video_id"], r["object_id"]): [] for r in reentry_rows}
    for r in reentry_rows:
        event_frames.setdefault((r["video_id"], r["object_id"]), []).append(int(float(r["frame"])))
    panels = []
    for vid in _video_ids(tl):
        frames = tl[f"{vid}__frames"]
        oids = [str(x) for x in tl.get(f"{vid}__objids", [])]
        names = [str(x) for x in tl.get(f"{vid}__objnames", [])]
        for oid, name in zip(oids, names):
            key = f"{vid}__obj__{oid}"
            if key in tl:
                panels.append((vid, oid, name, frames, tl[key], tl.get(f"{vid}__present__{oid}")))
    if not panels:
        return
    ncols = min(2, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 2.8 * nrows), squeeze=False, sharey=True)
    tau = float(results.get("threshold", 0.5))
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (vid, oid, name, frames, vals, present) in zip(axes.flat, panels):
        ax.set_visible(True)
        ax.plot(frames, vals, color="#9467BD", lw=1.0)
        if present is not None:
            y0 = np.full(frames.shape, -0.015, dtype=float)
            ax.fill_between(frames, y0, 0.02, where=present.astype(bool), color="#2CA02C", alpha=0.25, linewidth=0)
        for frame in event_frames.get((vid, oid), []):
            ax.axvline(frame, color="#2CA02C", alpha=0.45, lw=0.9)
        ax.axhline(tau, color="0.25", ls=":", lw=0.9)
        ax.set_title(f"{name} ({oid})")
        ax.set_ylim(-0.04, 1.03)
        ax.set_xlabel("frame")
        ax.set_ylabel("IoU")
    fig.suptitle("Per-object IoU; green baseline indicates anchor-memory object present", y=1.0)
    fig.tight_layout()
    _save(fig, fig_dir, "object_iou_facets")


def _plot_summary_bars(object_rows, video_rows, fig_dir):
    if not object_rows:
        return
    labels = [r["object_name"] for r in object_rows]
    mean_iou = [_f(r["mean_iou"]) for r in object_rows]
    missed = [100 * _f(r["frac_present_frames_missed"]) for r in object_rows]
    y = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3, 0.55 * len(labels) + 1.4)), sharey=True)
    ax1.barh(y, mean_iou, color="#1F77B4")
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("mean IoU")
    ax1.set_title("Queue-memory quality")
    ax2.barh(y, missed, color="#D62728")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% anchor-present frames missed")
    ax2.set_title("Lost tracking")
    fig.suptitle("Per-object summary: queue memory evaluated against anchor-frame memory")
    fig.tight_layout()
    _save(fig, fig_dir, "object_summary_bars")


def _plot_missing_timeline(per_frame, fig_dir):
    if not per_frame:
        return
    frames = np.asarray([int(float(r["frame"])) for r in per_frame])
    missing = np.asarray([int(float(r["n_missing_objects"])) for r in per_frame])
    mean_iou = np.asarray([_f(r["mean_iou"]) for r in per_frame])
    fig, ax1 = plt.subplots(figsize=(12, 3.5))
    ax1.fill_between(frames, 0, missing, color="#D62728", alpha=0.35, step="mid", label="missing objects")
    ax1.set_ylabel("missing objects")
    ax1.set_xlabel("frame")
    ax2 = ax1.twinx()
    ax2.plot(frames, mean_iou, color="#1F77B4", lw=1.0, label="mean IoU")
    ax2.set_ylabel("mean IoU")
    ax2.set_ylim(-0.03, 1.03)
    ax1.set_title("Object-frame loss over time")
    fig.tight_layout()
    _save(fig, fig_dir, "missing_objects_timeline")


def _plot_reentry_windows(reentry_rows, fig_dir):
    if not reentry_rows:
        return
    labels = [f"{r['object_name']}@{int(float(r['frame']))}" for r in reentry_rows]
    mean_iou = [_f(r["mean_iou_window"]) for r in reentry_rows]
    missed = [int(float(r["missed_frames_window"])) for r in reentry_rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(6, 0.7 * len(labels) + 2), 4))
    ax1.bar(x - 0.18, mean_iou, width=0.36, color="#1F77B4", label="mean IoU in window")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("mean IoU")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, missed, width=0.36, color="#D62728", alpha=0.75, label="missed frames")
    ax2.set_ylabel("missed frames")
    ax1.set_xticks(x, labels, rotation=45, ha="right")
    ax1.set_title("Tracking immediately after anchor-GT object re-entry")
    fig.tight_layout()
    _save(fig, fig_dir, "reentry_event_windows")


__all__ = ["make_figures"]
