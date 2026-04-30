"""Load project config and mask timelines for completed tracks."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Reuse mask loader from backend
_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from mask_store import VideoMaskStorage

from .tqdm_optional import try_tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoTrackContext:
    video_id: str
    video_name: str
    config: dict[str, Any]
    video_dir: Path


def load_project_config(project_dir: Path) -> dict[str, Any]:
    cfg = project_dir / "config.json"
    return json.loads(cfg.read_text())


def video_storage_dir(project_dir: Path, vid: str) -> Path | None:
    root = project_dir / "videos"
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.name == vid or d.name.startswith(vid + "_"):
            return d
    return None


def _masks_cover_all_frames(
    video: dict[str, Any],
    video_dir: Path,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> bool:
    start = int(video.get("start_frame") or 0)
    n = int(video["num_frames"])
    total = max(0, n - start)
    rng = range(start, n)
    if progress and progress_desc and total > 0:
        rng = try_tqdm(
            rng,
            total=total,
            desc=progress_desc,
            leave=False,
            unit="frm",
        )
    ms = VideoMaskStorage(video_dir)
    for fi in rng:
        if not ms.has_masks(fi):
            return False
    return True


def _try_make_complete_context(
    project_dir: Path,
    vid: str,
    v: dict[str, Any],
    *,
    verify_progress: bool = False,
    verify_desc: str | None = None,
) -> tuple[VideoTrackContext | None, str | None]:
    """
    If this video is fully tracked, return (context, None).
    Otherwise return (None, short_reason).
    """
    vdir = video_storage_dir(project_dir, vid)
    if vdir is None:
        return None, "missing video directory"
    if not v.get("propagation_complete"):
        return None, "propagation_complete is false"
    ok = _masks_cover_all_frames(
        v,
        vdir,
        progress=verify_progress,
        progress_desc=verify_desc,
    )
    if not ok:
        return None, "missing saved masks for some frames"
    return (
        VideoTrackContext(
            video_id=str(vid),
            video_name=str(v.get("name") or vid),
            config=dict(v),
            video_dir=vdir,
        ),
        None,
    )


def is_fully_tracked(video: dict[str, Any], video_dir: Path) -> bool:
    """Require propagation flag and persisted masks for every frame in range."""
    if not video.get("propagation_complete"):
        return False
    return _masks_cover_all_frames(video, video_dir, progress=False)


def partition_complete_videos(
    project_dir: Path,
    config: dict[str, Any],
    *,
    progress: bool = True,
) -> tuple[list[VideoTrackContext], list[tuple[str, str, str]]]:
    """
    Returns (complete contexts, skipped list of (video_id, video_name, reason)).
    Logs one line per skipped video at INFO.
    """
    skipped: list[tuple[str, str, str]] = []
    complete: list[VideoTrackContext] = []
    items = list(config.get("videos", {}).items())
    logger.info(
        "Scanning %d video(s) in project for fully tracked status (propagation_complete + mask per frame).",
        len(items),
    )
    vid_iter = try_tqdm(
        items,
        desc="Videos (eligibility)",
        unit="video",
        leave=True,
        disable=not progress,
    )
    for vid, v in vid_iter:
        vname = str(v.get("name") or vid)
        desc = f'Masks [{vname[:28]}]'
        ctx, reason = _try_make_complete_context(
            project_dir,
            vid,
            v,
            verify_progress=progress,
            verify_desc=desc if progress else None,
        )
        if ctx is None:
            skipped.append((vid, vname, reason or "unknown"))
            logger.info('SKIP video "%s" (%s): %s', vname, vid, reason)
            continue
        complete.append(ctx)
        logger.debug('INCLUDE video "%s" (%s): %d frames', vname, vid, int(v["num_frames"]))

    logger.info(
        "Eligibility done: %d video(s) included, %d skipped.",
        len(complete),
        len(skipped),
    )
    return complete, skipped


def iter_complete_videos(project_dir: Path, config: dict[str, Any]) -> Iterator[VideoTrackContext]:
    """Yield fully tracked videos (no tqdm bars, no eligibility logging)."""
    for vid, v in config.get("videos", {}).items():
        ctx, _reason = _try_make_complete_context(project_dir, vid, v, verify_progress=False)
        if ctx is not None:
            yield ctx


def npz_key_base_id(npz_key: str) -> str:
    """Map mask key '1_2' -> logical object id '1'."""
    return npz_key.split("_")[0] if "_" in npz_key else npz_key


def object_display_name(video_config: dict[str, Any], npz_key: str) -> str | None:
    base = npz_key_base_id(npz_key)
    obj = (video_config.get("objects") or {}).get(base)
    if obj is None:
        return None
    return str(obj.get("name") or base)


def mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    """Centroid in pixel coordinates (x, y). None if empty."""
    m = np.squeeze(mask).astype(bool)
    if m.size == 0 or not np.any(m):
        return None
    ys, xs = np.where(m)
    return float(xs.mean()), float(ys.mean())


def build_centroid_timelines(
    ctx: VideoTrackContext,
    *,
    progress: bool = True,
) -> tuple[dict[str, np.ndarray], int, int, int, int]:
    """
    For each mask npz key, array shape (n_frames - start, 2): columns x,y in pixels;
    NaN where missing / empty mask.

    Returns:
      series: mapping npz_key -> float array shape (T, 2)
      start_frame, num_frames, width, height
    """
    start = int(ctx.config.get("start_frame") or 0)
    n = int(ctx.config["num_frames"])
    w = int(ctx.config.get("width") or 1)
    h = int(ctx.config.get("height") or 1)
    total_fr = max(0, n - start)

    logger.info(
        'Loading centroid timelines for "%s" (%s): frames %d..%d (%d frames), reading persisted masks',
        ctx.video_name,
        ctx.video_id,
        start,
        n - 1,
        total_fr,
    )

    all_series: dict[str, np.ndarray] = {}

    frame_iter = range(start, n)
    if progress and total_fr > 0:
        frame_iter = try_tqdm(
            frame_iter,
            total=total_fr,
            desc=f'Centroids [{ctx.video_name[:22]}]',
            leave=False,
            unit="frm",
        )

    ms = VideoMaskStorage(ctx.video_dir)

    for fi in frame_iter:
        masks = ms.load_masks_dense(fi)
        row = fi - start
        for k, m in masks.items():
            if k not in all_series:
                all_series[k] = np.full((n - start, 2), np.nan, dtype=np.float64)
            c = mask_centroid_xy(m)
            if c is None:
                continue
            all_series[k][row, 0] = c[0]
            all_series[k][row, 1] = c[1]

    n_keys = len(all_series)
    logger.info(
        'Finished centroids for "%s": %d mask key(s) (object instances).',
        ctx.video_name,
        n_keys,
    )
    return all_series, start, n, w, h
