"""Orchestrator: iterate project videos, load masks from SQLite, compute features.

Produces a BehaviorDataset with all sliding-window sequence feature vectors
and per-window metadata.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Backend mask loader (SQLite-first, NPZ fallback)
_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from mask_store import VideoMaskStorage

from ..tracking_io import (
    VideoTrackContext,
    load_project_config,
    object_display_name,
    partition_complete_videos,
    video_storage_dir,
)
from ..tqdm_optional import try_tqdm

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _stdout_tqdm(iterable, **kwargs):
    """tqdm that writes to stdout (visible under conda run which buffers stderr)."""
    if _HAS_TQDM:
        return _tqdm(iterable, file=sys.stdout, **kwargs)
    return iterable
from .config import ClusteringConfig
from .dataset import (
    BehaviorDataset,
    WindowMetadata,
    camera_view_from_folder,
    interaction_type_from_names,
    session_from_video_name,
)
from .frame_features import N_FRAME_FEATURES, extract_frame_features
from .sequence_features import SEQUENCE_FEATURE_NAMES, compute_sequence_features

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def _fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable size."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    else:
        return f"{nbytes / 1024**2:.1f} MB"


# ---------------------------------------------------------------------------
# Per-video frame feature extraction
# ---------------------------------------------------------------------------

def _discover_object_keys(
    ms: VideoMaskStorage,
    ctx: VideoTrackContext,
) -> tuple[str, str] | None:
    """Find the object keys for a video by sampling frames.

    Returns (obj_a_key, obj_b_key) sorted canonically. If only 1 object
    is present, returns (obj_a_key, "") to support single-animal videos.
    Returns None if no objects are found.
    """
    start = int(ctx.config.get("start_frame") or 0)
    n = int(ctx.config["num_frames"])

    seen_keys = set()
    # Sample a few frames first. If we find a frame with >= 2 keys, we can instantly return them.
    for fi in range(start, min(start + 50, n)):
        masks = ms.load_masks_dense(fi)
        keys = sorted(masks.keys())
        if len(keys) >= 2:
            return keys[0], keys[1]
        for k in keys:
            seen_keys.add(k)

    # If we didn't find >= 2 keys in the first 50 frames, let's sample more frames across
    # the entire video (up to 200 frames) to be absolutely sure there isn't a second animal.
    sample_indices = np.linspace(start, n - 1, min(200, n - start), dtype=int)
    for fi in sample_indices:
        masks = ms.load_masks_dense(int(fi))
        keys = sorted(masks.keys())
        if len(keys) >= 2:
            return keys[0], keys[1]
        for k in keys:
            seen_keys.add(k)

    # If we still only see 1 key, return (key, "") to support single animal
    if len(seen_keys) == 1:
        return list(seen_keys)[0], ""

    return None


def _extract_video_frame_features(
    ctx: VideoTrackContext,
    cfg: ClusteringConfig,
    *,
    cache_dir: Path | None = None,
    progress: bool = True,
    video_index: int = 0,
    total_videos: int = 0,
) -> tuple[np.ndarray, str, str, int, int, int, int] | None:
    """Extract frame-level features for one video.

    Returns (frame_features, obj_a_key, obj_b_key, start_frame, num_frames, width, height)
    or None if the video can't be processed.
    """
    tag = f"[{video_index}/{total_videos}]" if total_videos else ""

    start = int(ctx.config.get("start_frame") or 0)
    n = int(ctx.config["num_frames"])
    w = int(ctx.config.get("width") or 1)
    h = int(ctx.config.get("height") or 1)
    total_fr = max(0, n - start)
    fps = float(ctx.config.get("fps") or 30)
    duration_str = _fmt_duration(total_fr / fps) if fps > 0 else "?"

    ms = VideoMaskStorage(ctx.video_dir)

    # Check cache
    if cache_dir is not None:
        cache_path = cache_dir / f"frame_features_{ctx.video_id}.npz"
        if cache_path.is_file():
            sqlite_path = ctx.video_dir / "masks.sqlite"
            if sqlite_path.is_file() and cache_path.stat().st_mtime > sqlite_path.stat().st_mtime:
                cache_size = _fmt_size(cache_path.stat().st_size)
                logger.info(
                    '%s Loaded cached frame features for "%s" (%s, %s)',
                    tag, ctx.video_name, cache_size, _fmt_duration(0),
                )
                data = np.load(cache_path, allow_pickle=True)
                return (
                    data["frame_features"],
                    str(data["obj_a_key"]),
                    str(data["obj_b_key"]),
                    int(data["start_frame"]),
                    int(data["num_frames"]),
                    int(data["width"]),
                    int(data["height"]),
                )

    # Discover object keys
    keys = _discover_object_keys(ms, ctx)
    if keys is None:
        logger.warning('%s SKIP "%s": could not find any object keys', tag, ctx.video_name)
        return None
    obj_a_key, obj_b_key = keys

    # Look up display names for progress messages
    name_a = object_display_name(ctx.config, obj_a_key) or obj_a_key
    name_b = (object_display_name(ctx.config, obj_b_key) or obj_b_key) if obj_b_key else "none"

    logger.info(
        '%s Extracting frame features for "%s" | %d frames (%s) | %dx%d | objects: %s, %s',
        tag, ctx.video_name, total_fr, duration_str, w, h, name_a, name_b,
    )

    frame_features = np.full((total_fr, N_FRAME_FEATURES), np.nan, dtype=np.float64)

    t0 = time.monotonic()
    n_empty = 0

    frame_iter = range(start, n)
    if progress and total_fr > 0:
        frame_iter = _stdout_tqdm(
            frame_iter,
            total=total_fr,
            desc=f"{tag} Frames [{ctx.video_name[:18]}]",
            leave=False,
            unit="frm",
            miniters=max(1, total_fr // 200),  # update ~200 times
        )

    for fi in frame_iter:
        masks = ms.load_masks_dense(fi)
        row = fi - start
        fv = extract_frame_features(
            masks, obj_a_key, obj_b_key,
            contour_sample_n=cfg.contour_sample_n,
        )
        frame_features[row] = fv
        if np.all(np.isnan(fv)):
            n_empty += 1

    elapsed = time.monotonic() - t0
    fps_rate = total_fr / elapsed if elapsed > 0 else 0

    logger.info(
        '%s Done frame features for "%s": %d frames in %s (%.0f frm/s), %d empty frames',
        tag, ctx.video_name, total_fr, _fmt_duration(elapsed), fps_rate, n_empty,
    )

    # NaN summary per feature
    nan_counts = np.sum(np.isnan(frame_features), axis=0)
    if np.any(nan_counts > 0):
        worst_idx = int(np.argmax(nan_counts))
        from .frame_features import FRAME_FEATURE_NAMES
        logger.debug(
            '%s NaN summary: worst feature = %s (%d/%d NaN, %.1f%%)',
            tag, FRAME_FEATURE_NAMES[worst_idx],
            nan_counts[worst_idx], total_fr,
            100.0 * nan_counts[worst_idx] / max(total_fr, 1),
        )

    # Save cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"frame_features_{ctx.video_id}.npz"
        np.savez_compressed(
            cache_path,
            frame_features=frame_features,
            obj_a_key=obj_a_key,
            obj_b_key=obj_b_key,
            start_frame=start,
            num_frames=n,
            width=w,
            height=h,
            video_diagonal=float(np.hypot(w, h)),
        )
        cache_size = _fmt_size(cache_path.stat().st_size)
        logger.info('%s Cached frame features to %s (%s)', tag, cache_path.name, cache_size)

    return frame_features, obj_a_key, obj_b_key, start, n, w, h


# ---------------------------------------------------------------------------
# Full pipeline: all videos -> BehaviorDataset
# ---------------------------------------------------------------------------

def _fast_discover_videos(
    project_dir: Path,
    config: dict,
    *,
    progress: bool = True,
) -> tuple[list[VideoTrackContext], list[tuple[str, str, str]]]:
    """Fast video discovery: trust propagation_complete flag + masks.sqlite existence.

    Skips the slow per-frame mask existence check that partition_complete_videos does.
    """
    complete: list[VideoTrackContext] = []
    skipped: list[tuple[str, str, str]] = []
    items = list(config.get("videos", {}).items())

    logger.info("Fast-scanning %d video(s) (checking propagation_complete + masks.sqlite)...", len(items))

    vid_iter = _stdout_tqdm(
        items,
        desc="Scanning videos",
        unit="video",
        leave=True,
        disable=not progress,
    )

    for vid, v in vid_iter:
        vname = str(v.get("name") or vid)

        if not v.get("propagation_complete"):
            skipped.append((vid, vname, "propagation_complete is false"))
            continue

        vdir = video_storage_dir(project_dir, vid)
        if vdir is None:
            skipped.append((vid, vname, "missing video directory"))
            continue

        sqlite_path = vdir / "masks.sqlite"
        if not sqlite_path.is_file():
            skipped.append((vid, vname, "no masks.sqlite"))
            continue

        complete.append(VideoTrackContext(
            video_id=str(vid),
            video_name=vname,
            config=dict(v),
            video_dir=vdir,
        ))

    logger.info(
        "Fast scan done: %d eligible, %d skipped.",
        len(complete), len(skipped),
    )
    return complete, skipped


def extract_all_features(
    project_dir: Path,
    cfg: ClusteringConfig | None = None,
    *,
    progress: bool = True,
    skip_mask_verification: bool = False,
) -> BehaviorDataset:
    """Extract sequence features for all complete videos in a project.

    Loads masks from SQLite via VideoMaskStorage, computes per-frame features,
    then aggregates into sliding-window sequence vectors.

    Returns a BehaviorDataset with (N_total_windows, 33) features + metadata.
    """
    if cfg is None:
        cfg = ClusteringConfig()

    project_dir = Path(project_dir).resolve()
    logger.info("Loading project config from %s...", project_dir / "config.json")
    t0 = time.monotonic()
    config = load_project_config(project_dir)
    n_videos_total = len(config.get("videos", {}))
    logger.info("  Config loaded in %.1fs: %d videos in project.", time.monotonic() - t0, n_videos_total)

    output_base = project_dir / "analysis_of_tracking_data" / "clustering"
    cache_dir = output_base / "features"

    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION")
    logger.info("  Project: %s", project_dir.name)
    logger.info("  Videos in config: %d", n_videos_total)
    logger.info("  Window: %d frames, stride: %d frames", cfg.window_size, cfg.stride)
    logger.info("  Cache dir: %s", cache_dir)
    logger.info("  Mask verification: %s", "SKIP (fast)" if skip_mask_verification else "full per-frame check")
    logger.info("=" * 70)

    # Video discovery
    t0 = time.monotonic()
    if skip_mask_verification:
        complete, skipped = _fast_discover_videos(project_dir, config, progress=progress)
    else:
        logger.info(
            "Running full mask verification (this checks every frame of every video — "
            "use --skip-mask-verification for faster startup)..."
        )
        complete, skipped = partition_complete_videos(
            project_dir, config, progress=progress,
        )
    discovery_elapsed = time.monotonic() - t0

    logger.info(
        "Video discovery done in %s: %d complete, %d skipped",
        _fmt_duration(discovery_elapsed), len(complete), len(skipped),
    )
    for vid, vname, reason in skipped:
        logger.info("  SKIP: %s — %s", vname, reason)

    if not complete:
        logger.warning("No fully tracked videos found in %s", project_dir)
        return BehaviorDataset(
            features=np.empty((0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float64),
            metadata=[],
        )

    logger.info("Processing %d video(s) for feature extraction.", len(complete))

    all_features: list[np.ndarray] = []
    all_metadata: list[WindowMetadata] = []
    total_windows = 0
    pipeline_t0 = time.monotonic()

    vid_iter = _stdout_tqdm(
        list(enumerate(complete, start=1)),
        total=len(complete),
        desc="Videos",
        unit="video",
        leave=True,
        disable=not progress,
    )

    for vi, ctx in vid_iter:
        video_t0 = time.monotonic()

        result = _extract_video_frame_features(
            ctx, cfg,
            cache_dir=cache_dir,
            progress=progress,
            video_index=vi,
            total_videos=len(complete),
        )
        if result is None:
            continue

        frame_features, obj_a_key, obj_b_key, start, num_frames, w, h = result
        diag = float(np.hypot(w, h))
        area = float(w * h)

        # Compute sequence features
        logger.info(
            "[%d/%d] Computing sequence features (window=%d, stride=%d)...",
            vi, len(complete), cfg.window_size, cfg.stride,
        )
        seq_feats, starts = compute_sequence_features(
            frame_features,
            window_size=cfg.window_size,
            stride=cfg.stride,
            video_diagonal=diag,
            video_area=area,
            stationary_speed_threshold=cfg.stationary_speed_threshold,
            close_proximity_threshold=cfg.close_proximity_threshold,
            chase_max_lag=cfg.chase_max_lag,
        )

        if seq_feats.shape[0] == 0:
            logger.warning('[%d/%d] No windows produced for "%s"', vi, len(complete), ctx.video_name)
            continue

        # Build metadata for each window
        name_a = object_display_name(ctx.config, obj_a_key) or obj_a_key
        name_b = (object_display_name(ctx.config, obj_b_key) or obj_b_key) if obj_b_key else "none"
        cam_view = camera_view_from_folder(ctx.video_dir.name)
        itype = interaction_type_from_names(name_a, name_b)
        vid_session = session_from_video_name(ctx.video_name)
        proj_id = project_dir.name.split("-")[0]

        for i, s in enumerate(starts):
            all_metadata.append(WindowMetadata(
                video_id=ctx.video_id,
                video_name=ctx.video_name,
                object_a_key=obj_a_key,
                object_b_key=obj_b_key,
                object_a_name=name_a,
                object_b_name=name_b,
                start_frame=int(s) + start,  # absolute frame index
                window_size=cfg.window_size,
                camera_view=cam_view,
                interaction_type=itype,
                project_id=proj_id,
                session=vid_session,
            ))

        all_features.append(seq_feats)
        total_windows += seq_feats.shape[0]
        video_elapsed = time.monotonic() - video_t0

        # NaN fraction in sequence features
        nan_frac = np.mean(np.isnan(seq_feats)) * 100

        logger.info(
            '[%d/%d] "%s": %d windows | %s, %s | %s | NaN: %.1f%% | total so far: %d windows',
            vi, len(complete), ctx.video_name, seq_feats.shape[0],
            name_a, name_b, itype, nan_frac, total_windows,
        )

        # ETA estimate
        elapsed_total = time.monotonic() - pipeline_t0
        rate = vi / elapsed_total if elapsed_total > 0 else 0
        remaining = (len(complete) - vi) / rate if rate > 0 else 0
        logger.info(
            "[%d/%d] Video took %s | Elapsed: %s | ETA: ~%s",
            vi, len(complete),
            _fmt_duration(video_elapsed),
            _fmt_duration(elapsed_total),
            _fmt_duration(remaining),
        )

    pipeline_elapsed = time.monotonic() - pipeline_t0

    if not all_features:
        logger.warning("No features extracted from any video.")
        return BehaviorDataset(
            features=np.empty((0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float64),
            metadata=[],
        )

    features = np.vstack(all_features)

    # Feature quality summary
    nan_per_feature = np.sum(np.isnan(features), axis=0)
    n_total = features.shape[0]
    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("  Total windows: %d", n_total)
    logger.info("  Features per window: %d", features.shape[1])
    logger.info("  Total time: %s", _fmt_duration(pipeline_elapsed))
    logger.info("  Videos processed: %d", len(complete))
    logger.info(
        "  Feature matrix: %s",
        _fmt_size(features.nbytes),
    )
    # Report features with most NaNs
    if np.any(nan_per_feature > 0):
        worst = np.argsort(nan_per_feature)[::-1][:5]
        logger.info("  Features with most NaN:")
        for idx in worst:
            cnt = nan_per_feature[idx]
            if cnt == 0:
                break
            logger.info(
                "    %s: %d/%d (%.1f%%)",
                SEQUENCE_FEATURE_NAMES[idx], cnt, n_total, 100.0 * cnt / n_total,
            )
    else:
        logger.info("  No NaN values in any features.")
    logger.info("=" * 70)

    dataset = BehaviorDataset(
        features=features,
        metadata=all_metadata,
    )

    # Save the combined dataset
    ds_path = cache_dir / "sequence_features.npz"
    dataset.save(ds_path)
    logger.info("Saved dataset to %s (%s)", ds_path, _fmt_size(ds_path.stat().st_size))

    return dataset
