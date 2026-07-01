"""
Pre-materialized JPEG frames and display WebP overlays for fast scrubbing.

Written during propagation and via scripts/backfill_display_cache.py.
Served via FileResponse — no runtime encode on the hot path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from video_processor import hex_to_rgb

logger = logging.getLogger(__name__)

MANIFEST_NAME = "cache_manifest.json"
DISPLAY_SUBDIR = "display"
DISPLAY_ALPHA = 0.45


def display_dir(video_dir: Path) -> Path:
    d = Path(video_dir) / DISPLAY_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def display_path(video_dir: Path, frame_idx: int) -> Path:
    return display_dir(video_dir) / f"{frame_idx:06d}.webp"


def manifest_path(video_dir: Path) -> Path:
    return Path(video_dir) / MANIFEST_NAME


def composite_masks_rgba(
    masks: dict[str, np.ndarray],
    colors: dict[str, str],
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Simple flat-fill composite (no contour/glow) for fast WebP encoding."""
    rgba = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    for obj_id, mask in masks.items():
        color_hex = colors.get(str(obj_id), "#5B8DD9")
        r, g, b = hex_to_rgb(color_hex)
        mask_bool = np.squeeze(mask).astype(bool)
        if mask_bool.shape != (frame_h, frame_w):
            continue
        rgba[mask_bool, 0] = r
        rgba[mask_bool, 1] = g
        rgba[mask_bool, 2] = b
        rgba[mask_bool, 3] = int(255 * DISPLAY_ALPHA)
    return rgba


def write_display_webp(
    video_dir: Path,
    frame_idx: int,
    masks: dict[str, np.ndarray],
    colors: dict[str, str],
    frame_w: int,
    frame_h: int,
) -> Path | None:
    """Write display/{frame:06d}.webp. Returns path if written."""
    if not masks:
        delete_display_frame(video_dir, frame_idx)
        return None
    out = display_path(video_dir, frame_idx)
    try:
        rgba = composite_masks_rgba(masks, colors, frame_h, frame_w)
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(str(out), format="WEBP", quality=85, method=4)
        return out
    except Exception as e:
        logger.warning(f"write_display_webp frame {frame_idx}: {e}")
        return None


def copy_frame_jpg(
    frames_src_dir: str | Path,
    frame_idx: int,
    frames_dst_dir: Path,
) -> bool:
    """Copy a batch temp JPG into persistent frames/ (skip if already present)."""
    src = Path(frames_src_dir) / f"{frame_idx:06d}.jpg"
    if not src.is_file():
        return False
    frames_dst_dir.mkdir(parents=True, exist_ok=True)
    dst = frames_dst_dir / f"{frame_idx:06d}.jpg"
    if dst.exists():
        return True
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.warning(f"copy_frame_jpg {frame_idx}: {e}")
        return False


def delete_display_frame(video_dir: Path, frame_idx: int) -> None:
    p = display_path(video_dir, frame_idx)
    if p.is_file():
        p.unlink()


def clear_display_cache(video_dir: Path) -> None:
    d = Path(video_dir) / DISPLAY_SUBDIR
    if d.is_dir():
        shutil.rmtree(d)
    mp = manifest_path(video_dir)
    if mp.is_file():
        mp.unlink()


def _dir_stats(directory: Path, ext: str) -> tuple[int, int]:
    if not directory.is_dir():
        return 0, 0
    files = list(directory.glob(f"*.{ext}"))
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    return len(files), total_bytes


def build_manifest(
    video_dir: Path,
    cache_revision: int,
    frame_w: int,
    frame_h: int,
    propagated_frames: list[int] | None = None,
) -> dict:
    video_dir = Path(video_dir)
    display_d = video_dir / DISPLAY_SUBDIR
    frames_d = video_dir / "frames"
    display_count, display_bytes = _dir_stats(display_d, "webp")
    frames_count, frames_bytes = _dir_stats(frames_d, "jpg")
    propagated = sorted(set(propagated_frames or []))
    return {
        "cache_revision": cache_revision,
        "display": {
            "dir": DISPLAY_SUBDIR,
            "ext": "webp",
            "width": frame_w,
            "height": frame_h,
            "count": display_count,
            "bytes": display_bytes,
        },
        "frames": {
            "dir": "frames",
            "ext": "jpg",
            "count": frames_count,
            "bytes": frames_bytes,
        },
        "propagated_frames": propagated,
        "bytes_total": display_bytes + frames_bytes,
    }


def write_manifest(
    video_dir: Path,
    cache_revision: int,
    frame_w: int,
    frame_h: int,
    propagated_frames: list[int] | None = None,
) -> dict:
    manifest = build_manifest(
        video_dir, cache_revision, frame_w, frame_h, propagated_frames
    )
    manifest_path(video_dir).write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    return manifest
