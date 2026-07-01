#!/usr/bin/env python3
"""
Backfill display/*.webp (and optionally frames/*.jpg) for an existing propagated video.

Usage:
  conda run -n sam3 python scripts/backfill_display_cache.py PROJECT_ID VIDEO_ID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from display_cache import build_manifest, write_display_webp, write_manifest  # noqa: E402
from mask_store import VideoMaskStorage  # noqa: E402
from project_manager import ProjectManager  # noqa: E402


def _object_display_colors(objects: dict) -> dict[str, str]:
    return {
        str(oid): obj.get("color", "#5B8DD9")
        for oid, obj in objects.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill display WebP scrub cache")
    parser.add_argument("pid")
    parser.add_argument("vid")
    parser.add_argument("--from-frame", type=int, default=0)
    parser.add_argument("--to-frame", type=int, default=-1)
    args = parser.parse_args()

    pm = ProjectManager()
    video = pm.get_video(args.pid, args.vid)
    if video is None:
        raise SystemExit(f"Video not found: {args.pid}/{args.vid}")

    video_dir = pm.video_dir(args.pid, args.vid)
    ms = VideoMaskStorage(video_dir)
    propagated = pm._read_propagated_frames(args.pid, args.vid)
    if not propagated:
        print("No propagated frames — nothing to backfill.")
        return

    lo = args.from_frame
    hi = args.to_frame if args.to_frame >= 0 else max(propagated)
    colors = _object_display_colors(video.get("objects", {}))
    w = int(video.get("width") or 1)
    h = int(video.get("height") or 1)

    done = 0
    for fidx in sorted(propagated):
        if fidx < lo or fidx > hi:
            continue
        if not ms.has_masks(fidx):
            continue
        masks = ms.load_masks_dense(fidx)
        if not masks:
            continue
        out = write_display_webp(video_dir, fidx, masks, colors, w, h)
        if out:
            done += 1
            if done % 100 == 0:
                print(f"  {done} frames...", flush=True)

    manifest = write_manifest(
        video_dir, ms.cache_revision_int(), w, h, propagated
    )
    print(f"Backfilled {done} display WebP files.")
    print(f"Manifest: {manifest['display']['count']} display, {manifest['bytes_total']} bytes total.")


if __name__ == "__main__":
    main()
