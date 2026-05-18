#!/usr/bin/env python3
"""Extract frame features for a single video. Designed to run as a bsub job.

Usage:
    python extract_single_video.py --project-dir /path/to/project --video-id abc12345

Saves cached frame features to:
    <project_dir>/analysis_of_tracking_data/clustering/features/frame_features_<video_id>.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Force unbuffered output for bsub logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract frame features for a single video.")
    p.add_argument("--project-dir", type=Path, required=True, help="Project folder with config.json")
    p.add_argument("--video-id", type=str, required=True, help="Video UUID (8-char id)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # Configure logging immediately
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SINGLE VIDEO FEATURE EXTRACTION")
    logger.info("  Project: %s", args.project_dir)
    logger.info("  Video ID: %s", args.video_id)
    logger.info("  Host: %s", __import__("socket").gethostname())
    logger.info("  PID: %d", __import__("os").getpid())
    logger.info("=" * 60)

    project_dir = Path(args.project_dir).resolve()
    if not (project_dir / "config.json").is_file():
        logger.error("No config.json in %s", project_dir)
        return 1

    # Add backend to path
    backend_dir = Path(__file__).resolve().parents[2] / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # Add repo root to path for downstream_analysis imports
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    logger.info("Loading project config...")
    from downstream_analysis.tracking_io import load_project_config, video_storage_dir, VideoTrackContext
    from downstream_analysis.clustering.config import ClusteringConfig
    from downstream_analysis.clustering.feature_extraction import _extract_video_frame_features

    config = load_project_config(project_dir)
    videos = config.get("videos", {})

    # Find the video
    vid = args.video_id.strip()
    if vid not in videos:
        logger.error("Video ID '%s' not found in config. Available: %s",
                      vid, list(videos.keys())[:5])
        return 1

    v = videos[vid]
    vname = str(v.get("name") or vid)
    vdir = video_storage_dir(project_dir, vid)
    if vdir is None:
        logger.error("No video directory found for %s", vid)
        return 1

    ctx = VideoTrackContext(
        video_id=vid,
        video_name=vname,
        config=dict(v),
        video_dir=vdir,
    )

    cfg = ClusteringConfig()
    cache_dir = project_dir / "analysis_of_tracking_data" / "clustering" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Video: %s", vname)
    logger.info("Frames: %d (start=%d)", v.get("num_frames", 0), v.get("start_frame", 0))
    logger.info("Resolution: %dx%d", v.get("width", 0), v.get("height", 0))
    logger.info("Cache dir: %s", cache_dir)

    t0 = time.monotonic()
    result = _extract_video_frame_features(
        ctx, cfg,
        cache_dir=cache_dir,
        progress=True,
        video_index=1,
        total_videos=1,
    )
    elapsed = time.monotonic() - t0

    if result is None:
        logger.error("Feature extraction FAILED for %s", vname)
        return 1

    ff, obj_a, obj_b, start, n, w, h = result
    logger.info("=" * 60)
    logger.info("SUCCESS")
    logger.info("  Video: %s (%s)", vname, vid)
    logger.info("  Shape: %s", ff.shape)
    logger.info("  Objects: %s, %s", obj_a, obj_b)
    logger.info("  Time: %.1fs (%.1f frm/s)", elapsed, ff.shape[0] / max(elapsed, 0.001))
    logger.info("  Cache: %s", cache_dir / f"frame_features_{vid}.npz")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
