#!/usr/bin/env python3
"""
Import legacy masks/*.npz (+ optional bboxes/*.json) into masks.sqlite per video.

Run from repo root with backend on PYTHONPATH, e.g.:
  conda run -n sam3 python scripts/migrate_masks_sqlite.py --video-dir /path/to/project/videos/<uuid>_name

Or migrate every propagation-complete video in a project:
  python scripts/migrate_masks_sqlite.py --project-dir /path/to/project --all-complete
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mask_seg_codec import decode_masks_blob, encode_masks_blob  # noqa: E402
from mask_store import VideoMaskStorage, legacy_npz_frame_indices  # noqa: E402
from video_processor import (  # noqa: E402
    bbox_norm_xywh_score_from_mask,
    load_masks_npz,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_masks_sqlite")


def _bbox_dict_for_masks(
    masks: dict[str, np.ndarray],
    existing_json: dict[str, list] | None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for oid, m in masks.items():
        if existing_json and oid in existing_json:
            out[oid] = [float(x) for x in existing_json[oid]]
        else:
            bx, sc = bbox_norm_xywh_score_from_mask(m)
            out[oid] = list(bx) + [float(sc)]
    return out


def _masks_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if not np.array_equal(np.asarray(a[k]).astype(np.uint8), np.asarray(b[k]).astype(np.uint8)):
            return False
    return True


def _bbox_close(a: dict, b: dict, atol: float = 1e-5) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        xa = [float(x) for x in a[k]]
        xb = [float(x) for x in b[k]]
        if len(xa) != len(xb):
            return False
        for u, v in zip(xa, xb):
            if abs(u - v) > atol:
                return False
    return True


def migrate_video(
    video_dir: Path,
    *,
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Returns (frames_imported, npz_frames_seen, bbox_json_missing_count).
    """
    masks_dir = video_dir / "masks"
    indices = legacy_npz_frame_indices(masks_dir)
    npz_seen = len(indices)
    bbox_missing = 0
    rows_ready: list[tuple[int, bytes, str]] = []

    for fi in indices:
        npz_path = masks_dir / f"{fi:06d}.npz"
        masks = load_masks_npz(str(npz_path))
        if not masks:
            continue
        masks_u8 = {str(k): np.asarray(v).astype(np.uint8) for k, v in masks.items()}
        bbox_path = video_dir / "bboxes" / f"{fi:06d}.json"
        bbox_existing: dict[str, list] | None = None
        if bbox_path.is_file():
            try:
                bbox_existing = json.loads(bbox_path.read_text())
            except json.JSONDecodeError:
                bbox_existing = {}
        else:
            bbox_missing += 1
            bbox_existing = {}

        bbox_final = _bbox_dict_for_masks(masks_u8, bbox_existing)
        blob = encode_masks_blob(masks_u8)
        rows_ready.append((fi, blob, json.dumps(bbox_final, separators=(",", ":"))))

    imported = len(rows_ready)
    logger.info(
        "Video %s: %d npz stem(s), %d non-empty frame row(s), %d missing bbox JSON.",
        video_dir.name,
        npz_seen,
        imported,
        bbox_missing,
    )

    if dry_run or imported == 0:
        return imported, npz_seen, bbox_missing

    storage = VideoMaskStorage(video_dir)
    conn = storage._connect()
    try:
        total_rows = len(rows_ready)
        for batch_start in range(0, total_rows, batch_size):
            chunk = rows_ready[batch_start : batch_start + batch_size]
            conn.execute("BEGIN IMMEDIATE")
            for fi, blob, bjson in chunk:
                conn.execute(
                    """INSERT OR REPLACE INTO frame_segmentation
                       (frame_idx, seg_blob, bbox_json, updated_at)
                       VALUES (?,?,?,datetime('now'))""",
                    (fi, blob, bjson),
                )
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        storage._bump_revision_locked(conn)
        conn.commit()
    finally:
        conn.close()

    logger.info("Wrote %s (%d rows).", storage.sqlite_path, imported)
    return imported, npz_seen, bbox_missing


def verify_frames(video_dir: Path, frame_indices: list[int]) -> tuple[bool, str]:
    masks_dir = video_dir / "masks"
    ms = VideoMaskStorage(video_dir)
    conn = ms._connect()
    try:
        for fi in frame_indices:
            row = conn.execute(
                "SELECT seg_blob, bbox_json FROM frame_segmentation WHERE frame_idx=?",
                (fi,),
            ).fetchone()
            if row is None:
                return False, f"missing sqlite row for frame {fi}"
            dec = decode_masks_blob(row[0])
            leg = {
                str(k): np.asarray(v).astype(np.uint8)
                for k, v in load_masks_npz(str(masks_dir / f"{fi:06d}.npz")).items()
            }
            if not _masks_equal(dec, leg):
                return False, f"mask mismatch frame {fi}"

            jj = json.loads(row[1])
            bbox_path = video_dir / "bboxes" / f"{fi:06d}.json"
            if bbox_path.is_file():
                disk_bb = json.loads(bbox_path.read_text())
                if not _bbox_close(jj, disk_bb):
                    return False, f"bbox mismatch frame {fi} (disk json)"
            else:
                derived = _bbox_dict_for_masks(dec, {})
                if not _bbox_close(jj, derived):
                    return False, f"bbox mismatch frame {fi} (derived)"
        return True, ""
    finally:
        conn.close()


def delete_legacy_masks(video_dir: Path) -> None:
    md = video_dir / "masks"
    bd = video_dir / "bboxes"
    if md.is_dir():
        for p in md.glob("*.npz"):
            try:
                p.unlink()
            except OSError:
                pass
    if bd.is_dir():
        for p in bd.glob("*.json"):
            try:
                p.unlink()
            except OSError:
                pass


def load_project_config(project_dir: Path) -> dict:
    return json.loads((project_dir / "config.json").read_text())


def video_storage_dir(project_dir: Path, vid: str) -> Path | None:
    root = project_dir / "videos"
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.name == vid or d.name.startswith(vid + "_"):
            return d
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate legacy NPZ masks into masks.sqlite")
    ap.add_argument("--video-dir", type=Path, help="Video directory (contains masks/, bboxes/)")
    ap.add_argument("--project-dir", type=Path, help="Project root with config.json")
    ap.add_argument("--video-id", type=str, help="Video UUID when using --project-dir")
    ap.add_argument(
        "--all-complete",
        action="store_true",
        help="Migrate all videos where propagation_complete is true",
    )
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-sample", type=int, default=8, help="Random frames to verify (0=skip)")
    ap.add_argument("--verify-full", action="store_true", help="Verify every migrated frame")
    ap.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Remove masks/*.npz and bboxes/*.json after successful verify",
    )
    ap.add_argument("--yes", action="store_true", help="Confirm --delete-legacy")
    args = ap.parse_args()

    video_dirs: list[Path] = []
    if args.video_dir:
        video_dirs.append(args.video_dir.resolve())
    elif args.project_dir and args.video_id:
        vdir = video_storage_dir(args.project_dir.resolve(), args.video_id)
        if vdir is None:
            logger.error("Video directory not found for id %s", args.video_id)
            return 2
        video_dirs.append(vdir)
    elif args.project_dir and args.all_complete:
        cfg = load_project_config(args.project_dir.resolve())
        for vid, v in cfg.get("videos", {}).items():
            if not v.get("propagation_complete"):
                continue
            vdir = video_storage_dir(args.project_dir.resolve(), vid)
            if vdir is not None:
                video_dirs.append(vdir)
        logger.info("--all-complete: %d video director(y/ies).", len(video_dirs))
    else:
        ap.error("Specify --video-dir or (--project-dir and --video-id) or (--project-dir and --all-complete)")
        return 2

    rng_frames = []
    for vd in video_dirs:
        imported, npz_seen, bbox_miss = migrate_video(
            vd, batch_size=args.batch_size, dry_run=args.dry_run
        )
        if args.dry_run:
            continue

        idx_all = legacy_npz_frame_indices(vd / "masks")
        if not idx_all:
            logger.warning("Skip verify: no legacy npz under %s", vd)
            continue

        if args.verify_full:
            check_idx = idx_all
        elif args.verify_sample > 0:
            check_idx = sorted(
                random.sample(idx_all, min(args.verify_sample, len(idx_all)))
            )
        else:
            check_idx = []

        if check_idx:
            ok, msg = verify_frames(vd, check_idx)
            if not ok:
                logger.error("VERIFY FAILED %s: %s", vd, msg)
                return 3
            logger.info("Verify OK (%d frame(s)) for %s", len(check_idx), vd.name)

        if args.delete_legacy:
            if not args.yes:
                logger.error("--delete-legacy requires --yes")
                return 4
            if args.verify_sample <= 0 and not args.verify_full:
                logger.error("--delete-legacy requires --verify-sample > 0 or --verify-full")
                return 4
            delete_legacy_masks(vd)
            logger.info("Deleted legacy npz/json under %s", vd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
