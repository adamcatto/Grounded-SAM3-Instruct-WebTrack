#!/usr/bin/env python3
"""
Export every video in a SAM3 project to an MP4 with tracked masks burned in.

Reuses the backend's export_video_with_masks (same code path as the UI's
Export button) but writes all videos into one output directory instead of
each video's own folder.

Reads only the source video + persisted masks (masks.sqlite / legacy masks/*.npz).
Does not touch SAM sessions, annotated_frames, or inference state.

Usage (run inside the `sam3` conda env so cv2 + ffmpeg are available):

    conda run -n sam3 python scripts/export_videos_to_mp4.py \
        /opt/projects/Adam/2026/tracked_behavior_projects/3c9bddf2-Home-Cage-Interactions-0126-test-day

    # Custom output dir (default is <project>/vos_mp4):
    conda run -n sam3 python scripts/export_videos_to_mp4.py <project_dir> --out /some/where
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Make backend importable (video_processor imports mask_store by bare name).
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from video_processor import export_video_with_masks  # noqa: E402


def _safe_stem(name: str, fallback: str) -> str:
    stem = Path(name).stem or fallback           # strip any .mp4 etc.
    stem = re.sub(r"[^\w\-.]", "_", stem).strip("_.-")
    return stem or fallback


def _find_video_dir(videos_root: Path, vid: str) -> Path | None:
    for d in videos_root.iterdir():
        if d.is_dir() and (d.name == vid or d.name.startswith(vid + "_")):
            return d
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_dir", help="Project folder containing config.json")
    ap.add_argument("--out", default=None,
                    help="Output dir for the MP4s (default: <project_dir>/vos_mp4)")
    args = ap.parse_args()

    proj = Path(args.project_dir).expanduser().resolve()
    cfg_path = proj / "config.json"
    if not cfg_path.is_file():
        print(f"ERROR: no config.json at {cfg_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve() if args.out else proj / "vos_mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text())
    videos = cfg.get("videos") or {}
    if not videos:
        print("No videos in project.")
        return 0

    videos_root = proj / "videos"
    n_ok = n_skip = 0

    for vid, vm in videos.items():
        if not isinstance(vm, dict):
            continue
        name = vm.get("name") or vid
        vdir = _find_video_dir(videos_root, vid)
        if vdir is None:
            print(f"SKIP {vid} ({name}): no video directory")
            n_skip += 1
            continue

        source_path = vm.get("source_path") or ""
        if not source_path or not Path(source_path).exists():
            print(f"SKIP {vid} ({name}): source_path missing/unresolved -> {source_path!r}")
            n_skip += 1
            continue

        objects = vm.get("objects") or {}
        colors = {oid: o.get("color", "#5B8DD9") for oid, o in objects.items()}
        labels = {oid: o.get("name", f"Object {oid}") for oid, o in objects.items()}

        out_path = out_dir / f"{vid}_{_safe_stem(name, vid)}.mp4"
        print(f"\n=== {name}  ->  {out_path.name}")

        def _progress(frame_idx: int, total: int):
            pct = 100 * frame_idx / max(total, 1)
            print(f"\r  {frame_idx}/{total} ({pct:5.1f}%)", end="", flush=True)

        try:
            result = export_video_with_masks(
                source_path=source_path,
                out_path=str(out_path),
                video_dir=str(vdir),
                colors=colors,
                labels=labels,
                num_frames_hint=int(vm.get("num_frames") or 0),
                progress_callback=_progress,
            )
            print(f"\r  done: {result['total_frames']} frames @ {result.get('fps')} fps")
            n_ok += 1
        except Exception as exc:
            print(f"\r  FAILED: {exc}")
            n_skip += 1

    print(f"\nExported {n_ok} video(s), skipped {n_skip}. Output: {out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
