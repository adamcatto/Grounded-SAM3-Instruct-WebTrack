#!/usr/bin/env python3
"""Create a local projects root + demo project, importing every *.mp4 in an inbox folder.

Usage:
  python scripts/setup_agent_test_project.py
  python scripts/setup_agent_test_project.py --inbox /tmp/sam3wt_agent_inbox --write-synthetic
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def write_synthetic_clips(inbox: Path, n_frames: int = 40) -> list[Path]:
    import cv2
    import numpy as np

    inbox.mkdir(parents=True, exist_ok=True)
    clips = []
    specs = [
        ("cage_a_two_mice.mp4", (36, 160, 36), (48, 70), (118, 55)),
        ("cage_b_two_mice.mp4", (36, 36, 160), (42, 80), (122, 48)),
    ]
    for name, bg, c1, c2 in specs:
        path = inbox / name
        wri = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
        if not wri.isOpened():
            raise RuntimeError(f"Could not write {path}")
        for i in range(n_frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = bg
            cv2.circle(frame, (c1[0] + (i % 12), c1[1]), 16, (230, 230, 230), -1)
            cv2.circle(frame, (c1[0] + (i % 12), c1[1] - 6), 5, (80, 80, 80), -1)  # "shave"
            cv2.circle(frame, (c2[0], c2[1] + (i % 7)), 14, (20, 20, 20), -1)
            wri.write(frame)
        wri.release()
        clips.append(path)
    return clips


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inbox", default="/tmp/sam3wt_agent_inbox")
    p.add_argument("--projects-root", default="/tmp/sam3wt_agent_projects")
    p.add_argument("--project-name", default="agent_prompt_demo")
    p.add_argument("--write-synthetic", action="store_true")
    args = p.parse_args()

    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "DROP_MP4S_HERE.txt").write_text(
        "Put real behavior MP4s in this folder, then re-run:\n"
        "  python scripts/setup_agent_test_project.py --inbox /tmp/sam3wt_agent_inbox\n"
    )

    if args.write_synthetic:
        clips = write_synthetic_clips(inbox)
        print(f"Wrote synthetic clips: {[c.name for c in clips]}")

    videos = sorted(inbox.glob("*.mp4")) + sorted(inbox.glob("*.MP4"))
    if not videos:
        print(f"No MP4 files in {inbox}. Add videos and re-run.")
        return 1

    import os
    os.environ["SAM3_TRACKING_PROJECTS_DIR"] = args.projects_root
    from project_manager import ProjectManager

    def _video_info(path: Path) -> dict:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"num_frames": n, "fps": fps, "width": w, "height": h}

    pm = ProjectManager()
    existing = next((pr for pr in pm.list_projects() if pr.get("name") == args.project_name), None)
    project = existing or pm.create_project(args.project_name)
    pid = project["id"]

    imported = []
    for src in videos:
        dest_dir = None  # copies go to projects_root/_inbox_copies
        info = _video_info(src)
        # Copy into a stable inbox-adjacent location referenced by config
        dest = Path(args.projects_root) / "_inbox_copies" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        already = any(v.get("name") == src.name for v in (pm.get_project(pid) or {}).get("videos", {}).values())
        if already:
            print(f"skip existing {src.name}")
            continue
        meta = pm.add_video(
            pid, src.name, str(dest),
            num_frames=int(info.get("num_frames") or 0),
            fps=float(info.get("fps") or 30),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
        )
        imported.append(meta)
        print(f"imported {src.name} -> video {meta['id']} ({meta['num_frames']} frames)")

    print(f"PROJECT_ID={pid}")
    print(f"PROJECTS_ROOT={args.projects_root}")
    print(f"INBOX={inbox}")
    print(f"imported={len(imported)} videos={len((pm.get_project(pid) or {}).get('videos') or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
