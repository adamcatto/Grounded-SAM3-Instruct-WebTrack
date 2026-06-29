"""Fork an anchor-memory project into a queue-memory ablation project."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import MANIFEST_NAME, RECENT_QUEUE_PREFIX
from .common import (
    find_video_dir,
    load_config,
    resolve_project_dir,
    slugify,
    source_file_for,
    write_config,
)


def _reset_video_meta(src_meta: dict[str, Any], video_dir: Path, source_file: Path) -> dict[str, Any]:
    vm = dict(src_meta)
    vm["source_path"] = str(video_dir / source_file.name)
    vm["sam3_session_id"] = None
    vm["propagated_frames"] = []
    vm["propagation_complete"] = False
    vm["whole_video_inference"] = {"status": "none", "updated_at": None, "host": None}
    vm["anchor_labeling_complete"] = True
    return vm


def build_recent_queue_project(
    source_ref: str,
    *,
    output_parent: Path | None = None,
    only_videos: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a label-only project that will be tracked with standard queue memory."""
    source_dir = resolve_project_dir(source_ref)
    src_cfg = load_config(source_dir)
    source_name = str(src_cfg.get("name") or source_dir.name)
    parent = (output_parent or source_dir.parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    pid = str(uuid.uuid4())[:8]
    project_name = f"{RECENT_QUEUE_PREFIX}{source_name}"
    project_dir = parent / f"{pid}-{slugify(project_name)}"
    if project_dir.exists():
        if not overwrite:
            raise FileExistsError(project_dir)
        shutil.rmtree(project_dir)
    (project_dir / "videos").mkdir(parents=True)

    now = datetime.now(timezone.utc).isoformat()
    new_cfg: dict[str, Any] = {
        "id": pid,
        "name": project_name,
        "created_at": now,
        "videos": {},
    }
    manifest: dict[str, Any] = {
        "experiment": "anchor_frame_memory",
        "source_project_dir": str(source_dir),
        "source_project_id": src_cfg.get("id"),
        "source_project_name": source_name,
        "queue_project_dir": str(project_dir),
        "queue_project_id": pid,
        "queue_project_name": project_name,
        "tracking_mode": "recent_queue_memory",
        "ground_truth_mode": "anchor_frame_memory",
        "created_at": now,
        "videos": [],
    }

    only = set(only_videos) if only_videos else None
    skipped: list[tuple[str, str]] = []
    for vid, meta in (src_cfg.get("videos") or {}).items():
        if only is not None and vid not in only:
            continue
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("name") or vid)
        src_video_dir = find_video_dir(source_dir, vid)
        if src_video_dir is None:
            skipped.append((name, "missing source video directory"))
            continue
        if not meta.get("point_prompts"):
            skipped.append((name, "no point prompts"))
            continue
        source_file = source_file_for(src_video_dir)
        if source_file is None:
            skipped.append((name, "missing source.* file"))
            continue

        new_video_dir = project_dir / "videos" / f"{vid}_{slugify(name)}"
        (new_video_dir / "masks").mkdir(parents=True)
        (new_video_dir / "bboxes").mkdir()
        link = new_video_dir / source_file.name
        try:
            link.symlink_to(source_file)
        except OSError:
            link.symlink_to(source_file.resolve())

        new_cfg["videos"][vid] = _reset_video_meta(meta, new_video_dir, source_file)
        manifest["videos"].append({
            "video_id": vid,
            "video_name": name,
            "source_video_id": vid,
            "source_video_dir": str(src_video_dir),
            "queue_video_id": vid,
            "queue_video_dir": str(new_video_dir),
            "num_frames": int(meta.get("num_frames") or 0),
            "start_frame": int(meta.get("start_frame") or 0),
            "anchor_batch_size": int(meta.get("anchor_batch_size") or 1000),
            "annotated_anchors": [int(x) for x in (meta.get("annotated_anchors") or [])],
        })

    if not new_cfg["videos"]:
        shutil.rmtree(project_dir)
        reasons = "; ".join(f"{n}: {r}" for n, r in skipped) or "no eligible videos"
        raise RuntimeError(f"No videos emitted from {source_name!r}: {reasons}")

    write_config(project_dir, new_cfg)
    (project_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    print(f"[build] source anchor project : {source_dir}")
    print(f"[build] recent-queue project : {project_dir}")
    print(f"[build] videos               : {len(new_cfg['videos'])}")
    for n, r in skipped:
        print(f"[build] SKIP {n}: {r}")
    return project_dir


def load_manifest(project_dir: Path) -> dict[str, Any]:
    path = project_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{MANIFEST_NAME} not found under {project_dir}")
    return json.loads(path.read_text())
