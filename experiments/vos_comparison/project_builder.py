"""
Emit a single-shot VOS project from a fully-tracked anchor-frame project.

The new project ``single_shot_vos_<slug>`` is created next to the source project
(same projects base dir, so the backend can find it). For every video:

  * the source video file is **symlinked** (no copy of pixels or masks),
  * only the *first* labeled frame's point prompts are kept — every later anchor
    is dropped, so propagation has a single human seed,
  * all propagation / inference state is reset to "not tracked",
  * empty ``masks`` / ``bboxes`` dirs are created for the predicted output.

A manifest (``vos_comparison_manifest.json``) records the link from each emitted
video back to its ground-truth video directory so evaluation needs no guessing.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    MANIFEST_NAME,
    SINGLE_SHOT_PREFIX,
    find_video_dir,
    first_labeled_frame,
    load_config,
    projects_base_dir,
    resolve_project_dir,
    slugify,
)


def _resolve_source_file(src_video_dir: Path) -> Path | None:
    """Return the real source.<ext> path inside a source video dir (deref symlinks)."""
    for cand in sorted(src_video_dir.glob("source.*")):
        if cand.is_file() or cand.is_symlink():
            return cand.resolve()
    return None


def _single_shot_video_meta(
    src_meta: dict[str, Any],
    new_vid: str,
    new_video_dir: Path,
    first_frame: int,
    source_file: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Copy source video metadata, keep only the first anchor, reset tracking."""
    vm = dict(src_meta)
    vm["id"] = new_vid
    vm["source_path"] = str(new_video_dir / source_file.name)
    vm["anchor_batch_size"] = int(batch_size)

    # Keep only the first labeled frame's prompts for each object.
    fk = str(first_frame)
    pruned: dict[str, Any] = {}
    for obj_id, obj_prompts in (src_meta.get("point_prompts") or {}).items():
        if isinstance(obj_prompts, dict) and fk in obj_prompts:
            pruned[obj_id] = {fk: obj_prompts[fk]}
    vm["point_prompts"] = pruned

    # The single retained anchor is the only annotated frame.
    vm["annotated_anchors"] = [int(first_frame)]
    vm["start_frame"] = int(first_frame)

    # Reset every "tracked" marker so the runner will (re)propagate.
    vm["sam3_session_id"] = None
    vm["propagated_frames"] = []
    vm["propagation_complete"] = False
    # Mark labeling "complete" = ready to propagate. There is intentionally only
    # one anchor, but the video IS ready to track, and this flag is what the HPC
    # parallel worker (run_pending_inference.video_eligibility) gates on when
    # claiming videos across many bsub jobs. Without it the single-shot videos
    # would be skipped as "anchor labeling incomplete".
    vm["anchor_labeling_complete"] = True
    vm["whole_video_inference"] = {"status": "none", "updated_at": None, "host": None}
    return vm


def build_single_shot_project(
    source_ref: str,
    *,
    output_parent: Path | None = None,
    batch_size: str | int = "inherit",
    only_videos: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Create the single-shot project and return its directory.

    batch_size:
      * "inherit" — reuse each source video's anchor_batch_size (controlled
        comparison: only the human anchors are removed; cross-batch mask handoff
        is identical to the anchor run).
      * "single"  — one batch covering the whole video (purest single-shot: no
        cross-batch re-seeding at all).
      * int       — explicit frames-per-batch for every video.
    """
    source_dir = resolve_project_dir(source_ref)
    src_cfg = load_config(source_dir)
    src_name = str(src_cfg.get("name") or source_dir.name)

    parent = (output_parent or source_dir.parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    pid = str(uuid.uuid4())[:8]
    proj_name = f"{SINGLE_SHOT_PREFIX}{src_name}"
    dir_name = f"{pid}-{slugify(proj_name)}"
    project_dir = parent / dir_name
    if project_dir.exists():
        if not overwrite:
            raise ValueError(f"Output project already exists: {project_dir}")
        import shutil

        shutil.rmtree(project_dir)
    (project_dir / "videos").mkdir(parents=True)

    new_config: dict[str, Any] = {
        "id": pid,
        "name": proj_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videos": {},
    }
    manifest: dict[str, Any] = {
        "source_project_dir": str(source_dir),
        "source_project_id": src_cfg.get("id"),
        "source_project_name": src_name,
        "single_shot_project_dir": str(project_dir),
        "single_shot_project_id": pid,
        "batch_size_mode": batch_size,
        "created_at": new_config["created_at"],
        "videos": [],
    }

    only = set(only_videos) if only_videos else None
    skipped: list[tuple[str, str]] = []

    for src_vid, src_meta in (src_cfg.get("videos") or {}).items():
        if only is not None and src_vid not in only:
            continue
        if not isinstance(src_meta, dict):
            continue
        vname = str(src_meta.get("name") or src_vid)

        src_video_dir = find_video_dir(source_dir, src_vid)
        if src_video_dir is None:
            skipped.append((vname, "missing video directory"))
            continue
        if not src_meta.get("propagation_complete"):
            skipped.append((vname, "source not fully tracked (propagation_complete is false)"))
            continue

        first_frame = first_labeled_frame(src_meta.get("point_prompts") or {})
        if first_frame is None:
            skipped.append((vname, "no point prompts to seed single-shot"))
            continue

        source_file = _resolve_source_file(src_video_dir)
        if source_file is None or not source_file.exists():
            skipped.append((vname, "source video file not found"))
            continue

        num_frames = int(src_meta.get("num_frames") or 0)
        if batch_size == "single":
            bs = max(1, num_frames - first_frame)
        elif batch_size == "inherit":
            bs = int(src_meta.get("anchor_batch_size") or 1000)
        else:
            bs = int(batch_size)

        new_vid = str(uuid.uuid4())[:8]
        new_video_dir = project_dir / "videos" / f"{new_vid}_{slugify(vname)}"
        (new_video_dir / "frames").mkdir(parents=True)
        (new_video_dir / "masks").mkdir()
        (new_video_dir / "bboxes").mkdir()

        # Symlink the source pixels (relative link keeps the project relocatable
        # alongside the source project).
        link = new_video_dir / source_file.name
        try:
            link.symlink_to(source_file)
        except OSError:
            link.symlink_to(source_file.resolve())

        vm = _single_shot_video_meta(
            src_meta, new_vid, new_video_dir, first_frame, source_file, bs
        )
        new_config["videos"][new_vid] = vm
        manifest["videos"].append(
            {
                "single_shot_video_id": new_vid,
                "single_shot_video_dir": str(new_video_dir),
                "source_video_id": src_vid,
                "source_video_dir": str(src_video_dir),
                "name": vname,
                "first_frame": int(first_frame),
                "num_frames": num_frames,
                "batch_size": bs,
            }
        )

    if not new_config["videos"]:
        import shutil

        shutil.rmtree(project_dir)
        reasons = "; ".join(f"{n}: {r}" for n, r in skipped) or "no eligible videos"
        raise RuntimeError(f"No videos emitted from {src_name!r} ({reasons})")

    (project_dir / "config.json").write_text(json.dumps(new_config, indent=2))
    (project_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    print(f"[build] source project : {src_name}  ({source_dir})")
    print(f"[build] emitted project: {proj_name}")
    print(f"[build]   dir          : {project_dir}")
    print(f"[build]   videos       : {len(new_config['videos'])} (batch_size={batch_size})")
    for v in manifest["videos"]:
        print(
            f"[build]     - {v['name']}: first_frame={v['first_frame']}, "
            f"frames={v['num_frames']}, batch={v['batch_size']}"
        )
    for n, r in skipped:
        print(f"[build]   SKIP {n}: {r}")
    return project_dir


def load_manifest(project_dir: Path) -> dict[str, Any]:
    mp = project_dir / MANIFEST_NAME
    if not mp.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_NAME} not found under {project_dir}. "
            "Run the 'build' stage first."
        )
    return json.loads(mp.read_text())
