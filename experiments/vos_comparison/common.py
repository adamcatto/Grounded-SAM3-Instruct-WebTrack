"""Shared helpers: backend imports, project/video resolution, mask access."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Backend modules (mask_store reads masks.sqlite with the same codec the app uses).
from mask_store import VideoMaskStorage  # noqa: E402

# Prefix for emitted single-shot projects.
SINGLE_SHOT_PREFIX = "single_shot_vos_"
# Filename of the manifest written into each emitted project linking it to its GT.
MANIFEST_NAME = "vos_comparison_manifest.json"


def projects_base_dir() -> Path:
    """Filesystem root for project folders (matches backend project_manager)."""
    for key in ("SAM3_TRACKING_PROJECTS_DIR", "SAM3_PROJECTS_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser()
    return Path.home() / ".sam3_zero_projects"


def slugify(name: str, max_len: int = 64) -> str:
    """Filesystem-safe slug (mirrors ProjectManager._slugify)."""
    slug = re.sub(r"[^\w\-.]", "_", name).strip("_.-")
    return slug[:max_len]


def resolve_project_dir(ref: str) -> Path:
    """Resolve a project by filesystem path, folder name, or short id."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("Empty project reference")

    p = Path(ref).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p.resolve()

    base = projects_base_dir()
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if (d.name == ref or d.name.startswith(ref + "-")) and (d / "config.json").is_file():
                return d.resolve()

    raise ValueError(f"Project not found: {ref!r} (looked under {base})")


def load_config(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "config.json").read_text())


def find_video_dir(project_dir: Path, vid: str) -> Path | None:
    """Locate a video subdirectory by id prefix under <project>/videos/."""
    root = project_dir / "videos"
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.is_dir() and (d.name == vid or d.name.startswith(vid + "_")):
            return d
    return None


def mask_storage(video_dir: Path) -> VideoMaskStorage:
    return VideoMaskStorage(video_dir)


def first_labeled_frame(point_prompts: dict[str, Any]) -> int | None:
    """Smallest frame index that carries any point prompt across all objects."""
    frames: list[int] = []
    for obj_prompts in (point_prompts or {}).values():
        if not isinstance(obj_prompts, dict):
            continue
        for fk in obj_prompts:
            try:
                frames.append(int(fk))
            except (ValueError, TypeError):
                pass
    return min(frames) if frames else None


def npz_key_base_id(npz_key: str) -> str:
    """Map an instance mask key like '1_2' to its logical object id '1'."""
    return npz_key.split("_")[0] if "_" in npz_key else npz_key
