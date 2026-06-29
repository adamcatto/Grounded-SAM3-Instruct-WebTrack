"""Shared helpers for the anchor-frame memory ablation."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mask_store import VideoMaskStorage  # noqa: E402


def projects_base_dir() -> Path:
    for key in ("SAM3_TRACKING_PROJECTS_DIR", "SAM3_PROJECTS_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser()
    return Path.home() / ".sam3_zero_projects"


def slugify(name: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^\w\-.]", "_", name).strip("_.-")
    return slug[:max_len]


def load_config(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "config.json").read_text())


def write_config(project_dir: Path, cfg: dict[str, Any]) -> None:
    (project_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def resolve_project_dir(ref: str) -> Path:
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("empty project reference")
    p = Path(ref).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p.resolve()
    base = projects_base_dir()
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d.name == ref or d.name.startswith(ref + "-")) and (d / "config.json").is_file():
                return d.resolve()
    raise ValueError(f"project not found: {ref!r} (looked under {base})")


def find_video_dir(project_dir: Path, vid: str) -> Path | None:
    root = project_dir / "videos"
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.is_dir() and (d.name == vid or d.name.startswith(vid + "_")):
            return d
    return None


def source_file_for(video_dir: Path) -> Path | None:
    for cand in sorted(video_dir.glob("source.*")):
        if cand.is_file() or cand.is_symlink():
            return cand.resolve()
    return None


def mask_storage(video_dir: Path) -> VideoMaskStorage:
    return VideoMaskStorage(video_dir)


def object_names(cfg: dict[str, Any], vid: str) -> dict[str, str]:
    vm = (cfg.get("videos") or {}).get(vid) or {}
    return {str(k): str(v.get("name") or k) for k, v in (vm.get("objects") or {}).items()}


def backup_project(project_dir: Path, *, backup_parent: Path | None = None) -> Path:
    project_dir = project_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = (backup_parent or project_dir.parent / "_backups").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    dst = parent / f"{project_dir.name}.backup-{stamp}"
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(project_dir, dst, symlinks=True)
    return dst
