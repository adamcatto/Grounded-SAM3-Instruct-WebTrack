"""Locate on-disk project folders (same rules as backend `project_manager`)."""

from __future__ import annotations

import os
import re
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def projects_base_dir() -> Path:
    for key in ("SAM3_TRACKING_PROJECTS_DIR", "SAM3_PROJECTS_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser()
    return Path.home() / ".sam3_zero_projects"


def find_project_dir(project_id: str) -> Path | None:
    """
    Resolve directory for short project uuid (config["id"]),
    matching dirs named `{id}` or `{id}-{slug}`.
    """
    base = projects_base_dir()
    if not base.is_dir():
        return None
    pid = project_id.strip()
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name == pid or d.name.startswith(pid + "-"):
            if (d / "config.json").is_file():
                return d
    return None


def safe_filename_fragment(name: str, max_len: int = 64) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return (s[:max_len] if s else "unnamed")

