"""
Load configs/env.yaml from the repo root (optional).

Path resolution:
  • os.environ SAM3_ENV_YAML if set (absolute or relative cwd)
  • else <repo>/configs/env.yaml if that file exists
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REL = Path("configs") / "env.yaml"


def repo_root() -> Path:
    return REPO_ROOT


def resolve_env_yaml_path() -> Path | None:
    override = (os.environ.get("SAM3_ENV_YAML") or "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(REPO_ROOT / _DEFAULT_REL)
    for p in candidates:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return None


def load_env_yaml_raw() -> dict[str, Any]:
    path = resolve_env_yaml_path()
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to load env.yaml. Install with: pip install pyyaml "
            "(or install backend/requirements.txt in your env)."
        ) from e

    blob = yaml.safe_load(path.read_text(encoding="utf-8"))
    if blob is None:
        return {}
    if not isinstance(blob, dict):
        return {}
    return blob


def parallel_tracking_section(cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg.get("parallel_tracking")
    return dict(s) if isinstance(s, dict) else {}


def lsf_parallel_tracking_section(cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg.get("lsf_parallel_tracking")
    return dict(s) if isinstance(s, dict) else {}
