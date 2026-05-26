"""Experiment identity registry: maps videos to mouse IDs, roles, and housing.

Loads experiment_registry.csv and provides lookup functions to resolve
which mouse is which in each video based on (experiment, session, box,
shave_pattern).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).with_name("experiment_registry.csv")

# Project directory slug → experiment_name mapping
_PROJECT_SLUG_MAP = {
    "185b3df6": "hab",
    "3c9bddf2": "test_day",
    "a3d11289": "sh_intruder",
    "37254501": "locomotion",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MouseIdentity:
    mouse_id: str       # "GH1", "SH2", "novel_GH_intruder", "GH1_littermate"
    housing: str        # "SH" or "GH"
    role: str           # "resident", "intruder", "littermate"
    shave_pattern: str  # "backshave", "headshave", "noshave"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_shave_name(name: str) -> str:
    """Normalize object display names to canonical shave patterns.

    'BackShave', 'BackShaved', 'Back Shave' → 'backshave'
    'HeadShave', 'HeadShaved'               → 'headshave'
    'NoShave', 'No Shave'                   → 'noshave'
    """
    s = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    # Strip trailing 'd' from "backshaved" / "headshaved"
    if s.endswith("shaved"):
        s = s[:-1]  # "backshaved" → "backshave"
    return s


def parse_video_name(video_name: str) -> tuple[str, str] | None:
    """Extract (camera_view, session) from a video name.

    Video name format: {camera_view}_video_{session}_{YYYYMMDD}_{HHMMSS}.mp4
    Session may contain underscores (e.g. 'HCGH_ISH1').

    Returns (camera_view, session) or None if parsing fails.
    """
    m = re.match(r"^(\w+)_video_(.+)_(\d{8})_(\d{6})\.mp4$", video_name)
    if m:
        return m.group(1), m.group(2)
    return None


def box_from_camera_view(camera_view: str) -> int:
    """Extract box number from camera view: '1A' → 1, '3B' → 3."""
    return int(camera_view[0])


def experiment_name_from_project_dir(project_dir: Path) -> str:
    """Derive experiment_name from the project directory name.

    e.g. '3c9bddf2-Home-Cage-Interactions-0126-test-day' → 'test_day'
    """
    slug = project_dir.name.split("-")[0]
    return _PROJECT_SLUG_MAP.get(slug, "")


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(csv_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the experiment registry CSV into a list of row dicts."""
    path = Path(csv_path) if csv_path else _CSV_PATH
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["experiment"] = int(row["experiment"])
            row["box"] = int(row["box"])
            rows.append(row)
    return rows


def _build_lookup(
    registry: list[dict[str, Any]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    """Build a fast lookup: (experiment_name, session, box, shave_pattern) → row."""
    lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in registry:
        key = (row["experiment_name"], row["session"], row["box"], row["shave_pattern"])
        lookup[key] = row
    return lookup


# Module-level cache
_registry_cache: list[dict[str, Any]] | None = None
_lookup_cache: dict[tuple[str, str, int, str], dict[str, Any]] | None = None


def _get_registry_and_lookup(
    csv_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, str], dict[str, Any]]]:
    global _registry_cache, _lookup_cache
    if _registry_cache is None or csv_path is not None:
        _registry_cache = load_registry(csv_path)
        _lookup_cache = _build_lookup(_registry_cache)
    return _registry_cache, _lookup_cache


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------

def resolve_identities(
    experiment_name: str,
    session: str,
    box: int,
    obj_a_name: str,
    obj_b_name: str,
    *,
    csv_path: Path | str | None = None,
) -> tuple[MouseIdentity, MouseIdentity] | None:
    """Resolve mouse identities for a video's two objects.

    Parameters
    ----------
    experiment_name : 'hab', 'test_day', 'sh_intruder'
    session : e.g. 'test1a', 'HCGH_ISH1'
    box : 1-4
    obj_a_name : display name of object A (e.g. 'HeadShave')
    obj_b_name : display name of object B (e.g. 'NoShave')

    Returns
    -------
    (identity_a, identity_b) matching obj_a and obj_b respectively,
    or None if the lookup fails.
    """
    _, lookup = _get_registry_and_lookup(csv_path)

    shave_a = normalize_shave_name(obj_a_name)
    shave_b = normalize_shave_name(obj_b_name)

    row_a = lookup.get((experiment_name, session, box, shave_a))
    row_b = lookup.get((experiment_name, session, box, shave_b))

    if row_a is None or row_b is None:
        logger.warning(
            "Registry lookup failed: exp=%s session=%s box=%d shave=(%s, %s)",
            experiment_name, session, box, shave_a, shave_b,
        )
        return None

    id_a = MouseIdentity(
        mouse_id=row_a["mouse_id"],
        housing=row_a["housing"],
        role=row_a["role"],
        shave_pattern=shave_a,
    )
    id_b = MouseIdentity(
        mouse_id=row_b["mouse_id"],
        housing=row_b["housing"],
        role=row_b["role"],
        shave_pattern=shave_b,
    )
    return id_a, id_b


def resolve_single_identity(
    experiment_name: str,
    session: str,
    box: int,
    obj_name: str,
    *,
    csv_path: Path | str | None = None,
) -> MouseIdentity | None:
    """Resolve identity for a single-mouse video (e.g. locomotion)."""
    _, lookup = _get_registry_and_lookup(csv_path)
    shave = normalize_shave_name(obj_name)
    row = lookup.get((experiment_name, session, box, shave))
    if row is None:
        return None
    return MouseIdentity(
        mouse_id=row["mouse_id"],
        housing=row["housing"],
        role=row["role"],
        shave_pattern=shave,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_registry(
    project_dirs: list[Path],
    *,
    csv_path: Path | str | None = None,
) -> list[str]:
    """Validate that every 2-object video in the given projects can be resolved.

    Returns a list of error messages (empty = all good).
    """
    errors: list[str] = []
    registry, _ = _get_registry_and_lookup(csv_path)

    for project_dir in project_dirs:
        project_dir = Path(project_dir).resolve()
        exp_name = experiment_name_from_project_dir(project_dir)
        if not exp_name:
            errors.append(f"Unknown project slug: {project_dir.name}")
            continue

        config_path = project_dir / "config.json"
        if not config_path.is_file():
            errors.append(f"No config.json in {project_dir}")
            continue

        config = json.loads(config_path.read_text())
        videos = config.get("videos", {})

        n_ok = 0
        n_skip = 0
        for vid, v in videos.items():
            objects = v.get("objects", {})
            if len(objects) < 2:
                n_skip += 1
                continue

            vname = v.get("name", "")
            parsed = parse_video_name(vname)
            if parsed is None:
                errors.append(f"  Cannot parse video name: {vname}")
                continue

            camera_view, session = parsed
            box = box_from_camera_view(camera_view)

            obj_names = [o.get("name", "") for o in objects.values()]
            result = resolve_identities(
                exp_name, session, box, obj_names[0], obj_names[1],
                csv_path=csv_path,
            )
            if result is None:
                errors.append(
                    f"  UNRESOLVED: {vname} | exp={exp_name} session={session} "
                    f"box={box} objects={obj_names}"
                )
            else:
                n_ok += 1

        logger.info(
            "Validated %s: %d resolved, %d skipped (single-object), %d errors",
            project_dir.name, n_ok, n_skip,
            sum(1 for e in errors if project_dir.name in e),
        )

    return errors
