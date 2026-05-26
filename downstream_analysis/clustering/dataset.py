"""Data containers for the clustering pipeline.

BehaviorDataset holds the (N, 33) feature matrix plus per-window metadata.
Supports save/load via npz + JSON sidecar.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .sequence_features import SEQUENCE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Per-window metadata
# ---------------------------------------------------------------------------

@dataclass
class WindowMetadata:
    video_id: str
    video_name: str
    object_a_key: str       # npz / mask dict key, e.g. "1"
    object_b_key: str
    object_a_name: str      # display name, e.g. "HeadShave"
    object_b_name: str
    start_frame: int        # absolute frame index of window start
    window_size: int
    camera_view: str        # extracted from video folder name (e.g. "3A")
    interaction_type: str   # "isolated+group", "group+group", etc.

    # Multi-project identity fields (optional, backward-compatible defaults)
    project_id: str = ""
    experiment_name: str = ""       # "hab", "test_day", "sh_intruder"
    experiment_number: int = 0      # 1, 2, 3
    session: str = ""               # "hab2a", "test1d", "HCGH_ISH1"
    mouse_a_id: str = ""            # "GH1", "SH2", "novel_GH_intruder"
    mouse_b_id: str = ""
    mouse_a_role: str = ""          # "resident", "intruder", "littermate"
    mouse_b_role: str = ""
    mouse_a_housing: str = ""       # "SH" or "GH"
    mouse_b_housing: str = ""
    batch_id: str = ""              # "{experiment_name}_{camera_view}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowMetadata:
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Housing condition helpers
# ---------------------------------------------------------------------------

_ISOLATED_NAMES = {"headshave", "headshaved"}
_GROUP_NAMES = {"noshave", "backshave", "backshaved"}


def housing_condition(object_name: str) -> str:
    """Classify an object name as 'isolated' or 'group-housed'."""
    low = object_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if low in _ISOLATED_NAMES:
        return "isolated"
    if low in _GROUP_NAMES:
        return "group-housed"
    return "unknown"


def interaction_type_from_names(name_a: str, name_b: str) -> str:
    """Derive interaction type from two object display names."""
    cond_a = housing_condition(name_a)
    cond_b = housing_condition(name_b)
    conditions = sorted([cond_a, cond_b])
    return "+".join(conditions)


def session_from_video_name(video_name: str) -> str:
    """Extract session from video name.

    e.g. '1A_video_test1a_20260130_090255.mp4' → 'test1a'
         '1A_video_HCGH_ISH1_20260131_084152.mp4' → 'HCGH_ISH1'
    """
    m = re.match(r"^(\w+)_video_(.+)_(\d{8})_(\d{6})\.mp4$", video_name)
    return m.group(2) if m else "unknown"


def camera_view_from_folder(folder_name: str) -> str:
    """Extract camera view identifier from video folder name.

    Folder name format: {hash}_{camera_view}_{rest...}
    e.g. '00d48b80_3A_video_test1d_20260130_121202.mp4' -> '3A'
    """
    parts = folder_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class BehaviorDataset:
    features: np.ndarray                        # (N, 33) float64
    metadata: list[WindowMetadata]              # length N
    feature_names: list[str] = field(
        default_factory=lambda: list(SEQUENCE_FEATURE_NAMES)
    )

    def __len__(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1] if self.features.ndim == 2 else 0

    def save(self, path: Path) -> None:
        """Save to npz (features) + json sidecar (metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=self.features)
        meta_path = path.with_suffix(".json")
        payload = {
            "feature_names": self.feature_names,
            "metadata": [m.to_dict() for m in self.metadata],
        }
        meta_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> BehaviorDataset:
        """Load from npz + json sidecar."""
        path = Path(path)
        data = np.load(path)
        features = data["features"]
        meta_path = path.with_suffix(".json")
        payload = json.loads(meta_path.read_text())
        metadata = [WindowMetadata.from_dict(d) for d in payload["metadata"]]
        feature_names = payload.get("feature_names", list(SEQUENCE_FEATURE_NAMES))
        return cls(features=features, metadata=metadata, feature_names=feature_names)
