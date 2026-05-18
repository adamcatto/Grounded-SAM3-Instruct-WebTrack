"""Data containers for the clustering pipeline.

BehaviorDataset holds the (N, 33) feature matrix plus per-window metadata.
Supports save/load via npz + JSON sidecar.
"""

from __future__ import annotations

import json
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowMetadata:
        return cls(**d)


# ---------------------------------------------------------------------------
# Housing condition helpers
# ---------------------------------------------------------------------------

_ISOLATED_NAMES = {"headshave"}
_GROUP_NAMES = {"noshave", "backshave"}


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
