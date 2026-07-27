"""Single-animal analysis mode.

Splits each pair-level window into TWO single-animal samples.  Each sample has:
  - 12 per-object features (focal animal, with generic names like ``speed_mean``)
  - 8 interaction features (shared, excluding ``overlap_frac_mean``)
  - Total: 20 features per sample

The focal animal's metadata is placed in the ``mouse_a_*`` fields; the
``focal_side`` field records which side ("a" or "b") the sample was derived
from.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from .dataset import BehaviorDataset, WindowMetadata
from .sequence_features import (
    SINGLE_ANIMAL_FEATURE_NAMES,
    SINGLE_ANIMAL_PER_OBJECT_NAMES,
    interaction_column_indices,
    per_object_column_indices,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata swap helper
# ---------------------------------------------------------------------------

def _swap_metadata_for_b_focal(m: WindowMetadata) -> WindowMetadata:
    """Create metadata copy where animal b becomes the focal (mouse_a_* fields)."""
    swapped = copy.copy(m)
    swapped.focal_side = "b"
    # Swap a <-> b identity fields
    swapped.mouse_a_id = m.mouse_b_id
    swapped.mouse_a_role = m.mouse_b_role
    swapped.mouse_a_housing = m.mouse_b_housing
    swapped.mouse_b_id = m.mouse_a_id
    swapped.mouse_b_role = m.mouse_a_role
    swapped.mouse_b_housing = m.mouse_a_housing
    # Also swap object names/keys for clarity
    swapped.object_a_key = m.object_b_key
    swapped.object_b_key = m.object_a_key
    swapped.object_a_name = m.object_b_name
    swapped.object_b_name = m.object_a_name
    return swapped


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def split_to_single_animal(dataset: BehaviorDataset) -> BehaviorDataset:
    """Split a pair-level dataset into single-animal samples.

    Each window ``(N, 33)`` becomes TWO rows ``(each 20 features)``:

    - **Sample "a"**: focal = animal a's 12 per-object features + 8 interaction
    - **Sample "b"**: focal = animal b's 12 per-object features + 8 interaction

    Metadata for "b" samples is swapped so the focal animal always appears in
    ``mouse_a_*`` fields.

    Returns
    -------
    BehaviorDataset
        ``(2N, 20)`` dataset with ``feature_names = SINGLE_ANIMAL_FEATURE_NAMES``.
    """
    n = len(dataset)
    if n == 0:
        return BehaviorDataset(
            features=np.empty((0, len(SINGLE_ANIMAL_FEATURE_NAMES)), dtype=np.float64),
            metadata=[],
            feature_names=list(SINGLE_ANIMAL_FEATURE_NAMES),
        )

    feat_names = list(dataset.feature_names)

    # Column indices in the pair-level feature vector
    a_cols = per_object_column_indices("a", feat_names)  # 12
    b_cols = per_object_column_indices("b", feat_names)  # 12
    i_cols = interaction_column_indices(feat_names, exclude_analysis_excluded=True)  # 8

    n_single = len(a_cols) + len(i_cols)  # 20
    assert n_single == len(SINGLE_ANIMAL_FEATURE_NAMES), (
        f"Expected {len(SINGLE_ANIMAL_FEATURE_NAMES)} single-animal features, "
        f"got {n_single} ({len(a_cols)} per-object + {len(i_cols)} interaction)"
    )

    # Build output arrays
    out_features = np.empty((2 * n, n_single), dtype=np.float64)
    out_metadata: list[WindowMetadata] = []

    for i in range(n):
        row = dataset.features[i]
        m = dataset.metadata[i]

        # Sample for focal animal "a"
        out_features[2 * i] = np.concatenate([row[a_cols], row[i_cols]])
        m_a = copy.copy(m)
        m_a.focal_side = "a"
        out_metadata.append(m_a)

        # Sample for focal animal "b"
        out_features[2 * i + 1] = np.concatenate([row[b_cols], row[i_cols]])
        out_metadata.append(_swap_metadata_for_b_focal(m))

    logger.info(
        "Split %d pair windows -> %d single-animal samples (%d features each)",
        n, 2 * n, n_single,
    )

    return BehaviorDataset(
        features=out_features,
        metadata=out_metadata,
        feature_names=list(SINGLE_ANIMAL_FEATURE_NAMES),
    )


# ---------------------------------------------------------------------------
# Native single-animal restriction (no pair to split)
# ---------------------------------------------------------------------------

def restrict_to_focal_animal(dataset: BehaviorDataset) -> BehaviorDataset:
    """Restrict a pair-schema dataset to the focal animal's 12 per-object features.

    Unlike :func:`split_to_single_animal` (which turns a genuine *social pair*
    into two focal samples), this is for projects where each video tracks a
    **single** animal from the start.  Feature extraction still emits the full
    33-dim pair vector, but the ``b_*`` and interaction columns are all-NaN and
    would only dilute clustering.  This drops them, keeping only the 12 ``a_*``
    per-object features and renaming them to the generic per-object names.

    Each window's metadata is preserved but tagged ``focal_side = "a"`` so the
    downstream single-animal detection (used by the plots) fires.

    Returns
    -------
    BehaviorDataset
        ``(N, 12)`` dataset with ``feature_names = SINGLE_ANIMAL_PER_OBJECT_NAMES``.
    """
    n = len(dataset)
    if n == 0:
        return BehaviorDataset(
            features=np.empty((0, len(SINGLE_ANIMAL_PER_OBJECT_NAMES)), dtype=np.float64),
            metadata=[],
            feature_names=list(SINGLE_ANIMAL_PER_OBJECT_NAMES),
        )

    feat_names = list(dataset.feature_names)
    a_cols = per_object_column_indices("a", feat_names)  # 12

    assert len(a_cols) == len(SINGLE_ANIMAL_PER_OBJECT_NAMES), (
        f"Expected {len(SINGLE_ANIMAL_PER_OBJECT_NAMES)} per-object features, "
        f"got {len(a_cols)}"
    )

    out_features = dataset.features[:, a_cols].astype(np.float64, copy=True)

    out_metadata: list[WindowMetadata] = []
    for m in dataset.metadata:
        m_a = copy.copy(m)
        m_a.focal_side = "a"
        out_metadata.append(m_a)

    logger.info(
        "Restricted %d windows to single-animal focal features (%d -> %d features)",
        n, len(feat_names), len(a_cols),
    )

    return BehaviorDataset(
        features=out_features,
        metadata=out_metadata,
        feature_names=list(SINGLE_ANIMAL_PER_OBJECT_NAMES),
    )
