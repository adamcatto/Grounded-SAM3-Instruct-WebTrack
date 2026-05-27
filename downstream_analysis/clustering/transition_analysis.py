"""Cluster state-transition analysis from ethogram timelines.

Aggregates consecutive window-to-window transitions (A -> B) per stratification
group (e.g. condition group or interaction type) and produces row-normalized
transition probability matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def count_transitions(cluster_labels: list[int], n_clusters: int) -> np.ndarray:
    """Count consecutive transitions in one video ethogram.

    Returns an (n_clusters, n_clusters) count matrix where entry [a, b] is the
    number of times cluster *a* is immediately followed by cluster *b*.
    """
    counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    if len(cluster_labels) < 2:
        return counts

    for a, b in zip(cluster_labels[:-1], cluster_labels[1:]):
        if 0 <= a < n_clusters and 0 <= b < n_clusters:
            counts[a, b] += 1
    return counts


def normalize_transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalize a count matrix to transition probabilities P(B | A)."""
    mat = counts.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, mat / row_sums, 0.0)
    return probs


def _matrix_to_json(probs: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in probs]


def _counts_to_json(counts: np.ndarray) -> list[list[int]]:
    return [[int(x) for x in row] for row in counts]


def compute_transition_matrices_by_group(
    ethograms: dict[str, dict[str, Any]],
    n_clusters: int,
    *,
    group_field: str = "condition_group",
    group_labels: dict[str, str] | None = None,
    exclude_groups: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate transition counts and probabilities per stratification group.

    Parameters
    ----------
    ethograms
        Per-video ethogram dicts (must include ``cluster_labels`` and ``group_field``).
    n_clusters
        Number of behavioral clusters.
    group_field
        Ethogram key used to stratify (e.g. ``condition_group`` or ``interaction_type``).
    group_labels
        Optional human-readable labels keyed by group id.
    exclude_groups
        Group ids to skip (e.g. ``{"unknown"}``).

    Returns
    -------
    dict
        Keys are group ids. Each value contains counts, row-normalized matrix,
        and summary metadata suitable for JSON export.
    """
    if n_clusters <= 0 or not ethograms:
        return {}

    exclude = exclude_groups or set()
    grouped_counts: dict[str, np.ndarray] = {}
    n_videos: dict[str, int] = {}
    n_windows: dict[str, int] = {}

    for eth in ethograms.values():
        group = eth.get(group_field) or "unknown"
        if group in exclude:
            continue

        labels = eth.get("cluster_labels") or []
        if len(labels) < 2:
            continue

        if group not in grouped_counts:
            grouped_counts[group] = np.zeros((n_clusters, n_clusters), dtype=np.int64)
            n_videos[group] = 0
            n_windows[group] = 0

        grouped_counts[group] += count_transitions([int(x) for x in labels], n_clusters)
        n_videos[group] += 1
        n_windows[group] += len(labels)

    results: dict[str, dict[str, Any]] = {}
    for group, counts in grouped_counts.items():
        total_transitions = int(counts.sum())
        if total_transitions == 0:
            continue

        probs = normalize_transition_matrix(counts)
        results[group] = {
            "label": (group_labels or {}).get(group, group),
            "group_field": group_field,
            "n_videos": n_videos[group],
            "n_windows": n_windows[group],
            "n_transitions": total_transitions,
            "counts": _counts_to_json(counts),
            "matrix": _matrix_to_json(probs),
        }

    return results
