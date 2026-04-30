"""Sliding-window centroid path length (locomotion) per tracked object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocomotionConfig:
    chunk_size: int = 90
    stride: int = 30


def window_starts(track_length: int, chunk: int, stride: int) -> list[int]:
    """Inclusive window [s, s+chunk) fully inside [0, track_length)."""
    out: list[int] = []
    s = 0
    while s + chunk <= track_length:
        out.append(s)
        s += stride
    return out


def path_length_centroids(xy: np.ndarray) -> float:
    """
    Sum of Euclidean step lengths along rows of xy shape (T, 2).
    Rows with any NaN are treated as breaks: only consecutive finite pairs contribute.
    For a clean window with all finite, equal to sum of T-1 displacements.
    """
    if xy.shape[0] < 2:
        return 0.0
    total = 0.0
    for i in range(xy.shape[0] - 1):
        a, b = xy[i], xy[i + 1]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        total += float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return total


def locomotion_chunks_for_series(
    xy_full: np.ndarray,
    *,
    cfg: LocomotionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      starts: indices into timeline (0..len-1) for each chunk
      values: locomotion sum (pixels) per chunk; NaN if any frame in window missing centroid
    """
    t_len = xy_full.shape[0]
    starts = window_starts(t_len, cfg.chunk_size, cfg.stride)
    vals = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        seg = xy_full[s : s + cfg.chunk_size]
        if not np.all(np.isfinite(seg)):
            vals[i] = np.nan
            continue
        vals[i] = path_length_centroids(seg)
    return np.array(starts, dtype=np.int32), vals


def normalize_by_diagonal(values: np.ndarray, width: int, height: int) -> np.ndarray:
    diag = float(np.hypot(max(width, 1), max(height, 1)))
    if diag <= 0:
        return values
    return values / diag
