"""Configuration for the behavioral sequence clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClusteringConfig:
    # Sliding window (matches locomotion pipeline defaults)
    window_size: int = 90       # frames (3 seconds at 30fps)
    stride: int = 30            # frames (1 second)

    # Proximity thresholds (fraction of video diagonal)
    close_proximity_threshold: float = 0.05
    stationary_speed_threshold: float = 0.001  # fraction of diagonal per frame

    # Contour sampling for min-distance computation
    contour_sample_n: int = 200

    # Chase cross-correlation max lag (frames)
    chase_max_lag: int = 10

    # Feature group toggles
    use_position_features: bool = True  # absolute centroid xy
    use_invariant_features: bool = True  # relative / normalized features

    # Clustering
    n_neighbors: int = 30
    leiden_resolution: float = 1.0
    umap_n_components: int = 2
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # Normalization
    normalize_method: str = "zscore"  # "zscore" | "robust" | "minmax"
