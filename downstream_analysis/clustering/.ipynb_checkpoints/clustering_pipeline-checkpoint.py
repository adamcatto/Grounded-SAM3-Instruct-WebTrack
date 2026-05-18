"""k-NN graph construction, Leiden clustering, UMAP embedding, and main CLI.

Follows the same structural pattern as the locomotion LocomotionAnalysisPipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths as pathutil
from ..logging_setup import configure_logging
from .config import ClusteringConfig
from .dataset import BehaviorDataset, WindowMetadata
from .feature_extraction import extract_all_features
from .sequence_features import SEQUENCE_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import igraph as ig
    import leidenalg
    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClusteringResult:
    features_normalized: np.ndarray     # (N, 33)
    cluster_labels: np.ndarray          # (N,) int
    n_clusters: int
    embedding_2d: np.ndarray            # (N, 2)
    metadata: list[WindowMetadata]
    feature_names: list[str]
    method: str                         # "leiden" or "spectral"
    embedding_method: str               # "umap" or "tsne"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_features(
    features: np.ndarray,
    method: str = "zscore",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize feature matrix, handling NaN columns gracefully.

    Returns (normalized, stats_dict).
    """
    X = features.copy()
    stats: dict[str, Any] = {"method": method}

    # Replace any remaining NaN with column median
    for col in range(X.shape[1]):
        mask = np.isfinite(X[:, col])
        if not np.all(mask):
            median_val = float(np.nanmedian(X[:, col])) if np.any(mask) else 0.0
            X[~mask, col] = median_val

    if method == "zscore":
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds < 1e-12] = 1.0  # avoid division by zero for constant features
        X = (X - means) / stds
        stats["means"] = means.tolist()
        stats["stds"] = stds.tolist()
    elif method == "robust":
        medians = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-12] = 1.0
        X = (X - medians) / iqr
        stats["medians"] = medians.tolist()
        stats["iqr"] = iqr.tolist()
    elif method == "minmax":
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-12] = 1.0
        X = (X - mins) / ranges
        stats["mins"] = mins.tolist()
        stats["ranges"] = ranges.tolist()

    # Replace any inf with 0
    X[~np.isfinite(X)] = 0.0
    return X, stats


# ---------------------------------------------------------------------------
# k-NN graph + clustering
# ---------------------------------------------------------------------------

def _build_knn_graph(X: np.ndarray, n_neighbors: int) -> Any:
    """Build a k-NN graph and return as igraph Graph or scipy sparse matrix."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    n = X.shape[0]

    if _HAS_LEIDEN:
        # Build igraph weighted graph
        edges = []
        weights = []
        max_d = distances[:, -1].max() if distances.size > 0 else 1.0
        max_d = max(max_d, 1e-12)
        for i in range(n):
            for j_idx in range(n_neighbors):
                j = indices[i, j_idx]
                if i != j:
                    edges.append((i, j))
                    weights.append(1.0 - distances[i, j_idx] / max_d)
        g = ig.Graph(n=n, edges=edges, directed=True)
        g.es["weight"] = weights
        # Convert to undirected (mutual k-NN)
        g = g.as_undirected(mode="collapse", combine_edges={"weight": "max"})
        return g

    # Fallback: return sparse affinity matrix for sklearn
    from scipy.sparse import lil_matrix
    W = lil_matrix((n, n), dtype=np.float64)
    max_d = distances[:, -1].max() if distances.size > 0 else 1.0
    max_d = max(max_d, 1e-12)
    for i in range(n):
        for j_idx in range(n_neighbors):
            j = indices[i, j_idx]
            if i != j:
                w = 1.0 - distances[i, j_idx] / max_d
                W[i, j] = max(W[i, j], w)
                W[j, i] = max(W[j, i], w)
    return W.tocsr()


def _cluster_leiden(graph: Any, resolution: float) -> np.ndarray:
    """Run Leiden community detection."""
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    return np.array(partition.membership, dtype=np.int32)


def _cluster_spectral(affinity: Any, n_clusters: int = 20) -> np.ndarray:
    """Fallback: spectral clustering via sklearn."""
    from sklearn.cluster import SpectralClustering

    # Estimate n_clusters from data if not specified
    sc = SpectralClustering(
        n_clusters=min(n_clusters, affinity.shape[0]),
        affinity="precomputed",
        random_state=42,
        n_init=5,
    )
    return sc.fit_predict(affinity).astype(np.int32)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_umap(X: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """UMAP 2D embedding."""
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=42,
    )
    return reducer.fit_transform(X).astype(np.float64)


def _embed_tsne(X: np.ndarray) -> np.ndarray:
    """Fallback: t-SNE via sklearn."""
    from sklearn.manifold import TSNE

    perp = min(30, max(5, X.shape[0] // 5))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X).astype(np.float64)


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def run_clustering(
    dataset: BehaviorDataset,
    cfg: ClusteringConfig | None = None,
) -> ClusteringResult:
    """Normalize features, build k-NN graph, cluster, embed.

    Parameters
    ----------
    dataset : BehaviorDataset with (N, 33) features
    cfg : clustering configuration

    Returns
    -------
    ClusteringResult with labels, embedding, normalized features
    """
    if cfg is None:
        cfg = ClusteringConfig()

    n_samples = len(dataset)
    logger.info("Clustering %d windows with %d features.", n_samples, dataset.n_features)

    if n_samples == 0:
        return ClusteringResult(
            features_normalized=dataset.features,
            cluster_labels=np.array([], dtype=np.int32),
            n_clusters=0,
            embedding_2d=np.empty((0, 2), dtype=np.float64),
            metadata=dataset.metadata,
            feature_names=dataset.feature_names,
            method="none",
            embedding_method="none",
        )

    # 1. Normalize
    X_norm, norm_stats = _normalize_features(dataset.features, cfg.normalize_method)
    logger.info("Normalized features (%s): shape %s", cfg.normalize_method, X_norm.shape)

    # 2. Build k-NN graph
    k = min(cfg.n_neighbors, n_samples - 1)
    logger.info("Building k-NN graph (k=%d)...", k)
    graph = _build_knn_graph(X_norm, k)

    # 3. Cluster
    if _HAS_LEIDEN:
        logger.info("Running Leiden clustering (resolution=%.2f)...", cfg.leiden_resolution)
        labels = _cluster_leiden(graph, cfg.leiden_resolution)
        method = "leiden"
    else:
        logger.info("leidenalg not installed, falling back to spectral clustering.")
        labels = _cluster_spectral(graph)
        method = "spectral"

    n_clusters = int(labels.max() + 1) if len(labels) > 0 else 0
    logger.info("Found %d clusters via %s.", n_clusters, method)

    # Log cluster sizes
    for c in range(n_clusters):
        size = int(np.sum(labels == c))
        logger.debug("  Cluster %d: %d windows (%.1f%%)", c, size, 100.0 * size / n_samples)

    # 4. Embed
    if _HAS_UMAP:
        logger.info("Computing UMAP embedding...")
        embedding = _embed_umap(X_norm, cfg)
        emb_method = "umap"
    else:
        logger.info("umap-learn not installed, falling back to t-SNE.")
        embedding = _embed_tsne(X_norm)
        emb_method = "tsne"

    logger.info("Embedding complete (%s): shape %s", emb_method, embedding.shape)

    return ClusteringResult(
        features_normalized=X_norm,
        cluster_labels=labels,
        n_clusters=n_clusters,
        embedding_2d=embedding,
        metadata=dataset.metadata,
        feature_names=dataset.feature_names,
        method=method,
        embedding_method=emb_method,
    )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class ClusteringPipeline:
    """End-to-end clustering pipeline for a project folder.

    Usage::

        pipe = ClusteringPipeline(project_dir)
        result = pipe.run()
    """

    def __init__(
        self,
        project_dir: Path,
        *,
        cfg: ClusteringConfig | None = None,
        show_progress: bool = True,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.cfg = cfg or ClusteringConfig()
        self.show_progress = show_progress

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "analysis_of_tracking_data" / "clustering"

    def run(self) -> ClusteringResult:
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting clustering pipeline | dir=%s | window=%d stride=%d",
            self.project_dir, self.cfg.window_size, self.cfg.stride,
        )

        # 1. Feature extraction
        dataset = extract_all_features(
            self.project_dir, self.cfg, progress=self.show_progress,
        )

        if len(dataset) == 0:
            logger.warning("No features extracted. Writing empty summary.")
            self._write_summary(out, dataset, None)
            return ClusteringResult(
                features_normalized=dataset.features,
                cluster_labels=np.array([], dtype=np.int32),
                n_clusters=0,
                embedding_2d=np.empty((0, 2), dtype=np.float64),
                metadata=dataset.metadata,
                feature_names=dataset.feature_names,
                method="none",
                embedding_method="none",
            )

        # 2. Clustering
        result = run_clustering(dataset, self.cfg)

        # 3. Downstream comparison
        from .comparison import compare_housing_conditions
        comparison = compare_housing_conditions(result)

        # 4. Plots
        from .plots import generate_all_plots
        plots_dir = out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(result, comparison, plots_dir)

        # 5. Save results
        results_dir = out / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        self._save_results(results_dir, result, comparison)
        self._write_summary(out, dataset, result, comparison)

        logger.info("Clustering pipeline complete. Outputs in %s", out)
        return result

    def _save_results(
        self,
        results_dir: Path,
        result: ClusteringResult,
        comparison: dict[str, Any],
    ) -> None:
        # Cluster assignments
        assignments = []
        for i, m in enumerate(result.metadata):
            assignments.append({
                "window_index": i,
                "cluster": int(result.cluster_labels[i]),
                "video_id": m.video_id,
                "start_frame": m.start_frame,
                "object_a_name": m.object_a_name,
                "object_b_name": m.object_b_name,
                "interaction_type": m.interaction_type,
            })
        (results_dir / "cluster_assignments.json").write_text(
            json.dumps(assignments, indent=2)
        )

        # Cluster summary (centroids + sizes)
        cluster_summary = {}
        for c in range(result.n_clusters):
            mask = result.cluster_labels == c
            centroid = result.features_normalized[mask].mean(axis=0)
            cluster_summary[str(c)] = {
                "size": int(np.sum(mask)),
                "fraction": float(np.sum(mask)) / len(result.cluster_labels),
                "feature_centroid": {
                    name: float(centroid[i])
                    for i, name in enumerate(result.feature_names)
                },
            }
        (results_dir / "cluster_summary.json").write_text(
            json.dumps(cluster_summary, indent=2)
        )

        # Housing comparison
        (results_dir / "housing_comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str)
        )

        # Embedding coordinates
        np.savez_compressed(
            results_dir / "embedding.npz",
            embedding=result.embedding_2d,
            labels=result.cluster_labels,
        )

        logger.info("Saved results to %s", results_dir)

    def _write_summary(
        self,
        out_dir: Path,
        dataset: BehaviorDataset,
        result: ClusteringResult | None,
        comparison: dict[str, Any] | None = None,
    ) -> None:
        summary: dict[str, Any] = {
            "project_dir": str(self.project_dir),
            "config": {
                "window_size": self.cfg.window_size,
                "stride": self.cfg.stride,
                "n_neighbors": self.cfg.n_neighbors,
                "leiden_resolution": self.cfg.leiden_resolution,
                "normalize_method": self.cfg.normalize_method,
            },
            "n_windows": len(dataset),
            "n_features": dataset.n_features,
        }
        if result is not None:
            summary["n_clusters"] = result.n_clusters
            summary["clustering_method"] = result.method
            summary["embedding_method"] = result.embedding_method

            # Cluster size distribution
            sizes = []
            for c in range(result.n_clusters):
                sizes.append(int(np.sum(result.cluster_labels == c)))
            summary["cluster_sizes"] = sizes

        if comparison is not None:
            summary["housing_comparison_summary"] = {
                k: v for k, v in comparison.items()
                if k in ("overall_enrichment", "interaction_type_distribution")
            }

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote summary.json to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Behavioral sequence clustering for SAM3 Web Tracker projects.",
    )
    p.add_argument("--project-dir", type=Path, help="Folder containing config.json")
    p.add_argument("--project-id", type=str, help="Short project uuid")
    p.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    p.add_argument("--n-neighbors", type=int, default=30, help="k-NN neighbors (default: 30)")
    p.add_argument("--window-size", type=int, default=90, help="Window size in frames (default: 90)")
    p.add_argument("--stride", type=int, default=30, help="Stride in frames (default: 30)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("-q", "--quiet", action="store_true", help="WARNING only")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm bars")
    args = p.parse_args(argv)

    if args.verbose and args.quiet:
        print("Cannot use both --verbose and --quiet", file=sys.stderr)
        return 2

    configure_logging(verbose=args.verbose, quiet=args.quiet)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    show_progress = not args.no_progress and not args.quiet

    if args.project_dir:
        proj = Path(args.project_dir).resolve()
        if not (proj / "config.json").is_file():
            print("No config.json in", proj, file=sys.stderr)
            return 1
    elif args.project_id:
        found = pathutil.find_project_dir(args.project_id.strip())
        if found is None:
            print("Project id not found:", args.project_id, file=sys.stderr)
            return 1
        proj = found
    else:
        p.print_help()
        return 2

    cfg = ClusteringConfig(
        window_size=args.window_size,
        stride=args.stride,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.resolution,
    )

    pipe = ClusteringPipeline(proj, cfg=cfg, show_progress=show_progress)
    result = pipe.run()

    if not args.quiet:
        print(f"Done: {result.n_clusters} clusters from {len(result.metadata)} windows")
        print(f"Output: {pipe.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
