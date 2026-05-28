#!/usr/bin/env python3
"""Generate the Excel report from already-saved clustering results.

Usage:
    conda run -n sam3 python -m downstream_analysis.clustering.generate_excel \
        --project-dir /path/to/project

    # Merged multi-project output directory:
    conda run -n sam3 python -m downstream_analysis.clustering.generate_excel \
        --project-dir /path/to/analysis_of_tracking_data/clustering
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate Excel report from saved results.")
    p.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Single project root, or multi-project clustering output directory",
    )
    args = p.parse_args(argv)

    project_dir = Path(args.project_dir).resolve()
    out_dir = project_dir / "analysis_of_tracking_data" / "clustering"
    if (project_dir / "results").is_dir() or (project_dir / "summary.json").is_file():
        out_dir = project_dir
    results_dir = out_dir / "results"

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from downstream_analysis.clustering.dataset import BehaviorDataset
    from downstream_analysis.clustering.clustering_pipeline import ClusteringResult
    from downstream_analysis.clustering.config import ClusteringConfig
    from downstream_analysis.clustering.report_outputs import (
        write_multi_project_excel_report,
        write_single_project_excel_report,
    )

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    pipeline = summary.get("pipeline", "single_project")

    # Load features + metadata
    if pipeline == "multi_project":
        ds_path = results_dir / "combined_dataset.npz"
        if not ds_path.is_file():
            print(f"ERROR: {ds_path} not found", file=sys.stderr)
            return 1
        dataset = BehaviorDataset.load(ds_path)
    else:
        seq_path = out_dir / "features" / "sequence_features.npz"
        if not seq_path.is_file():
            print(f"ERROR: {seq_path} not found", file=sys.stderr)
            return 1
        dataset = BehaviorDataset.load(seq_path)
    print(f"Loaded dataset: {len(dataset)} windows, {dataset.n_features} features")

    emb_path = results_dir / "embedding.npz"
    if not emb_path.is_file():
        print(f"ERROR: {emb_path} not found", file=sys.stderr)
        return 1
    emb_data = np.load(emb_path)
    embedding = emb_data["embedding"]
    labels = emb_data["labels"]
    print(f"Loaded embedding: {embedding.shape}, {int(labels.max()) + 1} clusters")

    from sklearn.preprocessing import StandardScaler
    features = dataset.features.copy()
    for col in range(features.shape[1]):
        col_mask = ~np.isfinite(features[:, col])
        if col_mask.any():
            median = np.nanmedian(features[:, col])
            features[col_mask, col] = median if np.isfinite(median) else 0.0
    features_norm = StandardScaler().fit_transform(features)

    result = ClusteringResult(
        features_normalized=features_norm,
        cluster_labels=labels,
        n_clusters=int(labels.max()) + 1,
        embedding_2d=embedding,
        metadata=dataset.metadata,
        feature_names=dataset.feature_names,
        method=summary.get("clustering_method", "leiden"),
        embedding_method=summary.get("embedding_method", "umap"),
    )

    cfg_data = summary.get("config", {})
    cfg = ClusteringConfig(
        window_size=cfg_data.get("window_size", 90),
        stride=cfg_data.get("stride", 30),
        n_neighbors=cfg_data.get("n_neighbors", 30),
        leiden_resolution=cfg_data.get("leiden_resolution", 1.0),
        normalize_method=cfg_data.get("normalize_method", "zscore"),
    )

    if pipeline == "multi_project":
        comp_path = results_dir / "condition_comparison.json"
        if not comp_path.is_file():
            print(f"ERROR: {comp_path} not found", file=sys.stderr)
            return 1
        comparison = json.loads(comp_path.read_text())
        project_dirs = [Path(d) for d in summary.get("project_dirs", [])]
        write_multi_project_excel_report(
            out_dir,
            result,
            comparison,
            cfg,
            project_dirs=project_dirs,
            batch_correction_method=summary.get("batch_correction_method", "combat"),
        )
    else:
        comp_path = results_dir / "housing_comparison.json"
        if not comp_path.is_file():
            print(f"ERROR: {comp_path} not found", file=sys.stderr)
            return 1
        comparison = json.loads(comp_path.read_text())
        write_single_project_excel_report(
            out_dir, result, comparison, cfg, project_dir.name,
        )

    xlsx_path = out_dir / "clustering_report.xlsx"
    if xlsx_path.is_file():
        size_mb = xlsx_path.stat().st_size / (1024 * 1024)
        print(f"Done! {xlsx_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
