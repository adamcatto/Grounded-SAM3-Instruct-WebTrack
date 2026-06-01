"""Multi-project orchestrator: merge, batch-correct, cluster, and compare.

Combines interaction data from multiple experiment projects, applies batch
correction at the (experiment_name, camera_view) level, runs clustering,
and produces condition-group comparisons and plots.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .batch_correction import combat_correct, zscore_per_batch
from .clustering_pipeline import ClusteringResult, run_clustering, _normalize_features
from .config import ClusteringConfig
from .dataset import BehaviorDataset, WindowMetadata
from .experiment_registry import (
    box_from_camera_view,
    experiment_name_from_project_dir,
    load_registry,
    parse_video_name,
    resolve_identities,
    resolve_single_identity,
)
from .feature_extraction import extract_all_features, _fmt_duration, _fmt_size
from .multi_project_comparison import compare_experiments
from .sequence_features import (
    ANALYSIS_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    analysis_feature_indices,
    analysis_feature_names,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: enrich metadata with registry identity info
# ---------------------------------------------------------------------------

def _enrich_metadata(
    metadata: list[WindowMetadata],
    experiment_name: str,
    experiment_number: int,
    *,
    csv_path: Path | str | None = None,
) -> int:
    """Enrich WindowMetadata in-place with identity fields from the registry.

    Returns the number of windows that could not be resolved.
    """
    n_unresolved = 0
    for m in metadata:
        m.experiment_name = experiment_name
        m.experiment_number = experiment_number
        m.batch_id = f"{experiment_name}_{m.camera_view}"

        parsed = parse_video_name(m.video_name)
        if parsed is None:
            n_unresolved += 1
            continue
        camera_view, session = parsed
        m.session = session
        box = box_from_camera_view(camera_view)

        if not m.object_b_name or m.object_b_name.lower() == "none":
            identity = resolve_single_identity(
                experiment_name, session, box, m.object_a_name,
                csv_path=csv_path,
            )
            if identity is None:
                n_unresolved += 1
                continue
            m.mouse_a_id = identity.mouse_id
            m.mouse_a_role = identity.role
            m.mouse_a_housing = identity.housing
        else:
            result = resolve_identities(
                experiment_name, session, box,
                m.object_a_name, m.object_b_name,
                csv_path=csv_path,
            )
            if result is None:
                n_unresolved += 1
                continue

            id_a, id_b = result
            m.mouse_a_id = id_a.mouse_id
            m.mouse_a_role = id_a.role
            m.mouse_a_housing = id_a.housing
            m.mouse_b_id = id_b.mouse_id
            m.mouse_b_role = id_b.role
            m.mouse_b_housing = id_b.housing

    return n_unresolved


# ---------------------------------------------------------------------------
# Experiment number mapping
# ---------------------------------------------------------------------------

_EXP_NUMBER = {
    "hab": 1,
    "test_day": 2,
    "sh_intruder": 3,
    "locomotion": 4,
}


# ---------------------------------------------------------------------------
# Multi-project pipeline
# ---------------------------------------------------------------------------

class MultiProjectPipeline:
    """Orchestrate cross-project interaction clustering.

    Usage::

        pipe = MultiProjectPipeline(
            project_dirs=[hab_dir, test_day_dir, sh_intruder_dir],
            output_dir=Path("multi_project_analysis"),
        )
        result = pipe.run()
    """

    def __init__(
        self,
        project_dirs: list[Path],
        output_dir: Path,
        *,
        registry_csv: Path | str | None = None,
        cfg: ClusteringConfig | None = None,
        batch_correction_method: str = "combat",
        skip_mask_verification: bool = False,
        locomotion_dir: Path | None = None,
    ):
        self.project_dirs = [Path(d).resolve() for d in project_dirs]
        self.output_dir = Path(output_dir).resolve()
        self.registry_csv = registry_csv
        self.cfg = cfg or ClusteringConfig()
        self.batch_correction_method = batch_correction_method
        self.skip_mask_verification = skip_mask_verification
        self.locomotion_dir = Path(locomotion_dir).resolve() if locomotion_dir else None

    def run(self) -> ClusteringResult:
        """Run the full multi-project pipeline.

        1. Extract features per project (reuses caches)
        2. Enrich metadata with identity info
        3. Concatenate + batch-correct
        4. Cluster + embed
        5. Compare conditions
        6. Generate plots
        7. Save results
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        overall_t0 = time.monotonic()

        logger.info("#" * 70)
        logger.info("# MULTI-PROJECT CLUSTERING PIPELINE")
        logger.info("#" * 70)
        logger.info("Projects: %d", len(self.project_dirs))
        for d in self.project_dirs:
            logger.info("  - %s (%s)", d.name, experiment_name_from_project_dir(d))
        logger.info("Output: %s", self.output_dir)
        logger.info("Batch correction: %s", self.batch_correction_method)
        logger.info("Config: window=%d stride=%d k=%d resolution=%.2f",
                     self.cfg.window_size, self.cfg.stride,
                     self.cfg.n_neighbors, self.cfg.leiden_resolution)
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 1: Feature extraction per project
        # ------------------------------------------------------------------
        logger.info("PHASE 1: Feature extraction")
        logger.info("-" * 40)
        t0 = time.monotonic()

        all_datasets: list[BehaviorDataset] = []
        for proj_dir in self.project_dirs:
            exp_name = experiment_name_from_project_dir(proj_dir)
            exp_num = _EXP_NUMBER.get(exp_name, 0)
            logger.info("Extracting features from %s (exp=%s)...", proj_dir.name, exp_name)

            dataset = extract_all_features(
                proj_dir, self.cfg,
                progress=True,
                skip_mask_verification=self.skip_mask_verification,
            )

            # Enrich metadata
            n_unresolved = _enrich_metadata(
                dataset.metadata, exp_name, exp_num,
                csv_path=self.registry_csv,
            )
            if n_unresolved > 0:
                logger.warning(
                    "  %d/%d windows could not be resolved from registry",
                    n_unresolved, len(dataset),
                )

            logger.info(
                "  %s: %d windows from %d features",
                exp_name, len(dataset), dataset.n_features,
            )
            all_datasets.append(dataset)

        phase1_time = time.monotonic() - t0
        logger.info("Phase 1 done in %s", _fmt_duration(phase1_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 2: Concatenate datasets
        # ------------------------------------------------------------------
        logger.info("PHASE 2: Concatenation")
        logger.info("-" * 40)

        all_features = [ds.features for ds in all_datasets if len(ds) > 0]
        all_metadata: list[WindowMetadata] = []
        for ds in all_datasets:
            all_metadata.extend(ds.metadata)

        if not all_features:
            logger.error("No features extracted from any project.")
            return ClusteringResult(
                features_normalized=np.empty((0, len(ANALYSIS_FEATURE_NAMES))),
                cluster_labels=np.array([], dtype=np.int32),
                n_clusters=0,
                embedding_2d=np.empty((0, 2)),
                metadata=[],
                feature_names=list(ANALYSIS_FEATURE_NAMES),
                method="none",
                embedding_method="none",
            )

        combined_features = np.vstack(all_features)
        combined_dataset = BehaviorDataset(
            features=combined_features,
            metadata=all_metadata,
        )

        logger.info("Combined: %d windows, %d features", len(combined_dataset), combined_dataset.n_features)

        # Log per-experiment counts
        from collections import Counter
        exp_counts = Counter(m.experiment_name for m in all_metadata)
        for exp, cnt in exp_counts.most_common():
            logger.info("  %s: %d windows", exp, cnt)
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 3: Batch correction
        # ------------------------------------------------------------------
        logger.info("PHASE 3: Batch correction (%s)", self.batch_correction_method)
        logger.info("-" * 40)
        t0 = time.monotonic()

        batch_labels = np.array([m.batch_id for m in all_metadata])
        unique_batches = np.unique(batch_labels)
        logger.info("  %d unique batches", len(unique_batches))

        # Impute NaN before batch correction (column median)
        X = combined_features.copy()
        for col in range(X.shape[1]):
            mask = np.isfinite(X[:, col])
            if not np.all(mask):
                median_val = float(np.nanmedian(X[:, col])) if np.any(mask) else 0.0
                X[~mask, col] = median_val

        if self.batch_correction_method == "combat" and len(unique_batches) >= 2:
            corrected, batch_info = combat_correct(X, batch_labels)
        elif self.batch_correction_method == "zscore_per_batch" and len(unique_batches) >= 2:
            corrected, batch_info = zscore_per_batch(X, batch_labels)
        else:
            corrected = X
            batch_info = {"method": "none", "n_batches": len(unique_batches)}
            if len(unique_batches) < 2:
                logger.warning("Only %d batch(es) — skipping correction.", len(unique_batches))

        # Replace the features in the combined dataset
        combined_dataset.features = corrected

        phase3_time = time.monotonic() - t0
        logger.info("Batch correction done in %s", _fmt_duration(phase3_time))
        logger.info("")

        # Save pre/post correction features for diagnostic plots
        pre_correction_features = X.copy()

        # ------------------------------------------------------------------
        # Phase 3.5: Batch diagnostics
        # ------------------------------------------------------------------
        logger.info("PHASE 3.5: Batch diagnostics")
        logger.info("-" * 40)
        t0 = time.monotonic()

        batch_diag_dir = self.output_dir / "batch_diagnostics"
        try:
            from .batch_diagnostics import run_batch_diagnostics
            _feat_names = list(combined_dataset.feature_names)
            _a_idx = analysis_feature_indices(_feat_names)
            run_batch_diagnostics(
                pre_correction_features[:, _a_idx],
                corrected[:, _a_idx],
                batch_labels,
                analysis_feature_names(_feat_names),
                all_metadata,
                batch_diag_dir,
                batch_info,
            )
        except Exception as e:
            logger.warning("Batch diagnostics failed: %s", e)

        phase35_time = time.monotonic() - t0
        logger.info("Phase 3.5 done in %s", _fmt_duration(phase35_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 4: Clustering
        # ------------------------------------------------------------------
        logger.info("PHASE 4: Clustering + embedding")
        logger.info("-" * 40)
        t0 = time.monotonic()

        result = run_clustering(combined_dataset, self.cfg)

        phase4_time = time.monotonic() - t0
        logger.info("Phase 4 done in %s", _fmt_duration(phase4_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 4.5: Single-animal analysis
        # ------------------------------------------------------------------
        logger.info("PHASE 4.5: Single-animal analysis")
        logger.info("-" * 40)
        t0 = time.monotonic()

        from .single_animal import split_to_single_animal
        from .multi_project_comparison import compare_single_animal_experiments

        sa_dataset = split_to_single_animal(combined_dataset)
        logger.info(
            "  Split to single-animal: %d pair windows -> %d single-animal samples",
            len(combined_dataset), len(sa_dataset),
        )

        # Batch correct single-animal dataset
        sa_batch_labels = np.array([m.batch_id for m in sa_dataset.metadata])
        sa_X = sa_dataset.features.copy()
        for col in range(sa_X.shape[1]):
            mask = np.isfinite(sa_X[:, col])
            if not np.all(mask):
                median_val = float(np.nanmedian(sa_X[:, col])) if np.any(mask) else 0.0
                sa_X[~mask, col] = median_val

        sa_unique_batches = np.unique(sa_batch_labels)
        if self.batch_correction_method == "combat" and len(sa_unique_batches) >= 2:
            sa_corrected, _ = combat_correct(sa_X, sa_batch_labels)
        elif self.batch_correction_method == "zscore_per_batch" and len(sa_unique_batches) >= 2:
            sa_corrected, _ = zscore_per_batch(sa_X, sa_batch_labels)
        else:
            sa_corrected = sa_X
        sa_dataset.features = sa_corrected

        sa_result = run_clustering(sa_dataset, self.cfg)
        sa_comparison = compare_single_animal_experiments(sa_result)

        phase45_time = time.monotonic() - t0
        logger.info("Phase 4.5 done in %s", _fmt_duration(phase45_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 5: Condition comparisons
        # ------------------------------------------------------------------
        logger.info("PHASE 5: Condition comparisons")
        logger.info("-" * 40)
        t0 = time.monotonic()

        comparison = compare_experiments(result)

        phase5_time = time.monotonic() - t0
        logger.info("Phase 5 done in %s", _fmt_duration(phase5_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 6: Locomotion comparison (optional)
        # ------------------------------------------------------------------
        locomotion_results: dict[str, Any] | None = None
        phase6_time = 0.0
        if self.locomotion_dir is not None:
            logger.info("PHASE 6: Locomotion comparison")
            logger.info("-" * 40)
            t0 = time.monotonic()
            locomotion_results = self._run_locomotion_comparison()
            phase6_time = time.monotonic() - t0
            logger.info("Phase 6 done in %s", _fmt_duration(phase6_time))
            logger.info("")

        # ------------------------------------------------------------------
        # Phase 6.5: Social interaction time
        # ------------------------------------------------------------------
        logger.info("PHASE 6.5: Social interaction time")
        logger.info("-" * 40)
        t0 = time.monotonic()
        social_data: dict[str, Any] | None = None

        try:
            from .social_interaction import compute_social_interaction_time
            social_data = compute_social_interaction_time(
                self.project_dirs,
                all_metadata,
            )
        except Exception as e:
            logger.warning("Social interaction time computation failed: %s", e)

        phase65_time = time.monotonic() - t0
        logger.info("Phase 6.5 done in %s", _fmt_duration(phase65_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 7: Plots
        # ------------------------------------------------------------------
        logger.info("PHASE 7: Generating plots")
        logger.info("-" * 40)
        t0 = time.monotonic()

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        self._generate_plots(
            result, comparison, plots_dir,
            pre_correction_features=pre_correction_features,
            locomotion_results=locomotion_results,
            sa_result=sa_result,
            sa_comparison=sa_comparison,
            social_data=social_data,
        )

        phase7_time = time.monotonic() - t0
        logger.info("Phase 7 done in %s", _fmt_duration(phase7_time))
        logger.info("")

        # ------------------------------------------------------------------
        # Phase 8: Save results
        # ------------------------------------------------------------------
        logger.info("PHASE 8: Saving results")
        logger.info("-" * 40)

        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        self._save_results(results_dir, result, comparison, batch_info, locomotion_results)

        # Single-animal results
        if sa_result is not None:
            sa_results_dir = results_dir / "single_animal"
            sa_results_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(sa_results_dir, sa_result, sa_comparison, batch_info)
            logger.info("  Wrote single-animal results to %s", sa_results_dir)

        # Social interaction data
        if social_data is not None:
            import json as _json
            p = results_dir / "social_interaction_time.json"
            p.write_text(_json.dumps(social_data, indent=2, default=str))
            logger.info("  Wrote %s", p.name)

        from .report_outputs import (
            build_multi_project_summary,
            write_multi_project_excel_report,
            write_summary_json,
        )

        summary = build_multi_project_summary(
            self.output_dir,
            self.project_dirs,
            self.cfg,
            result,
            comparison,
            batch_correction_method=self.batch_correction_method,
            locomotion_dir=self.locomotion_dir,
        )
        write_summary_json(self.output_dir, summary)
        write_multi_project_excel_report(
            self.output_dir,
            result,
            comparison,
            self.cfg,
            project_dirs=self.project_dirs,
            batch_correction_method=self.batch_correction_method,
        )

        overall_elapsed = time.monotonic() - overall_t0
        logger.info("")
        logger.info("#" * 70)
        logger.info("# MULTI-PROJECT PIPELINE COMPLETE")
        logger.info("#   Clusters: %d", result.n_clusters)
        logger.info("#   Windows:  %d", len(result.metadata))
        logger.info("#   Single-animal clusters: %d (%d samples)",
                     sa_result.n_clusters, len(sa_result.metadata))
        logger.info("#   Time breakdown:")
        logger.info("#     Feature extraction:     %s", _fmt_duration(phase1_time))
        logger.info("#     Batch correction:       %s", _fmt_duration(phase3_time))
        logger.info("#     Batch diagnostics:      %s", _fmt_duration(phase35_time))
        logger.info("#     Clustering + embedding: %s", _fmt_duration(phase4_time))
        logger.info("#     Single-animal analysis: %s", _fmt_duration(phase45_time))
        logger.info("#     Condition comparison:   %s", _fmt_duration(phase5_time))
        if locomotion_results is not None:
            logger.info("#     Locomotion comparison:  %s", _fmt_duration(phase6_time))
        logger.info("#     Social interaction:     %s", _fmt_duration(phase65_time))
        logger.info("#     Plots:                  %s", _fmt_duration(phase7_time))
        logger.info("#     Total:                  %s", _fmt_duration(overall_elapsed))
        logger.info("#   Output: %s", self.output_dir)
        logger.info("#" * 70)

        return result

    # ------------------------------------------------------------------
    # Locomotion comparison
    # ------------------------------------------------------------------

    def _run_locomotion_comparison(self) -> dict[str, Any]:
        """Run locomotion vs paired resident comparison."""
        from .locomotion_comparison import (
            compare_locomotion_vs_paired,
            extract_locomotion_features,
            extract_resident_features,
        )

        results: dict[str, Any] = {}

        logger.info("Extracting locomotion features from %s...", self.locomotion_dir)
        loc_features = extract_locomotion_features(
            self.locomotion_dir, self.cfg,
        )
        logger.info("  %d locomotion videos processed", len(loc_features))

        # Extract resident features from each interaction project
        all_paired_features: dict[str, dict[str, Any]] = {}
        for proj_dir in self.project_dirs:
            exp_name = experiment_name_from_project_dir(proj_dir)
            if exp_name not in ("test_day", "sh_intruder"):
                continue
            logger.info("Extracting resident features from %s...", proj_dir.name)
            paired = extract_resident_features(proj_dir, self.cfg)
            logger.info("  %d paired videos processed", len(paired))
            all_paired_features.update(paired)

        # Run comparison
        results = compare_locomotion_vs_paired(loc_features, all_paired_features)
        n_paired = len(results.get("paired_comparisons", []))
        logger.info("Locomotion comparison: %d housing-specific paired tests", n_paired)
        for comp in results.get("paired_comparisons", []):
            logger.info(
                "  %s: %d mouse-camera pairs",
                comp.get("key", "?"), comp.get("n_pairs", 0),
            )

        return results

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plots(
        self,
        result: ClusteringResult,
        comparison: dict[str, Any],
        plots_dir: Path,
        *,
        pre_correction_features: np.ndarray | None = None,
        locomotion_results: dict[str, Any] | None = None,
        sa_result: ClusteringResult | None = None,
        sa_comparison: dict[str, Any] | None = None,
        social_data: dict[str, Any] | None = None,
    ) -> None:
        """Generate all multi-project plots."""
        # Standard plots (heatmap, composition, enrichment, ethograms, violins)
        try:
            from .plots import generate_all_plots
            generate_all_plots(result, comparison, plots_dir)
        except Exception as e:
            logger.warning("Standard plots failed: %s", e)

        # Condition group + locomotion boxplots
        try:
            from .condition_boxplots import (
                plot_condition_boxplots,
                plot_cluster_composition,
                plot_locomotion_paired,
            )
            plot_condition_boxplots(result, comparison, plots_dir)
            plot_cluster_composition(result, comparison, plots_dir)
            if locomotion_results is not None:
                plot_locomotion_paired(locomotion_results, plots_dir)
        except Exception as e:
            logger.warning("Condition boxplots failed: %s", e)

        # Role condition boxplots belong in single-animal (see below)

        # Consolidated UMAP plots (into plots/umap/ subfolder)
        try:
            from .plots_umap import generate_all_umap_plots
            generate_all_umap_plots(
                result, comparison, plots_dir,
                pre_correction_features=pre_correction_features,
            )
        except Exception as e:
            logger.warning("UMAP plots failed: %s", e)

        # Pairwise comparison plots (with feature boxplots)
        try:
            from .plots_pairwise import generate_pairwise_comparison_plots
            generate_pairwise_comparison_plots(comparison, plots_dir, result=result)
        except Exception as e:
            logger.warning("Pairwise comparison plots failed: %s", e)

        # Transition plots
        try:
            from .condition_boxplots import _GROUP_ORDER
            from .plots_transitions import generate_transition_plots
            generate_transition_plots(
                result, comparison, plots_dir, group_order=_GROUP_ORDER,
            )
        except Exception as e:
            logger.warning("Transition network plots failed: %s", e)

        # Social interaction time plots
        if social_data is not None:
            try:
                from .social_interaction import plot_social_interaction_time
                plot_social_interaction_time(social_data, plots_dir)
            except Exception as e:
                logger.warning("Social interaction plots failed: %s", e)

        # Single-animal plots
        if sa_result is not None and sa_comparison is not None:
            sa_plots_dir = plots_dir / "single_animal"
            try:
                from .plots import generate_all_plots as _gen_plots
                _gen_plots(sa_result, sa_comparison, sa_plots_dir)
            except Exception as e:
                logger.warning("Single-animal standard plots failed: %s", e)

            try:
                from .plots_umap import generate_all_umap_plots as _gen_umap
                _gen_umap(sa_result, sa_comparison, sa_plots_dir)
            except Exception as e:
                logger.warning("Single-animal UMAP plots failed: %s", e)

            try:
                from .plots_pairwise import generate_pairwise_comparison_plots as _gen_pw
                _gen_pw(sa_comparison, sa_plots_dir, result=sa_result)
            except Exception as e:
                logger.warning("Single-animal pairwise plots failed: %s", e)

            # Role condition boxplots (uses SA mask builders for focal-animal roles)
            try:
                from .condition_boxplots import plot_role_condition_boxplots
                from .multi_project_comparison import SA_ROLE_MASK_BUILDERS
                plot_role_condition_boxplots(
                    sa_result, sa_plots_dir,
                    mask_builders=SA_ROLE_MASK_BUILDERS,
                )
            except Exception as e:
                logger.warning("Single-animal role boxplots failed: %s", e)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results_dir: Path,
        result: ClusteringResult,
        comparison: dict[str, Any],
        batch_info: dict[str, Any],
        locomotion_results: dict[str, Any] | None = None,
    ) -> None:
        """Save all result files."""

        # 1. Cluster assignments JSON
        assignments = []
        for i, m in enumerate(result.metadata):
            assignments.append({
                "window_index": i,
                "cluster": int(result.cluster_labels[i]),
                "video_id": m.video_id,
                "video_name": m.video_name,
                "start_frame": m.start_frame,
                "experiment_name": m.experiment_name,
                "experiment_number": m.experiment_number,
                "session": m.session,
                "mouse_a_id": m.mouse_a_id,
                "mouse_b_id": m.mouse_b_id,
                "mouse_a_role": m.mouse_a_role,
                "mouse_b_role": m.mouse_b_role,
                "mouse_a_housing": m.mouse_a_housing,
                "mouse_b_housing": m.mouse_b_housing,
                "batch_id": m.batch_id,
                "object_a_name": m.object_a_name,
                "object_b_name": m.object_b_name,
                "interaction_type": m.interaction_type,
                "camera_view": m.camera_view,
                "umap_x": float(result.embedding_2d[i, 0]),
                "umap_y": float(result.embedding_2d[i, 1]),
            })

        p = results_dir / "cluster_assignments.json"
        p.write_text(json.dumps(assignments, indent=2))
        logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))

        # 2. Cluster assignments CSV (with features)
        csv_path = results_dir / "cluster_assignments.csv"
        with open(csv_path, "w", newline="") as f:
            meta_cols = list(assignments[0].keys()) if assignments else []
            writer = csv.DictWriter(f, fieldnames=meta_cols + list(result.feature_names))
            writer.writeheader()
            for i, row in enumerate(assignments):
                for fi, fname in enumerate(result.feature_names):
                    row[fname] = float(result.features_normalized[i, fi])
                writer.writerow(row)
        logger.info("  Wrote %s (%s)", csv_path.name, _fmt_size(csv_path.stat().st_size))

        # 3. Cluster summary
        cluster_summary: dict[str, Any] = {}
        for c in range(result.n_clusters):
            mask = result.cluster_labels == c
            centroid = result.features_normalized[mask].mean(axis=0)
            cluster_summary[str(c)] = {
                "size": int(np.sum(mask)),
                "fraction": float(np.sum(mask)) / max(len(result.cluster_labels), 1),
                "feature_centroid": {
                    name: float(centroid[i])
                    for i, name in enumerate(result.feature_names)
                },
            }
        p = results_dir / "cluster_summary.json"
        p.write_text(json.dumps(cluster_summary, indent=2))
        logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))

        # 4. Condition comparison
        p = results_dir / "condition_comparison.json"
        p.write_text(json.dumps(comparison, indent=2, default=str))
        logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))

        try:
            from .plots_pairwise import save_pairwise_comparison_results
            save_pairwise_comparison_results(comparison, results_dir)
        except Exception as e:
            logger.warning("Pairwise comparison CSV export failed: %s", e)

        # 5. Batch correction info
        p = results_dir / "batch_correction_info.json"
        p.write_text(json.dumps(batch_info, indent=2, default=str))
        logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))

        # 6. Locomotion results
        if locomotion_results is not None:
            p = results_dir / "locomotion_comparison.json"
            p.write_text(json.dumps(locomotion_results, indent=2, default=str))
            logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))

            loc_dir = results_dir / "locomotion_comparisons"
            loc_dir.mkdir(parents=True, exist_ok=True)
            for comp in locomotion_results.get("paired_comparisons", []):
                key = comp.get("key", "comparison")
                comp_path = loc_dir / f"{key}.json"
                comp_path.write_text(json.dumps(comp, indent=2, default=str))
                per_feat = comp.get("per_feature", [])
                if per_feat:
                    csv_path = loc_dir / f"{key}_stats.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "feature", "mean_alone", "mean_paired",
                                "wilcoxon_p_value", "mannwhitney_p_value",
                            ],
                            extrasaction="ignore",
                        )
                        writer.writeheader()
                        writer.writerows(per_feat)
            logger.info("  Wrote locomotion comparison tables to %s", loc_dir.name)

        # 7. Combined dataset (npz + json sidecar) for reproducibility
        ds_path = results_dir / "combined_dataset.npz"
        combined_ds = BehaviorDataset(
            features=result.features_normalized,
            metadata=result.metadata,
            feature_names=list(result.feature_names),
        )
        combined_ds.save(ds_path)
        logger.info("  Wrote %s + .json sidecar", ds_path.name)

        # 8. Embedding coordinates (for Excel regeneration)
        p = results_dir / "embedding.npz"
        np.savez_compressed(
            p,
            embedding=result.embedding_2d,
            labels=result.cluster_labels,
        )
        logger.info("  Wrote %s (%s)", p.name, _fmt_size(p.stat().st_size))
