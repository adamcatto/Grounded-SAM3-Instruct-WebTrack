"""Downstream comparison: group clusters by housing condition, statistical tests."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

from .clustering_pipeline import ClusteringResult
from .dataset import housing_condition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared building blocks (used by both social and single-animal comparisons)
# ---------------------------------------------------------------------------

def _build_ethograms(meta: list, labels: np.ndarray) -> dict[str, dict[str, Any]]:
    """Per-video cluster-label timelines, keyed by video_id."""
    ethograms: dict[str, dict[str, Any]] = {}
    vid_groups: defaultdict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        vid_groups[m.video_id].append(i)

    for vid_id, indices in vid_groups.items():
        indices_sorted = sorted(indices, key=lambda i: meta[i].start_frame)
        ethograms[vid_id] = {
            "video_name": meta[indices_sorted[0]].video_name,
            "n_windows": len(indices_sorted),
            "start_frames": [meta[i].start_frame for i in indices_sorted],
            "cluster_labels": [int(labels[i]) for i in indices_sorted],
            "interaction_type": meta[indices_sorted[0]].interaction_type,
        }
    return ethograms


def _build_transition_data(
    ethograms: dict[str, dict[str, Any]],
    n_clusters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transition matrices grouped by interaction_type + differential matrices."""
    from .transition_analysis import (
        compute_differential_transition_matrices,
        compute_transition_matrices_by_group,
    )

    transition_matrices = compute_transition_matrices_by_group(
        ethograms,
        n_clusters,
        group_field="interaction_type",
        exclude_groups={"unknown", "single_animal"},
    )
    transition_group_order = sorted(transition_matrices.keys())
    differential_transition_matrices = compute_differential_transition_matrices(
        transition_matrices,
        transition_group_order,
        exclude_groups={"unknown", "single_animal"},
    )
    return transition_matrices, differential_transition_matrices


def compare_housing_conditions(result: ClusteringResult) -> dict[str, Any]:
    """Analyze cluster composition by housing condition and interaction type.

    Returns a dict with:
      - per_cluster: enrichment analysis per cluster
      - overall_enrichment: global test of association
      - per_feature_tests: Mann-Whitney U per feature between housing groups
      - interaction_type_distribution: cluster composition by interaction type
      - ethograms: per-video cluster label timelines
    """
    if len(result.metadata) == 0:
        return {"note": "no data"}

    t0 = time.monotonic()

    labels = result.cluster_labels
    n_clusters = result.n_clusters
    meta = result.metadata
    n_total = len(labels)

    logger.info("Running housing condition comparison on %d windows, %d clusters...", n_total, n_clusters)

    # ---------------------------------------------------------------------------
    # Classify each window's objects
    # ---------------------------------------------------------------------------

    obj_a_conditions = [housing_condition(m.object_a_name) for m in meta]
    obj_b_conditions = [housing_condition(m.object_b_name) for m in meta]
    interaction_types = [m.interaction_type for m in meta]

    # Mask of windows that contain valid social interactions (exclude single-animal and unknown)
    valid_comparison_mask = np.array([
        "single" not in it.lower() and "unknown" not in it.lower()
        for it in interaction_types
    ])

    itype_counts = Counter(interaction_types)
    logger.info("Interaction type breakdown:")
    for itype, cnt in itype_counts.most_common():
        logger.info("  %s: %d windows (%.1f%%)", itype, cnt, 100.0 * cnt / n_total)

    # ---------------------------------------------------------------------------
    # 1. Per-cluster enrichment by housing condition
    # ---------------------------------------------------------------------------

    logger.info("Computing per-cluster enrichment (Fisher's exact test)...")
    per_cluster: dict[str, dict[str, Any]] = {}

    for c in range(n_clusters):
        mask = labels == c
        n_in = int(np.sum(mask))
        n_out = int(np.sum(~mask))

        # Count object conditions in this cluster (both mice contribute)
        cond_counts: Counter[str] = Counter()
        for i in np.where(mask)[0]:
            cond_counts[obj_a_conditions[i]] += 1
            cond_counts[obj_b_conditions[i]] += 1

        # Count interaction types in this cluster
        itype_in_cluster: Counter[str] = Counter()
        for i in np.where(mask)[0]:
            itype_in_cluster[interaction_types[i]] += 1

        # Fisher's exact test: isolated enrichment (excluding single animal and unknown)
        iso_in = sum(1 for i in np.where(mask)[0]
                     if (obj_a_conditions[i] == "isolated" or obj_b_conditions[i] == "isolated") and valid_comparison_mask[i])
        iso_out = sum(1 for i in np.where(~mask)[0]
                      if (obj_a_conditions[i] == "isolated" or obj_b_conditions[i] == "isolated") and valid_comparison_mask[i])
        grp_in = sum(1 for i in np.where(mask)[0] if valid_comparison_mask[i]) - iso_in
        grp_out = sum(1 for i in np.where(~mask)[0] if valid_comparison_mask[i]) - iso_out

        if n_in > 0 and n_out > 0:
            table = [[iso_in, grp_in], [iso_out, grp_out]]
            try:
                odds_ratio, p_value = fisher_exact(table)
            except ValueError:
                odds_ratio, p_value = float("nan"), float("nan")
        else:
            odds_ratio, p_value = float("nan"), float("nan")

        sig = " ***" if (np.isfinite(p_value) and p_value < 0.001) else \
              " **" if (np.isfinite(p_value) and p_value < 0.01) else \
              " *" if (np.isfinite(p_value) and p_value < 0.05) else ""

        logger.info(
            "  Cluster %d: %d windows (%.1f%%) | iso=%d grp=%d | OR=%.2f p=%.4g%s",
            c, n_in, 100.0 * n_in / n_total, iso_in, grp_in,
            odds_ratio if np.isfinite(odds_ratio) else 0.0,
            p_value if np.isfinite(p_value) else 1.0,
            sig,
        )

        per_cluster[str(c)] = {
            "size": n_in,
            "fraction": n_in / n_total,
            "object_condition_counts": dict(cond_counts),
            "interaction_type_counts": dict(itype_in_cluster),
            "isolated_enrichment": {
                "isolated_in_cluster": iso_in,
                "group_in_cluster": grp_in,
                "isolated_outside": iso_out,
                "group_outside": grp_out,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
            },
        }

    # ---------------------------------------------------------------------------
    # 2. Per-feature Mann-Whitney U between interaction types
    # ---------------------------------------------------------------------------

    per_feature_tests: dict[str, dict[str, Any]] = {}
    itypes_unique = sorted(set(interaction_types))
    itypes_for_comparison = [
        it for it in itypes_unique
        if "single" not in it.lower() and "unknown" not in it.lower()
    ]

    if len(itypes_for_comparison) >= 2:
        type_a, type_b = itypes_for_comparison[0], itypes_for_comparison[1]
        mask_a = np.array([it == type_a for it in interaction_types])
        mask_b = np.array([it == type_b for it in interaction_types])
        n_a, n_b = int(np.sum(mask_a)), int(np.sum(mask_b))

        logger.info(
            "Per-feature Mann-Whitney U: %s (n=%d) vs %s (n=%d)",
            type_a, n_a, type_b, n_b,
        )

        if n_a >= 5 and n_b >= 5:
            n_sig = 0
            for fi, fname in enumerate(result.feature_names):
                vals_a = result.features_normalized[mask_a, fi]
                vals_b = result.features_normalized[mask_b, fi]
                vals_a = vals_a[np.isfinite(vals_a)]
                vals_b = vals_b[np.isfinite(vals_b)]
                if len(vals_a) >= 2 and len(vals_b) >= 2:
                    try:
                        stat, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                    except ValueError:
                        stat, pval = float("nan"), float("nan")
                    direction = "higher_in_a" if np.mean(vals_a) > np.mean(vals_b) else "higher_in_b"
                    per_feature_tests[fname] = {
                        "comparison": f"{type_a} vs {type_b}",
                        "U_statistic": float(stat),
                        "p_value": float(pval),
                        "mean_a": float(np.mean(vals_a)),
                        "mean_b": float(np.mean(vals_b)),
                        "effect_direction": direction,
                    }
                    if np.isfinite(pval) and pval < 0.05:
                        n_sig += 1

            logger.info(
                "  %d/%d features significant at p<0.05",
                n_sig, len(per_feature_tests),
            )
            # Log top 5 most significant
            sorted_feats = sorted(
                per_feature_tests.items(),
                key=lambda x: x[1]["p_value"] if np.isfinite(x[1]["p_value"]) else 999,
            )
            for fname, fdata in sorted_feats[:5]:
                logger.info(
                    "    %s: p=%.4g (mean: %.3f vs %.3f, %s)",
                    fname, fdata["p_value"],
                    fdata["mean_a"], fdata["mean_b"],
                    fdata["effect_direction"],
                )
        else:
            logger.warning("  Too few samples in one group for statistical testing.")

    # ---------------------------------------------------------------------------
    # 3. Interaction type distribution across clusters
    # ---------------------------------------------------------------------------

    itype_dist: dict[str, dict[str, int]] = {}
    for c in range(n_clusters):
        mask = labels == c
        counts: Counter[str] = Counter()
        for i in np.where(mask)[0]:
            counts[interaction_types[i]] += 1
        itype_dist[str(c)] = dict(counts)

    # ---------------------------------------------------------------------------
    # 4. Ethograms: per-video cluster label timelines
    # ---------------------------------------------------------------------------

    logger.info("Building ethograms...")
    ethograms = _build_ethograms(meta, labels)

    # ---------------------------------------------------------------------------
    # 5. Per-feature enrichment (effect size, fold change, odds-ratio-style)
    # ---------------------------------------------------------------------------

    logger.info("Computing per-feature enrichment (isolated vs group-only windows)...")
    feature_enrichment: list[dict[str, Any]] = []

    # Split windows: those involving an isolated mouse vs group-only pairs (excluding single-animal and unknown)
    iso_mask = np.array([
        (obj_a_conditions[i] == "isolated" or obj_b_conditions[i] == "isolated") and valid_comparison_mask[i]
        for i in range(n_total)
    ])
    grp_mask = np.array([
        valid_comparison_mask[i] and not iso_mask[i]
        for i in range(n_total)
    ])
    n_iso, n_grp = int(np.sum(iso_mask)), int(np.sum(grp_mask))

    if n_iso >= 5 and n_grp >= 5:
        for fi, fname in enumerate(result.feature_names):
            vals_iso = result.features_normalized[iso_mask, fi]
            vals_grp = result.features_normalized[grp_mask, fi]
            vals_iso = vals_iso[np.isfinite(vals_iso)]
            vals_grp = vals_grp[np.isfinite(vals_grp)]

            if len(vals_iso) < 2 or len(vals_grp) < 2:
                continue

            mean_iso = float(np.mean(vals_iso))
            mean_grp = float(np.mean(vals_grp))
            std_iso = float(np.std(vals_iso, ddof=1))
            std_grp = float(np.std(vals_grp, ddof=1))

            # Cohen's d (pooled SD)
            pooled_std = np.sqrt(
                ((len(vals_iso) - 1) * std_iso**2 + (len(vals_grp) - 1) * std_grp**2)
                / (len(vals_iso) + len(vals_grp) - 2)
            )
            cohens_d = (mean_iso - mean_grp) / pooled_std if pooled_std > 1e-12 else 0.0

            # Log2 fold change (shift to positive range to avoid log of negative)
            # Use absolute means offset by median to keep sign meaningful
            median_all = float(np.median(
                result.features_normalized[np.isfinite(result.features_normalized[:, fi]), fi]
            ))
            shift = abs(min(mean_iso, mean_grp, median_all)) + 1.0
            log2fc = float(np.log2((mean_iso + shift) / (mean_grp + shift)))

            # Odds ratio: binarize at median, compute 2x2 table
            above_iso = int(np.sum(vals_iso > median_all))
            below_iso = len(vals_iso) - above_iso
            above_grp = int(np.sum(vals_grp > median_all))
            below_grp = len(vals_grp) - above_grp
            table = [[above_iso, below_iso], [above_grp, below_grp]]
            try:
                odds_r, p_fisher = fisher_exact(table)
            except ValueError:
                odds_r, p_fisher = float("nan"), float("nan")

            # Mann-Whitney U
            try:
                u_stat, p_mw = mannwhitneyu(vals_iso, vals_grp, alternative="two-sided")
            except ValueError:
                u_stat, p_mw = float("nan"), float("nan")

            feature_enrichment.append({
                "feature": fname,
                "mean_isolated": mean_iso,
                "mean_group": mean_grp,
                "std_isolated": std_iso,
                "std_group": std_grp,
                "cohens_d": float(cohens_d),
                "log2_fold_change": log2fc,
                "odds_ratio": float(odds_r),
                "fisher_p_value": float(p_fisher),
                "mannwhitney_U": float(u_stat),
                "mannwhitney_p_value": float(p_mw),
                "n_isolated": len(vals_iso),
                "n_group": len(vals_grp),
                "direction": "higher_in_isolated" if mean_iso > mean_grp else "higher_in_group",
            })

        # Sort by p-value
        feature_enrichment.sort(
            key=lambda x: x["mannwhitney_p_value"] if np.isfinite(x["mannwhitney_p_value"]) else 999
        )

        n_sig = sum(1 for f in feature_enrichment
                    if np.isfinite(f["mannwhitney_p_value"]) and f["mannwhitney_p_value"] < 0.05)
        logger.info("  %d/%d features significant (MW p<0.05)", n_sig, len(feature_enrichment))
        for fe in feature_enrichment[:5]:
            sig = ""
            p = fe["mannwhitney_p_value"]
            if np.isfinite(p):
                sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            logger.info(
                "    %s: d=%.3f log2FC=%.3f OR=%.2f p=%.4g%s (%s)",
                fe["feature"], fe["cohens_d"], fe["log2_fold_change"],
                fe["odds_ratio"], p, sig, fe["direction"],
            )
    else:
        logger.warning("  Not enough isolated (%d) or group-only (%d) windows for enrichment.", n_iso, n_grp)

    # ---------------------------------------------------------------------------
    # 6. Overall enrichment summary
    # ---------------------------------------------------------------------------

    overall_enrichment: dict[str, Any] = {}
    total_iso = sum(1 for i, (a, b) in enumerate(zip(obj_a_conditions, obj_b_conditions))
                    if (a == "isolated" or b == "isolated") and valid_comparison_mask[i])
    total_grp = sum(1 for i in range(n_total) if valid_comparison_mask[i]) - total_iso
    overall_enrichment["total_windows_with_isolated"] = total_iso
    overall_enrichment["total_windows_group_only"] = total_grp
    overall_enrichment["interaction_type_counts"] = dict(Counter(interaction_types))

    transition_matrices, differential_transition_matrices = _build_transition_data(
        ethograms, n_clusters,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Comparison complete in %.1fs: %d clusters, %d feature tests, %d enriched features, "
        "%d video ethograms, %d transition groups.",
        elapsed, n_clusters, len(per_feature_tests), len(feature_enrichment), len(ethograms),
        len(transition_matrices),
    )

    return {
        "per_cluster": per_cluster,
        "overall_enrichment": overall_enrichment,
        "per_feature_tests": per_feature_tests,
        "feature_enrichment": feature_enrichment,
        "interaction_type_distribution": itype_dist,
        "ethograms": ethograms,
        "transition_matrices": transition_matrices,
        "differential_transition_matrices": differential_transition_matrices,
    }


def describe_single_animal_clusters(result: ClusteringResult) -> dict[str, Any]:
    """Descriptive cluster summary for single-animal projects (no social stats).

    Produces the same dict shape as :func:`compare_housing_conditions` but only
    the fields that are meaningful without a social pair:
      - ``per_cluster``: size + fraction only (no ``isolated_enrichment``)
      - ``interaction_type_distribution``: cluster composition (all "single_animal")
      - ``ethograms``: per-video cluster label timelines
      - ``transition_matrices`` / ``differential_transition_matrices``

    The social-only keys (``per_feature_tests``, ``feature_enrichment``) are
    omitted and ``overall_enrichment`` is left empty so downstream consumers that
    guard on presence simply skip them.
    """
    if len(result.metadata) == 0:
        return {"note": "no data"}

    t0 = time.monotonic()

    labels = result.cluster_labels
    n_clusters = result.n_clusters
    meta = result.metadata
    n_total = len(labels)

    logger.info(
        "Describing %d clusters over %d single-animal windows (no social comparison)...",
        n_clusters, n_total,
    )

    per_cluster: dict[str, dict[str, Any]] = {}
    itype_dist: dict[str, dict[str, int]] = {}
    for c in range(n_clusters):
        mask = labels == c
        n_in = int(np.sum(mask))
        per_cluster[str(c)] = {
            "size": n_in,
            "fraction": n_in / n_total if n_total else 0.0,
        }
        counts: Counter[str] = Counter()
        for i in np.where(mask)[0]:
            counts[meta[i].interaction_type] += 1
        itype_dist[str(c)] = dict(counts)
        logger.info("  Cluster %d: %d windows (%.1f%%)", c, n_in, 100.0 * n_in / n_total)

    ethograms = _build_ethograms(meta, labels)
    transition_matrices, differential_transition_matrices = _build_transition_data(
        ethograms, n_clusters,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Cluster description complete in %.1fs: %d clusters, %d video ethograms.",
        elapsed, n_clusters, len(ethograms),
    )

    return {
        "per_cluster": per_cluster,
        "overall_enrichment": {},
        "interaction_type_distribution": itype_dist,
        "ethograms": ethograms,
        "transition_matrices": transition_matrices,
        "differential_transition_matrices": differential_transition_matrices,
    }
