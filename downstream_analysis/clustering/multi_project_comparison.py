"""Statistical comparisons across experimental conditions for the multi-project pipeline.

Produces per-cluster enrichment, per-feature Mann-Whitney U tests, and
cluster composition analysis for each condition group.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

from .clustering_pipeline import ClusteringResult
from .dataset import WindowMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition group assignment
# ---------------------------------------------------------------------------

def _assign_condition_group(m: WindowMetadata) -> str:
    """Assign a window to one of the 6 condition groups based on metadata.

    Groups:
      'GH_littermate'       — Exp 1 (hab): two familiar GH mice
      'SH_res_GH_int'       — Exp 2 (test_day): SH resident + novel GH intruder
      'GH_res_GH_int'       — Exp 2 (test_day): GH resident + novel GH intruder
      'GH_res_SH_int'       — Exp 3 (sh_intruder): GH resident + SH intruder
      'unknown'             — could not classify
    """
    if m.experiment_name == "hab":
        return "GH_littermate"
    elif m.experiment_name == "test_day":
        # SH resident sessions vs GH resident sessions
        if m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH":
            return "SH_res_GH_int"
        else:
            return "GH_res_GH_int"
    elif m.experiment_name == "sh_intruder":
        return "GH_res_SH_int"
    return "unknown"


CONDITION_GROUP_LABELS = {
    "GH_littermate": "GH + Littermate",
    "SH_res_GH_int": "SH Resident + GH Intruder",
    "GH_res_GH_int": "GH Resident + GH Intruder",
    "GH_res_SH_int": "GH Resident + SH Intruder",
}


# ---------------------------------------------------------------------------
# Pairwise comparison helper
# ---------------------------------------------------------------------------

def _pairwise_comparison(
    result: ClusteringResult,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Run Fisher's exact (per-cluster) and Mann-Whitney U (per-feature) between two groups."""
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    labels = result.cluster_labels

    comparison: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": n_a,
        "n_b": n_b,
    }

    if n_a < 5 or n_b < 5:
        comparison["note"] = "too few samples for statistical testing"
        return comparison

    # Per-cluster Fisher's exact test (enrichment in group A)
    per_cluster: dict[str, dict[str, Any]] = {}
    for c in range(result.n_clusters):
        c_mask = labels == c
        a_in = int((mask_a & c_mask).sum())
        b_in = int((mask_b & c_mask).sum())
        a_out = n_a - a_in
        b_out = n_b - b_in
        try:
            odds_ratio, p_value = fisher_exact([[a_in, b_in], [a_out, b_out]])
        except ValueError:
            odds_ratio, p_value = float("nan"), float("nan")
        per_cluster[str(c)] = {
            "a_in_cluster": a_in,
            "b_in_cluster": b_in,
            "a_frac": a_in / max(n_a, 1),
            "b_frac": b_in / max(n_b, 1),
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
        }
    comparison["per_cluster"] = per_cluster

    # Per-feature Mann-Whitney U
    per_feature: list[dict[str, Any]] = []
    for fi, fname in enumerate(result.feature_names):
        vals_a = result.features_normalized[mask_a, fi]
        vals_b = result.features_normalized[mask_b, fi]
        vals_a = vals_a[np.isfinite(vals_a)]
        vals_b = vals_b[np.isfinite(vals_b)]
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue
        try:
            u_stat, p_val = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        except ValueError:
            u_stat, p_val = float("nan"), float("nan")

        mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
        std_a, std_b = float(np.std(vals_a, ddof=1)), float(np.std(vals_b, ddof=1))
        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * std_a ** 2 + (len(vals_b) - 1) * std_b ** 2)
            / max(len(vals_a) + len(vals_b) - 2, 1)
        )
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 1e-12 else 0.0

        per_feature.append({
            "feature": fname,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "cohens_d": float(cohens_d),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "direction": f"higher_in_{label_a}" if mean_a > mean_b else f"higher_in_{label_b}",
        })

    per_feature.sort(
        key=lambda x: x["p_value"] if np.isfinite(x["p_value"]) else 999
    )
    comparison["per_feature"] = per_feature

    n_sig = sum(1 for f in per_feature if np.isfinite(f["p_value"]) and f["p_value"] < 0.05)
    comparison["n_significant_features"] = n_sig

    return comparison


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_experiments(result: ClusteringResult) -> dict[str, Any]:
    """Run all multi-project comparisons.

    Returns a dict with:
      - condition_groups: per-group window counts and cluster composition
      - comparisons: the 4 specific researcher-requested comparisons
      - per_cluster_by_experiment: cluster enrichment by experiment number
      - ethograms: per-video cluster label timelines
    """
    if len(result.metadata) == 0:
        return {"note": "no data"}

    t0 = time.monotonic()
    meta = result.metadata
    labels = result.cluster_labels
    n_total = len(labels)

    logger.info("Multi-project comparison: %d windows, %d clusters", n_total, result.n_clusters)

    # Assign condition groups
    groups = np.array([_assign_condition_group(m) for m in meta])
    group_counts = Counter(groups)
    logger.info("Condition group breakdown:")
    for g, cnt in group_counts.most_common():
        label = CONDITION_GROUP_LABELS.get(g, g)
        logger.info("  %s: %d windows (%.1f%%)", label, cnt, 100.0 * cnt / n_total)

    # ---------------------------------------------------------------------------
    # 1. Cluster composition per condition group
    # ---------------------------------------------------------------------------
    condition_group_info: dict[str, dict[str, Any]] = {}
    for g in sorted(group_counts):
        g_mask = groups == g
        n_g = int(g_mask.sum())
        cluster_fracs: dict[str, float] = {}
        for c in range(result.n_clusters):
            c_in_g = int((g_mask & (labels == c)).sum())
            cluster_fracs[str(c)] = c_in_g / max(n_g, 1)
        condition_group_info[g] = {
            "label": CONDITION_GROUP_LABELS.get(g, g),
            "n_windows": n_g,
            "fraction": n_g / n_total,
            "cluster_composition": cluster_fracs,
        }

    # ---------------------------------------------------------------------------
    # 2. Specific comparisons
    # ---------------------------------------------------------------------------
    comparisons: dict[str, dict[str, Any]] = {}

    # 2a. SH resident (exp2) vs SH intruder (exp3)
    mask_sh_res = np.array([
        m.experiment_name == "test_day" and m.mouse_a_housing == "SH" and m.mouse_a_role == "resident"
        for m in meta
    ])
    # For exp3, the SH mice are intruders
    mask_sh_int = np.array([
        m.experiment_name == "sh_intruder" and (m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH")
        for m in meta
    ])
    comparisons["SH_resident_vs_SH_intruder"] = _pairwise_comparison(
        result, mask_sh_res, mask_sh_int,
        "SH_resident_exp2", "SH_intruder_exp3",
    )
    logger.info(
        "Comparison SH_resident vs SH_intruder: %d vs %d windows",
        int(mask_sh_res.sum()), int(mask_sh_int.sum()),
    )

    # 2b. GH resident vs intruder within exp2
    mask_gh_res_exp2 = np.array([
        m.experiment_name == "test_day" and m.mouse_a_housing == "GH" and m.mouse_a_role == "resident"
        for m in meta
    ])
    # The intruder windows are the same videos but from the intruder perspective —
    # since features are computed per-pair, we compare GH-resident sessions vs SH-resident sessions
    # Actually we compare group composition: GH_res_GH_int group
    mask_gh_int_exp2 = groups == "GH_res_GH_int"
    mask_sh_int_exp2 = groups == "SH_res_GH_int"
    comparisons["SH_vs_GH_residents_exp2"] = _pairwise_comparison(
        result, mask_sh_int_exp2, mask_gh_int_exp2,
        "SH_resident_exp2", "GH_resident_exp2",
    )
    logger.info(
        "Comparison SH vs GH residents (exp2): %d vs %d windows",
        int(mask_sh_int_exp2.sum()), int(mask_gh_int_exp2.sum()),
    )

    # 2c. Exp1 littermate profile (vs all other experiments)
    mask_exp1 = groups == "GH_littermate"
    mask_not_exp1 = ~mask_exp1 & (groups != "unknown")
    comparisons["littermate_vs_other"] = _pairwise_comparison(
        result, mask_exp1, mask_not_exp1,
        "GH_littermate", "other_interactions",
    )
    logger.info("Comparison littermate vs other: %d vs %d windows",
                int(mask_exp1.sum()), int(mask_not_exp1.sum()))

    # 2d. GH resident + GH intruder (exp2) vs GH resident + SH intruder (exp3)
    mask_gh_res_gh_int = groups == "GH_res_GH_int"
    mask_gh_res_sh_int = groups == "GH_res_SH_int"
    comparisons["GH_with_GH_intruder_vs_SH_intruder"] = _pairwise_comparison(
        result, mask_gh_res_gh_int, mask_gh_res_sh_int,
        "GH_res+GH_int", "GH_res+SH_int",
    )
    logger.info(
        "Comparison GH+GH_int vs GH+SH_int: %d vs %d windows",
        int(mask_gh_res_gh_int.sum()), int(mask_gh_res_sh_int.sum()),
    )

    # ---------------------------------------------------------------------------
    # 3. Per-cluster enrichment by experiment
    # ---------------------------------------------------------------------------
    exp_numbers = np.array([m.experiment_number for m in meta])
    per_cluster_by_exp: dict[str, dict[str, Any]] = {}
    for c in range(result.n_clusters):
        c_mask = labels == c
        exp_counts: dict[int, int] = {}
        for exp in sorted(set(exp_numbers)):
            if exp == 0:
                continue
            exp_counts[int(exp)] = int((c_mask & (exp_numbers == exp)).sum())
        per_cluster_by_exp[str(c)] = {
            "size": int(c_mask.sum()),
            "experiment_counts": exp_counts,
        }

    # ---------------------------------------------------------------------------
    # 4. Ethograms
    # ---------------------------------------------------------------------------
    from collections import defaultdict
    ethograms: dict[str, dict[str, Any]] = {}
    vid_groups: defaultdict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        vid_groups[m.video_id].append(i)

    for vid_id, indices in vid_groups.items():
        indices_sorted = sorted(indices, key=lambda i: meta[i].start_frame)
        m0 = meta[indices_sorted[0]]
        ethograms[vid_id] = {
            "video_name": m0.video_name,
            "experiment_name": m0.experiment_name,
            "session": m0.session,
            "condition_group": _assign_condition_group(m0),
            "n_windows": len(indices_sorted),
            "start_frames": [meta[i].start_frame for i in indices_sorted],
            "cluster_labels": [int(labels[i]) for i in indices_sorted],
        }

    elapsed = time.monotonic() - t0
    logger.info("Multi-project comparison done in %.1fs", elapsed)

    return {
        "condition_groups": condition_group_info,
        "comparisons": comparisons,
        "per_cluster_by_experiment": per_cluster_by_exp,
        "ethograms": ethograms,
    }
