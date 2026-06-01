"""Statistical comparisons across experimental conditions for the multi-project pipeline.

Produces per-cluster enrichment, per-feature Mann-Whitney U tests, feature enrichment
(Cohen's d, log2FC, Fisher OR), and all N-choose-2 pairwise comparisons for:
  - Condition groups (upper triangle of the 4-group matrix)
  - Role conditions (SH/GH resident, intruder, GH littermate)
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

from .clustering_pipeline import ClusteringResult
from .dataset import WindowMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition group assignment
# ---------------------------------------------------------------------------

def _assign_condition_group(m: WindowMetadata) -> str:
    """Assign a window to one of the 4 condition groups based on metadata."""
    if m.experiment_name == "hab":
        return "GH_littermate"
    if m.experiment_name == "test_day":
        if m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH":
            return "SH_res_GH_int"
        return "GH_res_GH_int"
    if m.experiment_name == "sh_intruder":
        return "GH_res_SH_int"
    return "unknown"


def _assign_sa_role(m: WindowMetadata) -> str:
    """Assign a single-animal sample to one of the 5 role categories.

    In SA mode the focal animal's identity is always in mouse_a_* fields.
    """
    if m.experiment_name == "hab":
        return "GH_littermate"
    if m.experiment_name == "test_day":
        if m.mouse_a_housing == "SH" and m.mouse_a_role == "resident":
            return "SH_resident"
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "resident":
            return "GH_resident"
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "intruder":
            return "GH_intruder"
    if m.experiment_name == "sh_intruder":
        if m.mouse_a_housing == "SH":
            return "SH_intruder"
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "resident":
            return "GH_resident"
    return "unknown"


CONDITION_GROUP_ORDER = [
    "GH_littermate",
    "SH_res_GH_int",
    "GH_res_GH_int",
    "GH_res_SH_int",
]

CONDITION_GROUP_LABELS = {
    "GH_littermate": "GH + Littermate",
    "SH_res_GH_int": "SH Resident + GH Intruder",
    "GH_res_GH_int": "GH Resident + GH Intruder",
    "GH_res_SH_int": "GH Resident + SH Intruder",
}

# Role-specific conditions for resident/intruder/littermate comparisons
ROLE_CONDITION_ORDER = [
    "SH_resident",
    "SH_intruder",
    "GH_resident",
    "GH_intruder",
    "GH_littermate",
]

ROLE_CONDITION_LABELS = {
    "SH_resident": "SH Resident",
    "SH_intruder": "SH Intruder",
    "GH_resident": "GH Resident",
    "GH_intruder": "GH Intruder",
    "GH_littermate": "GH Littermate",
}


def _mask_sh_resident(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "test_day"
        and (
            (m.mouse_a_role == "resident" and m.mouse_a_housing == "SH")
            or (m.mouse_b_role == "resident" and m.mouse_b_housing == "SH")
        )
        for m in meta
    ])


def _mask_sh_intruder(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "sh_intruder"
        and (m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH")
        for m in meta
    ])


def _mask_gh_resident(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name in ("test_day", "sh_intruder")
        and (
            (m.mouse_a_role == "resident" and m.mouse_a_housing == "GH")
            or (m.mouse_b_role == "resident" and m.mouse_b_housing == "GH")
        )
        for m in meta
    ])


def _mask_gh_intruder(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "test_day"
        and (
            (m.mouse_a_role == "intruder" and m.mouse_a_housing == "GH")
            or (m.mouse_b_role == "intruder" and m.mouse_b_housing == "GH")
        )
        for m in meta
    ])


def _mask_gh_littermate(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([m.experiment_name == "hab" for m in meta])


ROLE_MASK_BUILDERS: dict[str, Callable[[list[WindowMetadata]], np.ndarray]] = {
    "SH_resident": _mask_sh_resident,
    "SH_intruder": _mask_sh_intruder,
    "GH_resident": _mask_gh_resident,
    "GH_intruder": _mask_gh_intruder,
    "GH_littermate": _mask_gh_littermate,
}


def _pairwise_key(label_a: str, label_b: str) -> str:
    return f"{label_a}_vs_{label_b}"


def _compute_feature_enrichment(
    result: ClusteringResult,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> list[dict[str, Any]]:
    """Per-feature enrichment between two groups (mirrors single-project comparison.py)."""
    enrichment: list[dict[str, Any]] = []
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a < 5 or n_b < 5:
        return enrichment

    for fi, fname in enumerate(result.feature_names):
        vals_a = result.features_normalized[mask_a, fi]
        vals_b = result.features_normalized[mask_b, fi]
        vals_a = vals_a[np.isfinite(vals_a)]
        vals_b = vals_b[np.isfinite(vals_b)]
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        std_a = float(np.std(vals_a, ddof=1))
        std_b = float(np.std(vals_b, ddof=1))

        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * std_a ** 2 + (len(vals_b) - 1) * std_b ** 2)
            / max(len(vals_a) + len(vals_b) - 2, 1)
        )
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 1e-12 else 0.0

        median_all = float(np.median(
            result.features_normalized[np.isfinite(result.features_normalized[:, fi]), fi]
        ))
        shift = abs(min(mean_a, mean_b, median_all)) + 1.0
        log2fc = float(np.log2((mean_a + shift) / (mean_b + shift)))

        above_a = int(np.sum(vals_a > median_all))
        below_a = len(vals_a) - above_a
        above_b = int(np.sum(vals_b > median_all))
        below_b = len(vals_b) - above_b
        try:
            odds_r, p_fisher = fisher_exact([[above_a, below_a], [above_b, below_b]])
        except ValueError:
            odds_r, p_fisher = float("nan"), float("nan")

        try:
            u_stat, p_mw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        except ValueError:
            u_stat, p_mw = float("nan"), float("nan")

        enrichment.append({
            "feature": fname,
            f"mean_{label_a}": mean_a,
            f"mean_{label_b}": mean_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            f"std_{label_a}": std_a,
            f"std_{label_b}": std_b,
            "cohens_d": float(cohens_d),
            "log2_fold_change": log2fc,
            "odds_ratio": float(odds_r),
            "fisher_p_value": float(p_fisher),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p_value": float(p_mw),
            f"n_{label_a}": len(vals_a),
            f"n_{label_b}": len(vals_b),
            "n_a": len(vals_a),
            "n_b": len(vals_b),
            "direction": f"higher_in_{label_a}" if mean_a > mean_b else f"higher_in_{label_b}",
        })

    enrichment.sort(
        key=lambda x: x["mannwhitney_p_value"] if np.isfinite(x["mannwhitney_p_value"]) else 999
    )
    return enrichment


# ---------------------------------------------------------------------------
# Pairwise comparison helper
# ---------------------------------------------------------------------------

def _pairwise_comparison(
    result: ClusteringResult,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    label_a: str,
    label_b: str,
    *,
    display_a: str | None = None,
    display_b: str | None = None,
) -> dict[str, Any]:
    """Fisher (per-cluster), Mann-Whitney + enrichment (per-feature) between two groups."""
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    labels = result.cluster_labels
    disp_a = display_a or label_a
    disp_b = display_b or label_b

    comparison: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "display_a": disp_a,
        "display_b": disp_b,
        "n_a": n_a,
        "n_b": n_b,
    }

    if n_a < 5 or n_b < 5:
        comparison["note"] = "too few samples for statistical testing"
        return comparison

    # Per-cluster Fisher's exact test (enrichment in group A)
    per_cluster: dict[str, dict[str, Any]] = {}
    cluster_stats: dict[str, dict[str, Any]] = {}
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
        entry = {
            "a_in_cluster": a_in,
            "b_in_cluster": b_in,
            "a_frac": a_in / max(n_a, 1),
            "b_frac": b_in / max(n_b, 1),
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "comparison": f"{disp_a} vs {disp_b}",
        }
        per_cluster[str(c)] = entry
        cluster_stats[str(c)] = entry
    comparison["per_cluster"] = per_cluster
    comparison["cluster_stats"] = cluster_stats

    # Per-feature Mann-Whitney U (list + dict)
    per_feature: list[dict[str, Any]] = []
    per_feature_tests: dict[str, dict[str, Any]] = {}
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

        feat_entry = {
            "feature": fname,
            "comparison": f"{disp_a} vs {disp_b}",
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "cohens_d": float(cohens_d),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "effect_direction": f"higher_in_{label_a}" if mean_a > mean_b else f"higher_in_{label_b}",
            "direction": f"higher_in_{label_a}" if mean_a > mean_b else f"higher_in_{label_b}",
        }
        per_feature.append(feat_entry)
        per_feature_tests[fname] = feat_entry

    per_feature.sort(
        key=lambda x: x["p_value"] if np.isfinite(x["p_value"]) else 999
    )
    comparison["per_feature"] = per_feature
    comparison["per_feature_tests"] = per_feature_tests
    comparison["n_significant_features"] = sum(
        1 for f in per_feature if np.isfinite(f["p_value"]) and f["p_value"] < 0.05
    )

    comparison["feature_enrichment"] = _compute_feature_enrichment(
        result, mask_a, mask_b, label_a, label_b,
    )

    return comparison


def _run_pairwise_matrix(
    result: ClusteringResult,
    groups: list[str],
    masks: dict[str, np.ndarray],
    labels: dict[str, str],
    *,
    min_windows: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run all upper-triangular pairwise comparisons for ordered groups."""
    comparisons: dict[str, dict[str, Any]] = {}
    for ga, gb in combinations(groups, 2):
        mask_a, mask_b = masks[ga], masks[gb]
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        if n_a == 0 or n_b == 0:
            logger.info(
                "  Skip %s vs %s: n=%d vs %d (one group empty)",
                labels.get(ga, ga), labels.get(gb, gb), n_a, n_b,
            )
            continue
        if n_a < min_windows or n_b < min_windows:
            key = _pairwise_key(ga, gb)
            comparisons[key] = {
                "label_a": ga,
                "label_b": gb,
                "display_a": labels.get(ga, ga),
                "display_b": labels.get(gb, gb),
                "n_a": n_a,
                "n_b": n_b,
                "note": "too few samples for statistical testing",
            }
            logger.info(
                "  Skip %s vs %s: n=%d vs %d (< %d)",
                labels.get(ga, ga), labels.get(gb, gb), n_a, n_b, min_windows,
            )
            continue

        key = _pairwise_key(ga, gb)
        comparisons[key] = _pairwise_comparison(
            result, mask_a, mask_b, ga, gb,
            display_a=labels.get(ga, ga),
            display_b=labels.get(gb, gb),
        )
        n_sig = comparisons[key].get("n_significant_features", 0)
        logger.info(
            "  %s vs %s: %d vs %d windows, %d sig features (p<0.05)",
            labels.get(ga, ga), labels.get(gb, gb),
            comparisons[key]["n_a"], comparisons[key]["n_b"], n_sig,
        )
    return comparisons


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_experiments(result: ClusteringResult) -> dict[str, Any]:
    """Run all multi-project comparisons.

    Returns:
      - condition_groups: per-group cluster composition
      - pairwise_condition_groups: all C(4,2)=6 condition-group pairs
      - comparisons: alias for pairwise_condition_groups
      - per_cluster_by_experiment, ethograms, transition_matrices

    Role-condition pairwise comparisons live in compare_single_animal_experiments().
    """
    if len(result.metadata) == 0:
        return {"note": "no data"}

    t0 = time.monotonic()
    meta = result.metadata
    labels = result.cluster_labels
    n_total = len(labels)

    logger.info("Multi-project comparison: %d windows, %d clusters", n_total, result.n_clusters)

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
    # 2. All pairwise condition-group comparisons (upper triangle)
    # ---------------------------------------------------------------------------
    logger.info("Pairwise condition-group comparisons (N choose 2):")
    cg_masks = {g: groups == g for g in CONDITION_GROUP_ORDER}
    pairwise_condition_groups = _run_pairwise_matrix(
        result,
        CONDITION_GROUP_ORDER,
        cg_masks,
        CONDITION_GROUP_LABELS,
    )

    # Role-condition comparisons are only meaningful at the single-animal
    # level (pair-level role masks overlap: a SH_res_GH_int pair matches
    # both SH_resident and GH_intruder).  See compare_single_animal_experiments().
    comparisons = dict(pairwise_condition_groups)

    # ---------------------------------------------------------------------------
    # 4. Per-cluster enrichment by experiment
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
    # 5. Ethograms
    # ---------------------------------------------------------------------------
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

    from .transition_analysis import (
        compute_differential_transition_matrices,
        compute_transition_matrices_by_group,
    )

    transition_matrices = compute_transition_matrices_by_group(
        ethograms,
        result.n_clusters,
        group_field="condition_group",
        group_labels=CONDITION_GROUP_LABELS,
        exclude_groups={"unknown"},
    )
    differential_transition_matrices = compute_differential_transition_matrices(
        transition_matrices,
        CONDITION_GROUP_ORDER,
        exclude_groups={"unknown"},
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Multi-project comparison done in %.1fs (%d condition-group pairs)",
        elapsed, len(pairwise_condition_groups),
    )

    return {
        "condition_groups": condition_group_info,
        "pairwise_condition_groups": pairwise_condition_groups,
        "comparisons": comparisons,
        "per_cluster_by_experiment": per_cluster_by_exp,
        "ethograms": ethograms,
        "transition_matrices": transition_matrices,
        "differential_transition_matrices": differential_transition_matrices,
    }


# ---------------------------------------------------------------------------
# Single-animal role masks (focal always in mouse_a_* fields)
# ---------------------------------------------------------------------------

def _sa_mask_sh_resident(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "test_day"
        and m.mouse_a_role == "resident"
        and m.mouse_a_housing == "SH"
        for m in meta
    ])


def _sa_mask_sh_intruder(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "sh_intruder"
        and m.mouse_a_housing == "SH"
        for m in meta
    ])


def _sa_mask_gh_resident(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name in ("test_day", "sh_intruder")
        and m.mouse_a_role == "resident"
        and m.mouse_a_housing == "GH"
        for m in meta
    ])


def _sa_mask_gh_intruder(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([
        m.experiment_name == "test_day"
        and m.mouse_a_role == "intruder"
        and m.mouse_a_housing == "GH"
        for m in meta
    ])


def _sa_mask_gh_littermate(meta: list[WindowMetadata]) -> np.ndarray:
    return np.array([m.experiment_name == "hab" for m in meta])


SA_ROLE_MASK_BUILDERS: dict[str, Callable[[list[WindowMetadata]], np.ndarray]] = {
    "SH_resident": _sa_mask_sh_resident,
    "SH_intruder": _sa_mask_sh_intruder,
    "GH_resident": _sa_mask_gh_resident,
    "GH_intruder": _sa_mask_gh_intruder,
    "GH_littermate": _sa_mask_gh_littermate,
}


# ---------------------------------------------------------------------------
# Single-animal comparison
# ---------------------------------------------------------------------------

def compare_single_animal_experiments(result: ClusteringResult) -> dict[str, Any]:
    """Run single-animal comparisons using focal-animal role masks.

    In single-animal mode, each sample represents ONE focal animal whose
    identity is always in mouse_a_* fields. Role masks check only mouse_a_*.

    Returns same structure as compare_experiments() but without condition-group
    comparisons (those are pair-level concepts).
    """
    if len(result.metadata) == 0:
        return {"note": "no data"}

    t0 = time.monotonic()
    meta = result.metadata
    labels = result.cluster_labels
    n_total = len(labels)

    logger.info(
        "Single-animal comparison: %d samples, %d clusters",
        n_total, result.n_clusters,
    )

    # Role-condition pairwise comparisons
    logger.info("Pairwise single-animal role comparisons:")
    role_masks = {r: SA_ROLE_MASK_BUILDERS[r](meta) for r in ROLE_CONDITION_ORDER}
    for r, mask in role_masks.items():
        n = int(mask.sum())
        logger.info("  %s: %d samples", ROLE_CONDITION_LABELS.get(r, r), n)

    pairwise_role_conditions = _run_pairwise_matrix(
        result,
        ROLE_CONDITION_ORDER,
        role_masks,
        ROLE_CONDITION_LABELS,
    )

    # Ethograms
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
            "n_windows": len(indices_sorted),
            "start_frames": [meta[i].start_frame for i in indices_sorted],
            "cluster_labels": [int(labels[i]) for i in indices_sorted],
        }

    elapsed = time.monotonic() - t0
    logger.info(
        "Single-animal comparison done in %.1fs (%d role pairs)",
        elapsed, len(pairwise_role_conditions),
    )

    return {
        "pairwise_role_conditions": pairwise_role_conditions,
        "comparisons": pairwise_role_conditions,
        "ethograms": ethograms,
    }
