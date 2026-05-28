"""Boxplot visualizations across experimental conditions.

Produces:
  - Per-feature boxplots across condition groups
  - Cluster composition boxplots per condition
  - Locomotion vs paired paired-boxplots with connecting lines
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

from .clustering_pipeline import ClusteringResult
from .multi_project_comparison import _assign_condition_group, CONDITION_GROUP_LABELS

logger = logging.getLogger(__name__)

_DPI = 140

# Ordered groups for x-axis
_GROUP_ORDER = [
    "GH_littermate",
    "SH_res_GH_int",
    "GH_res_GH_int",
    "GH_res_SH_int",
]

_GROUP_COLORS = {
    "GH_littermate": "#4CAF50",
    "SH_res_GH_int": "#E05555",
    "GH_res_GH_int": "#2196F3",
    "GH_res_SH_int": "#FF9800",
}

# Key features to plot (subset for readability)
_KEY_FEATURES = [
    "a_speed_mean", "b_speed_mean",
    "a_area_mean", "b_area_mean",
    "a_eccentricity_mean", "b_eccentricity_mean",
    "distance_mean", "distance_std",
    "overlap_fraction",
    "close_proximity_fraction",
    "chase_score",
    "heading_alignment_mean",
    "relative_speed_mean",
]


# ---------------------------------------------------------------------------
# Per-feature boxplots across conditions
# ---------------------------------------------------------------------------

def plot_condition_boxplots(
    result: ClusteringResult,
    comparison: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate per-feature boxplots across condition groups."""
    if result.features_normalized.shape[0] == 0:
        return

    groups = np.array([_assign_condition_group(m) for m in result.metadata])
    present_groups = [g for g in _GROUP_ORDER if g in set(groups)]

    if len(present_groups) < 2:
        logger.warning("Fewer than 2 condition groups present — skipping boxplots")
        return

    # Select features to plot
    feature_names = list(result.feature_names)
    features_to_plot = [f for f in _KEY_FEATURES if f in feature_names]
    if not features_to_plot:
        features_to_plot = feature_names[:12]

    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for fi, fname in enumerate(features_to_plot):
        ax = axes[fi]
        col_idx = feature_names.index(fname)
        data = []
        labels = []
        colors = []

        for g in present_groups:
            mask = groups == g
            vals = result.features_normalized[mask, col_idx]
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            labels.append(CONDITION_GROUP_LABELS.get(g, g))
            colors.append(_GROUP_COLORS.get(g, "#888888"))

        bp = ax.boxplot(
            data,
            labels=None,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(fname, fontsize=9)
        ax.set_xticks(range(1, len(present_groups) + 1))
        ax.set_xticklabels(
            [CONDITION_GROUP_LABELS.get(g, g) for g in present_groups],
            rotation=30, ha="right", fontsize=7,
        )
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions by Condition Group", fontsize=13, y=1.01)
    fig.tight_layout()
    outfile = plots_dir / "condition_feature_boxplots.png"
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Cluster composition per condition
# ---------------------------------------------------------------------------

def plot_cluster_composition(
    result: ClusteringResult,
    comparison: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Stacked bar chart of cluster composition per condition group."""
    if result.features_normalized.shape[0] == 0:
        return

    groups = np.array([_assign_condition_group(m) for m in result.metadata])
    labels = result.cluster_labels
    present_groups = [g for g in _GROUP_ORDER if g in set(groups)]
    n_clusters = result.n_clusters

    if len(present_groups) < 2 or n_clusters == 0:
        return

    # Compute fractions
    fractions = np.zeros((len(present_groups), n_clusters))
    for gi, g in enumerate(present_groups):
        mask = groups == g
        n_g = mask.sum()
        if n_g == 0:
            continue
        for c in range(n_clusters):
            fractions[gi, c] = np.sum(mask & (labels == c)) / n_g

    # Stacked bar chart
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo
    cluster_colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(present_groups))
    bottom = np.zeros(len(present_groups))

    for c in range(n_clusters):
        ax.bar(
            x, fractions[:, c], bottom=bottom,
            color=cluster_colors[c], label=f"Cluster {c}",
            edgecolor="white", linewidth=0.3,
        )
        bottom += fractions[:, c]

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_GROUP_LABELS.get(g, g) for g in present_groups],
        rotation=25, ha="right", fontsize=9,
    )
    ax.set_ylabel("Fraction of Windows")
    ax.set_title("Cluster Composition by Condition Group")
    ax.legend(
        fontsize=7, ncol=max(1, n_clusters // 5),
        bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    outfile = plots_dir / "cluster_composition_by_condition.png"
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)

    # Also make per-cluster boxplot (fraction per video within each condition)
    _plot_per_video_cluster_fractions(result, groups, present_groups, plots_dir)


def _plot_per_video_cluster_fractions(
    result: ClusteringResult,
    groups: np.ndarray,
    present_groups: list[str],
    plots_dir: Path,
) -> None:
    """Per-cluster boxplot: fraction of windows in each cluster, per video, grouped by condition."""
    from collections import defaultdict

    labels = result.cluster_labels
    n_clusters = result.n_clusters
    metadata = result.metadata

    # Group windows by video
    vid_groups: defaultdict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        vid_groups[m.video_id].append(i)

    # For each cluster, compute per-video fraction, grouped by condition
    n_cols = min(4, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo

    for c in range(n_clusters):
        ax = axes[c]
        data = []
        colors = []

        for g in present_groups:
            fracs = []
            for vid_id, indices in vid_groups.items():
                # Check if this video belongs to this condition group
                g_vid = _assign_condition_group(metadata[indices[0]])
                if g_vid != g:
                    continue
                n_vid = len(indices)
                n_in_cluster = sum(1 for i in indices if labels[i] == c)
                fracs.append(n_in_cluster / n_vid)
            data.append(fracs if fracs else [0.0])
            colors.append(_GROUP_COLORS.get(g, "#888888"))

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.6, showfliers=True,
            flierprops=dict(marker=".", markersize=3),
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"Cluster {c}", fontsize=9)
        ax.set_xticks(range(1, len(present_groups) + 1))
        ax.set_xticklabels(
            [CONDITION_GROUP_LABELS.get(g, g) for g in present_groups],
            rotation=35, ha="right", fontsize=6,
        )
        ax.set_ylabel("Fraction", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-Video Cluster Fraction by Condition", fontsize=12, y=1.01)
    fig.tight_layout()
    outfile = plots_dir / "cluster_fraction_boxplots_by_condition.png"
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Locomotion vs paired comparison plots
# ---------------------------------------------------------------------------

def plot_locomotion_paired(
    locomotion_results: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Paired scatter plots: locomotion (alone) vs paired test, per housing subgroup."""
    paired = locomotion_results.get("paired_comparisons", [])
    if not paired:
        logger.info("No paired comparisons for locomotion plots")
        return

    loc_arr = np.array(locomotion_results.get("loc_features", []), dtype=np.float64)
    paired_arr = np.array(locomotion_results.get("paired_features", []), dtype=np.float64)
    housing = locomotion_results.get("housing", [])
    feature_names = locomotion_results.get("feature_names", [])

    for comp in paired:
        key = comp.get("key", "comparison")
        label_a = comp.get("label_a", "Alone")
        label_b = comp.get("label_b", "Paired")

        if key == "SH_resident_locomotion_vs_paired":
            mask = np.array([h == "SH" for h in housing])
        elif key == "GH_resident_locomotion_vs_paired":
            mask = np.array([h == "GH" for h in housing])
        else:
            mask = np.ones(len(housing), dtype=bool) if housing else np.array([], dtype=bool)

        if loc_arr.size == 0 or not mask.any():
            continue

        loc_sub = loc_arr[mask]
        paired_sub = paired_arr[mask]
        features_to_plot = feature_names[:12] if feature_names else []

        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = max(1, (n_features + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for fi, fname in enumerate(features_to_plot):
            if fname not in feature_names:
                continue
            col = feature_names.index(fname)
            ax = axes[fi]
            alone = loc_sub[:, col]
            paired_vals = paired_sub[:, col]
            valid = np.isfinite(alone) & np.isfinite(paired_vals)
            alone_v = alone[valid]
            paired_v = paired_vals[valid]

            for a, p in zip(alone_v, paired_v):
                ax.plot([0, 1], [a, p], color="#888888", linewidth=0.5, alpha=0.5)
            ax.scatter(np.zeros(len(alone_v)), alone_v, c="#4CAF50", s=20, zorder=3, label=label_a)
            ax.scatter(np.ones(len(paired_v)), paired_v, c="#E05555", s=20, zorder=3, label=label_b)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Locomotion", "Paired test"], fontsize=9)
            ax.set_title(fname, fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            if fi == 0:
                ax.legend(fontsize=7)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        title = f"{label_a} vs {label_b} (n={int(mask.sum())} mouse-camera pairs)"
        fig.suptitle(title, fontsize=12, y=1.01)
        fig.tight_layout()
        outfile = plots_dir / f"locomotion_{key}.png"
        fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", outfile)

        _plot_locomotion_effect_bars(comp, plots_dir / f"locomotion_{key}_effect_sizes.png")

    _plot_locomotion_group_boxplots(locomotion_results, plots_dir)


def _plot_locomotion_effect_bars(comp: dict[str, Any], outfile: Path) -> None:
    """Bar chart of mean difference (paired - locomotion) per feature with significance."""
    per_feature = comp.get("per_feature", [])
    if not per_feature:
        return

    feats = per_feature[:12]
    names = [f["feature"] for f in feats]
    diffs = [f["mean_paired"] - f["mean_alone"] for f in feats]
    pvals = [f.get("wilcoxon_p_value", float("nan")) for f in feats]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y = np.arange(len(names))
    colors = ["#E05555" if d > 0 else "#5588DD" for d in diffs]
    ax.barh(y, diffs, color=colors, alpha=0.75, height=0.7)
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        if np.isfinite(p) and p < 0.05:
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            offset = 0.02 if d >= 0 else -0.02
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, i, marker, ha=ha, va="center", fontsize=9, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Mean difference (paired test − locomotion)")
    ax.set_title(comp.get("label_b", "Paired") + " vs " + comp.get("label_a", "Locomotion"))
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_locomotion_group_boxplots(
    locomotion_results: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Wilcoxon effect summary for SH vs GH locomotion-only (unpaired)."""
    unpaired = locomotion_results.get("unpaired_comparisons", {})
    sh_vs_gh = locomotion_results.get("SH_vs_GH_locomotion")
    if sh_vs_gh:
        unpaired = {**unpaired, "SH_vs_GH_locomotion": {"per_feature": sh_vs_gh, "label_a": "SH", "label_b": "GH"}}
    if not unpaired:
        return

    for comp_name, comp in unpaired.items():
        per_feature = comp.get("per_feature", [])
        if not per_feature:
            continue

        features = per_feature[:12]
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for fi, feat in enumerate(features):
            ax = axes[fi]
            data_a = [feat.get("mean_a", 0)]
            data_b = [feat.get("mean_b", 0)]
            bp = ax.boxplot(
                [data_a, data_b],
                labels=[comp.get("label_a", "A"), comp.get("label_b", "B")],
                patch_artist=True,
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#4CAF50")
            bp["boxes"][1].set_facecolor("#E05555")
            ax.set_title(feat["feature"], fontsize=9)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(comp_name.replace("_", " ").title(), fontsize=12)
        fig.tight_layout()
        outfile = plots_dir / f"locomotion_{comp_name}.png"
        fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", outfile)
