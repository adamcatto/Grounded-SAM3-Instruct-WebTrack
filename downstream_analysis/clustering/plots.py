"""Visualization for the clustering pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
from ..figio import save_figure  # noqa: E402

from .clustering_pipeline import ClusteringResult
from .dataset import housing_condition

logger = logging.getLogger(__name__)

_DPI = 140


def _legend_layout_rect(n_items: int, ncol: int = 1) -> float:
    """Right edge of the axes area (figure coords) reserved for an external legend."""
    nrows = max(1, (n_items + ncol - 1) // ncol)
    if nrows >= 16:
        return 0.52
    if nrows >= 10:
        return 0.60
    if nrows >= 6:
        return 0.67
    return 0.74


def _add_legend_right_of_axes(
    ax: plt.Axes,
    *,
    ncol: int = 1,
    fontsize: float = 8,
    markerscale: float = 3,
    n_items: int | None = None,
) -> None:
    """Place legend to the right of the scatter axes (not over the points)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = n_items if n_items is not None else len(handles)
    ax.legend(
        fontsize=fontsize,
        markerscale=markerscale,
        ncol=max(1, ncol),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    rect_right = _legend_layout_rect(n, ncol)
    ax.figure.tight_layout(rect=[0, 0, rect_right, 1])


def _add_legend_on_plot(
    ax: plt.Axes,
    *,
    fontsize: float = 8,
    markerscale: float = 3,
    alpha: float = 0.85,
    min_fontsize: float = 5,
) -> None:
    """Place legend ON the plot in the corner with fewest data points.

    Tries all 4 corners, estimates the legend bounding box in data space,
    picks the corner overlapping the fewest scatter points.  If the best
    corner still covers >5 % of points, shrinks the font.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    n_items = len(handles)

    # Collect all plotted data points
    all_x, all_y = [], []
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if offsets is not None and len(offsets) > 0:
            arr = np.asarray(offsets)
            all_x.append(arr[:, 0])
            all_y.append(arr[:, 1])
    if not all_x:
        ax.legend(fontsize=fontsize, markerscale=markerscale, loc="upper right",
                  framealpha=alpha)
        return

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    n_points = len(all_x)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Heuristic legend size as fraction of axes
    legend_w_frac = 0.30
    legend_h_frac = min(0.04 * n_items, 0.50)
    legend_w = legend_w_frac * x_range
    legend_h = legend_h_frac * y_range

    corners = {
        "upper right": (1, 1),
        "upper left": (0, 1),
        "lower left": (0, 0),
        "lower right": (1, 0),
    }

    best_corner = "upper right"
    best_count = n_points + 1

    for loc_name, (fx, fy) in corners.items():
        x_min = xlim[0] if fx == 0 else xlim[1] - legend_w
        x_max = xlim[0] + legend_w if fx == 0 else xlim[1]
        y_min = ylim[0] if fy == 0 else ylim[1] - legend_h
        y_max = ylim[0] + legend_h if fy == 0 else ylim[1]

        count = int(np.sum(
            (all_x >= x_min) & (all_x <= x_max)
            & (all_y >= y_min) & (all_y <= y_max)
        ))
        if count < best_count:
            best_count = count
            best_corner = loc_name

    # Shrink font if still covering >5% of points
    final_fontsize = fontsize
    if best_count > 0.05 * n_points and fontsize > min_fontsize:
        final_fontsize = max(min_fontsize, fontsize * 0.6)

    ax.legend(
        fontsize=final_fontsize,
        markerscale=markerscale,
        loc=best_corner,
        framealpha=alpha,
        edgecolor="gray",
    )


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _cluster_cmap(n: int):
    """Return a list of n distinct colors for clusters."""
    if n <= 10:
        cmap = plt.cm.tab10
    elif n <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


_HOUSING_COLORS = {
    "isolated": "#E05555",
    "group-housed": "#5588DD",
    "unknown": "#888888",
}

_ITYPE_COLORS = {
    "group-housed+isolated": "#B06DD8",
    "group-housed+group-housed": "#5588DD",
    "isolated+isolated": "#E05555",
    "single_animal": "#2CA02C",
}


# ---------------------------------------------------------------------------
# UMAP scatter
# ---------------------------------------------------------------------------

def plot_umap_by_cluster(result: ClusteringResult, outfile: Path) -> None:
    """UMAP scatter colored by cluster label."""
    if result.embedding_2d.shape[0] == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = _cluster_cmap(result.n_clusters)
    for c in range(result.n_clusters):
        mask = result.cluster_labels == c
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[colors[c]],
            s=3,
            alpha=0.4,
            label=f"C{c} (n={int(np.sum(mask))})",
            rasterized=True,
        )
    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("Behavioral clusters")
    _add_legend_on_plot(ax, fontsize=6, markerscale=3)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


def plot_umap_by_housing(result: ClusteringResult, outfile: Path) -> None:
    """UMAP scatter colored by interaction type."""
    if result.embedding_2d.shape[0] == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    itypes = [m.interaction_type for m in result.metadata]
    unique_types = sorted(set(itypes))
    for itype in unique_types:
        mask = np.array([it == itype for it in itypes])
        color = _ITYPE_COLORS.get(itype, "#888888")
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[color],
            s=3,
            alpha=0.4,
            label=f"{itype} (n={int(np.sum(mask))})",
            rasterized=True,
        )
    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("Interaction types")
    _add_legend_on_plot(ax, fontsize=7, markerscale=3)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Cluster composition
# ---------------------------------------------------------------------------

def plot_cluster_composition(
    result: ClusteringResult,
    comparison: dict[str, Any],
    outfile: Path,
) -> None:
    """Stacked bar chart: interaction type composition per cluster."""
    if result.n_clusters == 0:
        return
    itype_dist = comparison.get("interaction_type_distribution", {})
    all_itypes = sorted({
        t for counts in itype_dist.values() for t in counts
    })
    if not all_itypes:
        return

    fig, ax = plt.subplots(figsize=(max(6, result.n_clusters * 0.6), 5))
    x = np.arange(result.n_clusters)
    bottoms = np.zeros(result.n_clusters)

    for itype in all_itypes:
        heights = []
        for c in range(result.n_clusters):
            counts = itype_dist.get(str(c), {})
            total = sum(counts.values()) or 1
            heights.append(counts.get(itype, 0) / total)
        heights = np.array(heights)
        color = _ITYPE_COLORS.get(itype, "#888888")
        ax.bar(x, heights, bottom=bottoms, color=color, label=itype, width=0.7)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in range(result.n_clusters)], fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("Interaction type composition per cluster")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Feature heatmap
# ---------------------------------------------------------------------------

def plot_feature_heatmap(result: ClusteringResult, outfile: Path) -> None:
    """Cluster x Feature heatmap of mean normalized feature values."""
    if result.n_clusters == 0:
        return
    n_feats = result.features_normalized.shape[1]
    mat = np.zeros((result.n_clusters, n_feats), dtype=np.float64)
    for c in range(result.n_clusters):
        mask = result.cluster_labels == c
        if np.sum(mask) > 0:
            mat[c] = result.features_normalized[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(10, n_feats * 0.35), max(4, result.n_clusters * 0.5)))
    vmax = max(abs(mat.min()), abs(mat.max()), 1.0)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(result.n_clusters))
    ax.set_yticklabels([f"C{c}" for c in range(result.n_clusters)], fontsize=8)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(result.feature_names, rotation=70, ha="right", fontsize=6)
    ax.set_title("Mean feature values per cluster (z-scored)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Ethograms
# ---------------------------------------------------------------------------

def plot_ethogram(
    result: ClusteringResult,
    ethogram_data: dict[str, Any],
    video_id: str,
    outfile: Path,
) -> None:
    """Horizontal timeline strip showing cluster labels over time for one video."""
    starts = ethogram_data["start_frames"]
    clabels = ethogram_data["cluster_labels"]
    if not starts:
        return

    colors = _cluster_cmap(result.n_clusters)
    fig, ax = plt.subplots(figsize=(14, 1.5))

    for i in range(len(starts)):
        s = starts[i]
        w = result.metadata[0].window_size if result.metadata else 90
        c = clabels[i]
        ax.barh(0, w, left=s, height=0.8, color=colors[c], edgecolor="none")

    ax.set_xlim(starts[0], starts[-1] + w)
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    vname = ethogram_data.get("video_name", video_id)
    itype = ethogram_data.get("interaction_type", "")
    ax.set_title(f"{vname[:60]} ({itype})", fontsize=9)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature violins per cluster
# ---------------------------------------------------------------------------

def plot_feature_violins(
    result: ClusteringResult,
    feature_indices: list[int],
    outfile: Path,
) -> None:
    """Violin plots for selected features, one panel per feature, split by cluster."""
    if result.n_clusters == 0 or not feature_indices:
        return

    n_panels = len(feature_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    colors = _cluster_cmap(result.n_clusters)

    for ax, fi in zip(axes, feature_indices):
        data = []
        positions = []
        for c in range(result.n_clusters):
            mask = result.cluster_labels == c
            vals = result.features_normalized[mask, fi]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data.append(vals)
                positions.append(c)

        if data:
            parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
            for i, pc in enumerate(parts.get("bodies", [])):
                if i < len(positions):
                    pc.set_facecolor(colors[positions[i]])
                    pc.set_alpha(0.6)

        fname = result.feature_names[fi] if fi < len(result.feature_names) else f"feat_{fi}"
        ax.set_title(fname, fontsize=8)
        ax.set_xlabel("Cluster")

    fig.suptitle("Feature distributions by cluster", fontsize=10)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Enrichment bar chart
# ---------------------------------------------------------------------------

def plot_enrichment_bars(
    result: ClusteringResult,
    comparison: dict[str, Any],
    outfile: Path,
) -> None:
    """Bar chart of isolation enrichment odds ratio per cluster."""
    per_cluster = comparison.get("per_cluster", {})
    if not per_cluster:
        return

    clusters = sorted(per_cluster.keys(), key=int)
    odds = [per_cluster[c]["isolated_enrichment"]["odds_ratio"] for c in clusters]
    pvals = [per_cluster[c]["isolated_enrichment"]["p_value"] for c in clusters]

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 0.6), 4))
    x = np.arange(len(clusters))
    bars = ax.bar(x, odds, color="#E05555", alpha=0.7, width=0.6)

    # Mark significant enrichments
    for i, (o, p) in enumerate(zip(odds, pvals)):
        if np.isfinite(p) and p < 0.05:
            ax.text(i, o + 0.05, "*", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
    ax.set_ylabel("Odds ratio (isolated enrichment)")
    ax.set_title("Isolation enrichment per cluster (* = p < 0.05)")
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Feature enrichment (isolated vs group-only)
# ---------------------------------------------------------------------------

def plot_feature_enrichment(
    comparison: dict[str, Any],
    outfile: Path,
) -> None:
    """Horizontal bar chart of per-feature Cohen's d (isolated vs group-only).

    Features sorted by effect size. Bars colored by direction, stars for significance.
    """
    enrichment = comparison.get("feature_enrichment", [])
    if not enrichment:
        return

    # Sort by absolute Cohen's d
    enrichment_sorted = sorted(enrichment, key=lambda x: abs(x["cohens_d"]))

    names = [e["feature"] for e in enrichment_sorted]
    cohens = [e["cohens_d"] for e in enrichment_sorted]
    pvals = [e["mannwhitney_p_value"] for e in enrichment_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y = np.arange(len(names))
    colors = ["#E05555" if d > 0 else "#5588DD" for d in cohens]
    ax.barh(y, cohens, color=colors, alpha=0.75, height=0.7)

    # Significance markers
    for i, (d, p) in enumerate(zip(cohens, pvals)):
        if np.isfinite(p) and p < 0.05:
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            offset = 0.02 if d >= 0 else -0.02
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, i, marker, ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Cohen's d (positive = higher in isolated)")
    ax.set_title("Feature enrichment: isolated vs group-only pairs")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E05555", alpha=0.75, label="Higher in isolated"),
        Patch(facecolor="#5588DD", alpha=0.75, label="Higher in group-only"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    result: ClusteringResult,
    comparison: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate all clustering plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Basic UMAP plots (multi-project pipeline generates enhanced versions via
    # plots_umap.generate_all_umap_plots, but single-project still needs these)
    umap_dir = plots_dir / "umap"
    plot_umap_by_cluster(result, umap_dir / "umap_by_cluster.png")
    # In single-animal mode there is no pair, so all housing / interaction /
    # enrichment plots are meaningless (and enrichment plots assume the
    # "isolated_enrichment" comparison keys). Skip them.
    is_single_animal = (
        result.metadata
        and getattr(result.metadata[0], "focal_side", "") != ""
    )
    if not is_single_animal:
        plot_umap_by_housing(result, umap_dir / "umap_by_housing.png")
        plot_cluster_composition(result, comparison, plots_dir / "cluster_composition.png")

    plot_feature_heatmap(result, plots_dir / "feature_heatmap.png")
    if not is_single_animal:
        plot_enrichment_bars(result, comparison, plots_dir / "enrichment_bars.png")
        plot_feature_enrichment(comparison, plots_dir / "feature_enrichment.png")

    # Violin plots for top discriminative features (first 8)
    top_features = list(range(min(8, result.features_normalized.shape[1])))
    if top_features:
        plot_feature_violins(result, top_features, plots_dir / "feature_violins.png")

    # Ethograms per video
    ethograms = comparison.get("ethograms", {})
    ethogram_dir = plots_dir / "ethograms"
    for vid_id, ethogram_data in ethograms.items():
        plot_ethogram(
            result, ethogram_data, vid_id,
            ethogram_dir / f"ethogram_{vid_id}.png",
        )

    try:
        from .plots_transitions import generate_transition_plots
        generate_transition_plots(result, comparison, plots_dir)
    except Exception as e:
        logger.warning("Transition plots failed: %s", e)

    logger.info("All plots saved to %s", plots_dir)
