"""Dedicated UMAP plotting: highlight plots, per-condition/role UMAPs.

All UMAP plots are saved under plots/umap/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .clustering_pipeline import ClusteringResult
from .plots import _add_legend_on_plot, _cluster_cmap, _DPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_CONDITION_COLORS = {
    "GH_littermate": "#4CAF50",
    "SH_res_GH_int": "#E05555",
    "GH_res_GH_int": "#2196F3",
    "GH_res_SH_int": "#FF9800",
    "unknown": "#888888",
}

_ROLE_COLORS = {
    "SH_resident": "#E05555",
    "SH_intruder": "#FF6B6B",
    "GH_resident": "#2196F3",
    "GH_intruder": "#64B5F6",
    "GH_littermate": "#4CAF50",
}


# ---------------------------------------------------------------------------
# UMAP by role (single-animal mode)
# ---------------------------------------------------------------------------

def plot_umap_by_role(result: ClusteringResult, outfile: Path) -> None:
    """UMAP colored by the 5 single-animal role categories.

    Replaces plot_umap_by_condition_group for single-animal data.
    """
    from .multi_project_comparison import (
        _assign_sa_role,
        ROLE_CONDITION_LABELS,
        ROLE_CONDITION_ORDER,
    )

    if result.embedding_2d.shape[0] == 0:
        return

    roles = np.array([_assign_sa_role(m) for m in result.metadata])
    present = [r for r in ROLE_CONDITION_ORDER if (roles == r).any()]
    if not present:
        present = sorted(set(roles))

    fig, ax = plt.subplots(figsize=(8, 7))
    for role in present:
        mask = roles == role
        label = ROLE_CONDITION_LABELS.get(role, role)
        color = _ROLE_COLORS.get(role, "#888888")
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[color],
            s=2,
            alpha=0.4,
            label=f"{label} (n={int(mask.sum())})",
            rasterized=True,
        )

    # Show "unknown" if present
    unk_mask = roles == "unknown"
    if unk_mask.any():
        ax.scatter(
            result.embedding_2d[unk_mask, 0],
            result.embedding_2d[unk_mask, 1],
            c=["#888888"],
            s=2,
            alpha=0.2,
            label=f"Unknown (n={int(unk_mask.sum())})",
            rasterized=True,
        )

    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("UMAP by Role")
    _add_legend_on_plot(ax, fontsize=7, markerscale=4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Single-highlight UMAP
# ---------------------------------------------------------------------------

def plot_umap_highlight(
    result: ClusteringResult,
    mask: np.ndarray,
    label: str,
    color: str,
    outfile: Path,
    *,
    title: str | None = None,
) -> None:
    """UMAP with highlighted points colored and background points gray."""
    if result.embedding_2d.shape[0] == 0:
        return

    bg_mask = ~mask
    fig, ax = plt.subplots(figsize=(8, 7))

    # Background (gray)
    if bg_mask.any():
        ax.scatter(
            result.embedding_2d[bg_mask, 0],
            result.embedding_2d[bg_mask, 1],
            c="#CCCCCC",
            s=1,
            alpha=0.15,
            rasterized=True,
        )

    # Highlighted
    n_highlight = int(mask.sum())
    if n_highlight > 0:
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[color],
            s=4,
            alpha=0.5,
            label=f"{label} (n={n_highlight})",
            rasterized=True,
        )

    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title(title or label)
    _add_legend_on_plot(ax, fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Generate all highlight plots
# ---------------------------------------------------------------------------

def generate_umap_highlight_plots(
    result: ClusteringResult,
    umap_dir: Path,
) -> None:
    """Generate highlight UMAP plots.

    - Pair-level mode: per condition group + per cluster
    - Single-animal mode: per role + per cluster
    """
    from .multi_project_comparison import (
        _assign_condition_group,
        CONDITION_GROUP_LABELS,
        CONDITION_GROUP_ORDER,
        ROLE_CONDITION_LABELS,
        ROLE_CONDITION_ORDER,
        SA_ROLE_MASK_BUILDERS,
    )

    umap_dir.mkdir(parents=True, exist_ok=True)

    # Detect single-animal mode
    is_single_animal = (
        result.metadata
        and getattr(result.metadata[0], "focal_side", "") != ""
    )

    if is_single_animal:
        # Per role (single-animal: roles are the meaningful categories)
        for role in ROLE_CONDITION_ORDER:
            builder = SA_ROLE_MASK_BUILDERS.get(role)
            if builder is None:
                continue
            mask = builder(result.metadata)
            if not mask.any():
                continue
            label = ROLE_CONDITION_LABELS.get(role, role)
            color = _ROLE_COLORS.get(role, "#888888")
            plot_umap_highlight(
                result, mask, label, color,
                umap_dir / f"highlight_{role}.png",
                title=f"UMAP highlight: {label}",
            )
    else:
        # Per condition group (pair-level: condition groups are meaningful)
        groups = np.array([_assign_condition_group(m) for m in result.metadata])
        for g in CONDITION_GROUP_ORDER:
            mask = groups == g
            if not mask.any():
                continue
            label = CONDITION_GROUP_LABELS.get(g, g)
            color = _CONDITION_COLORS.get(g, "#888888")
            plot_umap_highlight(
                result, mask, label, color,
                umap_dir / f"highlight_{g}.png",
                title=f"UMAP highlight: {label}",
            )

    # Per cluster (both modes)
    for c in range(result.n_clusters):
        mask = result.cluster_labels == c
        if not mask.any():
            continue
        colors = _cluster_cmap(result.n_clusters)
        plot_umap_highlight(
            result, mask, f"Cluster {c}", colors[c],
            umap_dir / f"highlight_cluster_{c}.png",
            title=f"UMAP highlight: Cluster {c} (n={int(mask.sum())})",
        )

    logger.info("UMAP highlight plots saved to %s", umap_dir)


# ---------------------------------------------------------------------------
# Consolidation: all UMAP plots into one subfolder
# ---------------------------------------------------------------------------

def generate_all_umap_plots(
    result: ClusteringResult,
    comparison: dict[str, Any],
    plots_dir: Path,
    *,
    pre_correction_features: np.ndarray | None = None,
) -> None:
    """Generate all UMAP plots into plots/umap/ subfolder."""
    umap_dir = plots_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    # Detect single-animal mode
    is_single_animal = (
        result.metadata
        and getattr(result.metadata[0], "focal_side", "") != ""
    )

    # Core UMAP plots
    from .plots import plot_umap_by_cluster, plot_umap_by_housing
    plot_umap_by_cluster(result, umap_dir / "umap_by_cluster.png")
    if not is_single_animal:
        plot_umap_by_housing(result, umap_dir / "umap_by_housing.png")

    # Mode-specific category UMAP
    if is_single_animal:
        plot_umap_by_role(result, umap_dir / "umap_by_role.png")
    else:
        try:
            from .plots_multi import plot_umap_by_condition_group
            plot_umap_by_condition_group(result, umap_dir / "umap_by_condition_group.png")
        except Exception as e:
            logger.warning("Condition group UMAP failed: %s", e)

    # Experiment and batch UMAPs (meaningful in both modes)
    try:
        from .plots_multi import (
            plot_umap_by_experiment,
            plot_umap_by_batch,
            plot_batch_correction_before_after,
        )
        plot_umap_by_experiment(result, umap_dir / "umap_by_experiment.png")
        plot_umap_by_batch(result, umap_dir / "umap_by_batch.png")
        if pre_correction_features is not None:
            plot_batch_correction_before_after(
                pre_correction_features,
                result.features_normalized,
                result.metadata,
                umap_dir / "batch_correction_before_after.png",
            )
    except Exception as e:
        logger.warning("Multi-project UMAP plots failed: %s", e)

    # Highlight plots
    try:
        generate_umap_highlight_plots(result, umap_dir)
    except Exception as e:
        logger.warning("UMAP highlight plots failed: %s", e)

    logger.info("All UMAP plots saved to %s", umap_dir)
