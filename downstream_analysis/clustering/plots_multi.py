"""Multi-project UMAP visualization variants.

Provides UMAP plots colored by experiment, condition group, batch,
and before/after batch correction comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ..figio import save_figure  # noqa: E402

from .clustering_pipeline import ClusteringResult
from .dataset import WindowMetadata
from .multi_project_comparison import _assign_condition_group, CONDITION_GROUP_LABELS
from .plots import _add_legend_on_plot, _add_legend_right_of_axes

logger = logging.getLogger(__name__)

_DPI = 140

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_EXPERIMENT_COLORS = {
    "hab": "#4CAF50",
    "test_day": "#2196F3",
    "sh_intruder": "#FF9800",
    "locomotion": "#9C27B0",
    "": "#888888",
}

_CONDITION_COLORS = {
    "GH_littermate": "#4CAF50",
    "SH_res_GH_int": "#E05555",
    "GH_res_GH_int": "#2196F3",
    "GH_res_SH_int": "#FF9800",
    "unknown": "#888888",
}


# ---------------------------------------------------------------------------
# UMAP by experiment
# ---------------------------------------------------------------------------

def plot_umap_by_experiment(result: ClusteringResult, outfile: Path) -> None:
    """UMAP scatter colored by experiment name."""
    if result.embedding_2d.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    experiments = sorted(set(m.experiment_name for m in result.metadata))

    for exp in experiments:
        mask = np.array([m.experiment_name == exp for m in result.metadata])
        color = _EXPERIMENT_COLORS.get(exp, "#888888")
        label = exp if exp else "unknown"
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[color],
            s=3,
            alpha=0.3,
            label=f"{label} (n={int(mask.sum())})",
            rasterized=True,
        )

    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("UMAP by Experiment")
    _add_legend_on_plot(ax, fontsize=7, markerscale=4)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# UMAP by condition group
# ---------------------------------------------------------------------------

def plot_umap_by_condition_group(result: ClusteringResult, outfile: Path) -> None:
    """UMAP scatter colored by condition group."""
    if result.embedding_2d.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    groups = [_assign_condition_group(m) for m in result.metadata]
    unique_groups = sorted(set(groups))

    for g in unique_groups:
        mask = np.array([grp == g for grp in groups])
        color = _CONDITION_COLORS.get(g, "#888888")
        label = CONDITION_GROUP_LABELS.get(g, g)
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[color],
            s=3,
            alpha=0.3,
            label=f"{label} (n={int(mask.sum())})",
            rasterized=True,
        )

    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("UMAP by Condition Group")
    _add_legend_on_plot(ax, fontsize=7, markerscale=4)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# UMAP by batch
# ---------------------------------------------------------------------------

def plot_umap_by_batch(result: ClusteringResult, outfile: Path) -> None:
    """UMAP scatter colored by batch_id — useful for verifying batch correction."""
    if result.embedding_2d.shape[0] == 0:
        return

    batches = sorted(set(m.batch_id for m in result.metadata))
    n_batches = len(batches)

    if n_batches <= 10:
        cmap = plt.cm.tab10
    elif n_batches <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo
    batch_colors = {b: cmap(i / max(n_batches - 1, 1)) for i, b in enumerate(batches)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for batch in batches:
        mask = np.array([m.batch_id == batch for m in result.metadata])
        ax.scatter(
            result.embedding_2d[mask, 0],
            result.embedding_2d[mask, 1],
            c=[batch_colors[batch]],
            s=2,
            alpha=0.3,
            label=f"{batch} ({int(mask.sum())})",
            rasterized=True,
        )

    ax.set_xlabel(f"{result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{result.embedding_method.upper()} 2")
    ax.set_title("UMAP by Batch")
    ncol = max(1, n_batches // 10)
    _add_legend_right_of_axes(
        ax, ncol=ncol, fontsize=5, markerscale=3, n_items=n_batches,
    )
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Before/after batch correction
# ---------------------------------------------------------------------------

def plot_batch_correction_before_after(
    pre_features: np.ndarray,
    post_features: np.ndarray,
    metadata: list[WindowMetadata],
    outfile: Path,
) -> None:
    """Side-by-side UMAP embeddings before and after batch correction."""
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed — skipping before/after plot")
        return

    if pre_features.shape[0] == 0:
        return

    # Clean inputs
    pre_clean = np.nan_to_num(pre_features, nan=0.0, posinf=0.0, neginf=0.0)
    post_clean = np.nan_to_num(post_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalize pre-correction for fair comparison
    mu = pre_clean.mean(axis=0)
    sigma = pre_clean.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    pre_clean = (pre_clean - mu) / sigma

    logger.info("Computing UMAP for before/after comparison...")
    reducer_pre = umap.UMAP(n_components=2, n_neighbors=30, random_state=42)
    emb_pre = reducer_pre.fit_transform(pre_clean)

    reducer_post = umap.UMAP(n_components=2, n_neighbors=30, random_state=42)
    emb_post = reducer_post.fit_transform(post_clean)

    batches = sorted(set(m.batch_id for m in metadata))
    n_batches = len(batches)
    cmap = plt.cm.tab20 if n_batches <= 20 else plt.cm.turbo
    batch_colors = {b: cmap(i / max(n_batches - 1, 1)) for i, b in enumerate(batches)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for batch in batches:
        mask = np.array([m.batch_id == batch for m in metadata])
        color = [batch_colors[batch]]
        ax1.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=color, s=2, alpha=0.3, rasterized=True)
        ax2.scatter(emb_post[mask, 0], emb_post[mask, 1], c=color, s=2, alpha=0.3,
                    label=batch, rasterized=True)

    ax1.set_title("Before Batch Correction")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    ax2.set_title("After Batch Correction")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    ncol = max(1, n_batches // 10)
    handles, labels = ax2.get_legend_handles_labels()
    leg = ax2.get_legend()
    if leg is not None:
        leg.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            fontsize=5,
            markerscale=3,
            ncol=ncol,
            borderaxespad=0,
        )

    fig.suptitle("Batch Effect Correction (color = batch)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.82, 0.96])
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)
