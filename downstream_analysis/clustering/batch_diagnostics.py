"""Batch-wise diagnostic statistics and visualizations for the multi-project pipeline.

Computes per-feature batch effect sizes, variance partitioning, PERMANOVA,
and PCA before/after correction.  Results are written to a dedicated output
subfolder (``batch_diagnostics/``).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import f_oneway  # noqa: E402

from .dataset import WindowMetadata

logger = logging.getLogger(__name__)

_DPI = 140
_MAX_PCA_SAMPLES = 5000
_MAX_PERMANOVA_SAMPLES = 2000
_N_PERMUTATIONS = 499
_TOP_FEATURES_HEATMAP = 30
_TOP_FEATURES_BAR = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score with zero-variance guard."""
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=1)
    sigma[sigma < 1e-12] = 1.0
    Z = (X - mu) / sigma
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_eta_squared(
    X: np.ndarray,
    batch_labels: np.ndarray,
    fi: int,
) -> tuple[float, float]:
    """One-way ANOVA eta-squared and p-value for a single feature."""
    batches = sorted(np.unique(batch_labels), key=str)
    vals = X[:, fi]
    valid = np.isfinite(vals)
    if valid.sum() < 3:
        return 0.0, float("nan")

    y = vals[valid]
    bl = batch_labels[valid]
    grand_mean = float(y.mean())
    ss_total = float(np.sum((y - grand_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0, float("nan")

    ss_between = 0.0
    group_arrays: list[np.ndarray] = []
    for b in batches:
        g = y[bl == b]
        if len(g) < 2:
            continue
        group_arrays.append(g)
        ss_between += len(g) * (float(g.mean()) - grand_mean) ** 2

    eta_sq = ss_between / ss_total
    if len(group_arrays) < 2:
        return float(eta_sq), float("nan")

    try:
        _, p_val = f_oneway(*group_arrays)
    except ValueError:
        p_val = float("nan")

    return float(eta_sq), float(p_val)


def _per_feature_effects(
    X: np.ndarray,
    batch_labels: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Compute eta-squared and ANOVA p-value for every feature."""
    rows: list[dict[str, Any]] = []
    for fi, fname in enumerate(feature_names):
        eta_sq, p_val = _feature_eta_squared(X, batch_labels, fi)
        rows.append({
            "feature": fname,
            "eta_squared": eta_sq,
            "anova_p_value": p_val,
        })
    rows.sort(key=lambda r: r["eta_squared"], reverse=True)
    return rows


def _batch_centroids_zscored(
    X: np.ndarray,
    batch_labels: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], np.ndarray]:
    """Z-scored global matrix, then per-batch mean vectors."""
    Z = _zscore_columns(X)
    batches = sorted(np.unique(batch_labels), key=str)
    centroids = np.zeros((len(batches), Z.shape[1]), dtype=np.float64)
    for i, b in enumerate(batches):
        mask = batch_labels == b
        if mask.sum() == 0:
            continue
        centroids[i] = Z[mask].mean(axis=0)
    return [str(b) for b in batches], centroids


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """First two principal components via SVD."""
    Z = _zscore_columns(X)
    if Z.shape[0] < 3 or Z.shape[1] < 2:
        return np.zeros((Z.shape[0], 2), dtype=np.float64)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:2].T


def _subsample_stratified(
    X: np.ndarray,
    batch_labels: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample capped at *max_samples*."""
    n = X.shape[0]
    if n <= max_samples:
        return X, batch_labels

    batches = np.unique(batch_labels)
    per_batch = max(2, max_samples // len(batches))
    indices: list[int] = []
    for b in batches:
        idx = np.where(batch_labels == b)[0]
        if len(idx) <= per_batch:
            indices.extend(idx.tolist())
        else:
            indices.extend(rng.choice(idx, size=per_batch, replace=False).tolist())

    indices = indices[:max_samples]
    idx_arr = np.array(indices, dtype=int)
    return X[idx_arr], batch_labels[idx_arr]


def _permanova(
    X: np.ndarray,
    batch_labels: np.ndarray,
    *,
    n_permutations: int = _N_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Distance-based PERMANOVA (Anderson 2001) on z-scored Euclidean features."""
    if rng is None:
        rng = np.random.default_rng(42)

    batches = np.unique(batch_labels)
    if len(batches) < 2 or X.shape[0] < 3:
        return {"R2": 0.0, "F_statistic": float("nan"), "p_value": float("nan")}

    Z = _zscore_columns(X)
    n = Z.shape[0]
    grand = Z.mean(axis=0)
    ss_total = float(np.sum((Z - grand) ** 2))
    if ss_total < 1e-12:
        return {"R2": 0.0, "F_statistic": 0.0, "p_value": 1.0, "n_permutations": float(n_permutations)}

    def _ss_within(labels: np.ndarray) -> float:
        ss = 0.0
        for b in batches:
            mask = labels == b
            if not mask.any():
                continue
            diff = Z[mask] - Z[mask].mean(axis=0)
            ss += float(np.sum(diff ** 2))
        return ss

    ss_within = _ss_within(batch_labels)
    ss_between = ss_total - ss_within
    r2 = ss_between / ss_total

    df_between = len(batches) - 1
    df_within = n - len(batches)
    if df_between < 1 or df_within < 1 or ss_within < 1e-12:
        f_obs = 0.0
    else:
        f_obs = (ss_between / df_between) / (ss_within / df_within)

    count = 0
    for _ in range(n_permutations):
        if _permanova_f_from_ss(_ss_within(rng.permutation(batch_labels)), ss_total, df_between, df_within) >= f_obs:
            count += 1
    p_val = (count + 1) / (n_permutations + 1)

    return {
        "R2": float(r2),
        "F_statistic": float(f_obs),
        "p_value": float(p_val),
        "n_permutations": float(n_permutations),
    }


def _permanova_f_from_ss(
    ss_within: float,
    ss_total: float,
    df_between: int,
    df_within: int,
) -> float:
    ss_between = ss_total - ss_within
    if df_between < 1 or df_within < 1 or ss_within < 1e-12:
        return 0.0
    return (ss_between / df_between) / (ss_within / df_within)


def _batch_colors(n_batches: int) -> list[Any]:
    cmap = plt.cm.tab10 if n_batches <= 10 else (plt.cm.tab20 if n_batches <= 20 else plt.cm.turbo)
    return [cmap(i / max(n_batches - 1, 1)) for i in range(n_batches)]


def _scatter_by_batch(
    ax: plt.Axes,
    coords: np.ndarray,
    batch_labels: np.ndarray,
    batches: list[str],
    *,
    title: str,
) -> None:
    colors = _batch_colors(len(batches))
    for i, batch in enumerate(batches):
        mask = batch_labels == batch
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], s=3, alpha=0.35,
            label=f"{batch} ({int(mask.sum())})",
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_pca_before_after(
    pre_features: np.ndarray,
    post_features: np.ndarray,
    batch_labels: np.ndarray,
    outfile: Path,
    rng: np.random.Generator,
) -> None:
    pre_sub, bl_pre = _subsample_stratified(pre_features, batch_labels, _MAX_PCA_SAMPLES, rng)
    post_sub, bl_post = _subsample_stratified(post_features, batch_labels, _MAX_PCA_SAMPLES, rng)

    emb_pre = _pca_2d(pre_sub)
    emb_post = _pca_2d(post_sub)
    batches = sorted(np.unique(batch_labels), key=str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _scatter_by_batch(ax1, emb_pre, bl_pre, batches, title="Before Correction (PCA)")
    _scatter_by_batch(ax2, emb_post, bl_post, batches, title="After Correction (PCA)")
    ax2.legend(fontsize=5, markerscale=3, ncol=max(1, len(batches) // 8),
               bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Batch Structure: PCA Before vs After Correction", fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_batch_mean_heatmap(
    pre_features: np.ndarray,
    batch_labels: np.ndarray,
    feature_names: list[str],
    top_features: list[str],
    outfile: Path,
) -> None:
    if not top_features:
        return

    batches, centroids = _batch_centroids_zscored(pre_features, batch_labels, feature_names)
    fi_map = {f: i for i, f in enumerate(feature_names)}
    idx = [fi_map[f] for f in top_features if f in fi_map]
    if not idx:
        return

    data = centroids[:, idx].T  # (n_features, n_batches)

    fig_h = max(6, 0.25 * len(idx))
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(batches)), fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=7)
    ax.set_title("Batch Centroids (z-scored feature means, before correction)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-score")
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_batch_effect_bars(
    pre_effects: list[dict[str, Any]],
    post_effects: list[dict[str, Any]],
    outfile: Path,
) -> None:
    pre_map = {r["feature"]: r["eta_squared"] for r in pre_effects}
    top = pre_effects[:_TOP_FEATURES_BAR]
    features = [r["feature"] for r in top]
    pre_vals = [pre_map[f] for f in features]
    post_vals = [next((r["eta_squared"] for r in post_effects if r["feature"] == f), 0.0) for f in features]

    y = np.arange(len(features))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(features))))
    ax.barh(y - height / 2, pre_vals, height, label="Before", color="#E05555", alpha=0.8)
    ax.barh(y + height / 2, post_vals, height, label="After", color="#2196F3", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel("Eta-squared (batch variance fraction)")
    ax.set_title("Top Feature Batch Effects Before vs After Correction")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_variance_partition(
    summary: dict[str, Any],
    outfile: Path,
) -> None:
    before = summary["before_correction"]
    after = summary["after_correction"]
    labels = ["Mean η²", "Median η²", "PERMANOVA R²"]
    pre_vals = [
        before["mean_eta_squared"],
        before["median_eta_squared"],
        before["permanova"]["R2"],
    ]
    post_vals = [
        after["mean_eta_squared"],
        after["median_eta_squared"],
        after["permanova"]["R2"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pre_vals, width, label="Before", color="#E05555", alpha=0.85)
    ax.bar(x + width / 2, post_vals, width, label="After", color="#2196F3", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Variance explained by batch")
    ax.set_title("Global Batch Variance Partition")
    ax.legend()
    ax.set_ylim(0, max(max(pre_vals), max(post_vals), 0.01) * 1.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_batch_sample_counts(
    batch_sizes: dict[str, int],
    outfile: Path,
) -> None:
    batches = sorted(batch_sizes.keys())
    counts = [batch_sizes[b] for b in batches]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(batches)), 5))
    ax.bar(range(len(batches)), counts, color="#607D8B", alpha=0.85)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Window count")
    ax.set_title("Samples per Batch")
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _summarize_effects(effects: list[dict[str, Any]]) -> dict[str, float]:
    eta = np.array([r["eta_squared"] for r in effects], dtype=np.float64)
    return {
        "mean_eta_squared": float(np.mean(eta)),
        "median_eta_squared": float(np.median(eta)),
        "max_eta_squared": float(np.max(eta)) if len(eta) else 0.0,
        "n_features_eta_gt_0.1": int(np.sum(eta > 0.1)),
        "n_features_eta_gt_0.25": int(np.sum(eta > 0.25)),
    }


def run_batch_diagnostics(
    pre_features: np.ndarray,
    post_features: np.ndarray,
    batch_labels: np.ndarray,
    feature_names: list[str],
    metadata: list[WindowMetadata],
    output_dir: Path,
    batch_info: dict[str, Any],
) -> dict[str, Any]:
    """Compute batch diagnostics and write stats + figures to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    batches_sorted = sorted(np.unique(batch_labels), key=str)
    batch_sizes = {str(b): int((batch_labels == b).sum()) for b in batches_sorted}

    # Cross-tab: which experiments contribute to each batch
    from collections import Counter
    batch_experiments: dict[str, dict[str, int]] = {}
    for m in metadata:
        batch_experiments.setdefault(m.batch_id, Counter())
        if m.experiment_name:
            batch_experiments[m.batch_id][m.experiment_name] += 1
    batch_experiment_map = {
        b: dict(sorted(c.items())) for b, c in batch_experiments.items()
    }

    logger.info("Batch diagnostics: %d samples, %d batches", pre_features.shape[0], len(batches_sorted))

    pre_effects = _per_feature_effects(pre_features, batch_labels, feature_names)
    post_effects = _per_feature_effects(post_features, batch_labels, feature_names)

    pre_perm_X, pre_perm_bl = _subsample_stratified(
        pre_features, batch_labels, _MAX_PERMANOVA_SAMPLES, rng,
    )
    post_perm_X, post_perm_bl = _subsample_stratified(
        post_features, batch_labels, _MAX_PERMANOVA_SAMPLES, rng,
    )

    pre_perm = _permanova(pre_perm_X, pre_perm_bl, rng=rng)
    post_perm = _permanova(post_perm_X, post_perm_bl, rng=rng)

    pre_summary = _summarize_effects(pre_effects)
    post_summary = _summarize_effects(post_effects)
    pre_summary["permanova"] = pre_perm
    post_summary["permanova"] = post_perm

    def _pct_reduction(before: float, after: float) -> float:
        if before < 1e-12:
            return 0.0
        return float(100.0 * (before - after) / before)

    summary: dict[str, Any] = {
        "n_samples": int(pre_features.shape[0]),
        "n_features": int(pre_features.shape[1]),
        "n_batches": len(batches_sorted),
        "batch_sizes": batch_sizes,
        "batch_experiments": batch_experiment_map,
        "correction_method": batch_info.get("method", "unknown"),
        "batch_correction_info": batch_info,
        "before_correction": pre_summary,
        "after_correction": post_summary,
        "reduction": {
            "mean_eta_squared_pct": _pct_reduction(
                pre_summary["mean_eta_squared"], post_summary["mean_eta_squared"],
            ),
            "median_eta_squared_pct": _pct_reduction(
                pre_summary["median_eta_squared"], post_summary["median_eta_squared"],
            ),
            "permanova_R2_pct": _pct_reduction(pre_perm["R2"], post_perm["R2"]),
            "permanova_p_before": pre_perm["p_value"],
            "permanova_p_after": post_perm["p_value"],
        },
    }

    # --- CSV outputs ---
    effects_csv = output_dir / "per_feature_batch_effects.csv"
    with open(effects_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature", "eta_squared_before", "anova_p_before",
                "eta_squared_after", "anova_p_after", "eta_squared_reduction_pct",
            ],
        )
        writer.writeheader()
        post_map = {r["feature"]: r for r in post_effects}
        for row in pre_effects:
            post_row = post_map.get(row["feature"], {})
            eta_pre = row["eta_squared"]
            eta_post = post_row.get("eta_squared", 0.0)
            writer.writerow({
                "feature": row["feature"],
                "eta_squared_before": eta_pre,
                "anova_p_before": row["anova_p_value"],
                "eta_squared_after": eta_post,
                "anova_p_after": post_row.get("anova_p_value", float("nan")),
                "eta_squared_reduction_pct": _pct_reduction(eta_pre, eta_post),
            })

    batch_csv = output_dir / "batch_summary.csv"
    with open(batch_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["batch_id", "n_windows", "fraction", "experiments"],
        )
        writer.writeheader()
        n_total = pre_features.shape[0]
        for b, cnt in batch_sizes.items():
            exps = batch_experiment_map.get(b, {})
            writer.writerow({
                "batch_id": b,
                "n_windows": cnt,
                "fraction": cnt / max(n_total, 1),
                "experiments": ";".join(f"{k}:{v}" for k, v in exps.items()),
            })

    json_path = output_dir / "batch_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote %s", json_path.name)

    # --- Plots ---
    if len(batches_sorted) >= 2:
        _plot_pca_before_after(
            pre_features, post_features, batch_labels,
            output_dir / "pca_before_after_by_batch.png", rng,
        )
        top_feat_names = [r["feature"] for r in pre_effects[:_TOP_FEATURES_HEATMAP]]
        _plot_batch_mean_heatmap(
            pre_features, batch_labels, feature_names, top_feat_names,
            output_dir / "batch_mean_heatmap.png",
        )
        _plot_batch_effect_bars(pre_effects, post_effects, output_dir / "batch_effect_top_features.png")
        _plot_variance_partition(summary, output_dir / "variance_partition_before_after.png")

    _plot_batch_sample_counts(batch_sizes, output_dir / "batch_sample_counts.png")

    logger.info("Batch diagnostics saved to %s", output_dir)
    return summary
