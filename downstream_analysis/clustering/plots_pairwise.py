"""Per-pairwise-comparison plots for multi-project analysis."""

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

from .clustering_pipeline import ClusteringResult

logger = logging.getLogger(__name__)

_DPI = 140


def plot_pairwise_cluster_enrichment(
    pair_result: dict[str, Any],
    outfile: Path,
) -> None:
    """Bar chart of group-A enrichment odds ratio per cluster."""
    per_cluster = pair_result.get("per_cluster") or pair_result.get("cluster_stats", {})
    if not per_cluster:
        return

    disp_a = pair_result.get("display_a", pair_result.get("label_a", "A"))
    disp_b = pair_result.get("display_b", pair_result.get("label_b", "B"))

    clusters = sorted(per_cluster.keys(), key=int)
    odds = [per_cluster[c]["odds_ratio"] for c in clusters]
    pvals = [per_cluster[c]["p_value"] for c in clusters]

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 0.6), 4))
    x = np.arange(len(clusters))
    ax.bar(x, odds, color="#E05555", alpha=0.7, width=0.6)

    for i, (o, p) in enumerate(zip(odds, pvals)):
        if np.isfinite(p) and p < 0.05:
            ax.text(i, o + 0.05 if o >= 0 else o - 0.1, "*", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
    ax.set_ylabel(f"Odds ratio ({disp_a} enrichment)")
    ax.set_title(f"Cluster enrichment: {disp_a} vs {disp_b} (* = p < 0.05)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def plot_pairwise_feature_enrichment(
    pair_result: dict[str, Any],
    outfile: Path,
) -> None:
    """Horizontal bar chart of Cohen's d per feature for one pairwise comparison."""
    enrichment = pair_result.get("feature_enrichment", [])
    if not enrichment:
        return

    disp_a = pair_result.get("display_a", pair_result.get("label_a", "A"))
    disp_b = pair_result.get("display_b", pair_result.get("label_b", "B"))

    enrichment_sorted = sorted(enrichment, key=lambda x: abs(x["cohens_d"]))
    names = [e["feature"] for e in enrichment_sorted]
    cohens = [e["cohens_d"] for e in enrichment_sorted]
    pvals = [e["mannwhitney_p_value"] for e in enrichment_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y = np.arange(len(names))
    colors = ["#E05555" if d > 0 else "#5588DD" for d in cohens]
    ax.barh(y, cohens, color=colors, alpha=0.75, height=0.7)

    for i, (d, p) in enumerate(zip(cohens, pvals)):
        if np.isfinite(p) and p < 0.05:
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            offset = 0.02 if d >= 0 else -0.02
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, i, marker, ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(f"Cohen's d (positive = higher in {disp_a})")
    ax.set_title(f"Feature enrichment: {disp_a} vs {disp_b}")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#E05555", alpha=0.75, label=f"Higher in {disp_a}"),
            Patch(facecolor="#5588DD", alpha=0.75, label=f"Higher in {disp_b}"),
        ],
        fontsize=7,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def generate_pairwise_comparison_plots(
    comparison: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate cluster + feature enrichment plots for every pairwise comparison."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        ("condition_groups", comparison.get("pairwise_condition_groups", {})),
        ("role_conditions", comparison.get("pairwise_role_conditions", {})),
    ]

    for section_name, pairs in sections:
        if not pairs:
            continue
        section_dir = plots_dir / "pairwise" / section_name
        section_dir.mkdir(parents=True, exist_ok=True)

        for pair_key, pair_result in pairs.items():
            if pair_result.get("note") and "per_cluster" not in pair_result:
                continue
            plot_pairwise_cluster_enrichment(
                pair_result,
                section_dir / f"{pair_key}_cluster_enrichment.png",
            )
            plot_pairwise_feature_enrichment(
                pair_result,
                section_dir / f"{pair_key}_feature_enrichment.png",
            )

    logger.info("Pairwise comparison plots saved under %s/pairwise/", plots_dir)


def save_pairwise_comparison_results(
    comparison: dict[str, Any],
    results_dir: Path,
) -> None:
    """Write per-pair JSON and CSV tables under results/pairwise_comparisons/."""
    base = results_dir / "pairwise_comparisons"
    base.mkdir(parents=True, exist_ok=True)

    sections = [
        ("condition_groups", comparison.get("pairwise_condition_groups", {})),
        ("role_conditions", comparison.get("pairwise_role_conditions", {})),
    ]

    for section_name, pairs in sections.items():
        section_dir = base / section_name
        section_dir.mkdir(parents=True, exist_ok=True)

        for pair_key, pair_result in pairs.items():
            pair_dir = section_dir / pair_key
            pair_dir.mkdir(parents=True, exist_ok=True)

            (pair_dir / "comparison.json").write_text(
                json.dumps(pair_result, indent=2, default=str)
            )

            per_cluster = pair_result.get("per_cluster", {})
            if per_cluster:
                csv_path = pair_dir / "cluster_stats.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "cluster", "a_in_cluster", "b_in_cluster",
                        "a_frac", "b_frac", "odds_ratio", "p_value",
                    ])
                    for c_str in sorted(per_cluster, key=int):
                        pc = per_cluster[c_str]
                        writer.writerow([
                            c_str, pc["a_in_cluster"], pc["b_in_cluster"],
                            pc["a_frac"], pc["b_frac"],
                            pc["odds_ratio"], pc["p_value"],
                        ])

            feat_enr = pair_result.get("feature_enrichment", [])
            if feat_enr:
                csv_path = pair_dir / "feature_enrichment.csv"
                cols = [
                    "feature", "mean_a", "mean_b", "cohens_d", "log2_fold_change",
                    "odds_ratio", "fisher_p_value", "mannwhitney_U",
                    "mannwhitney_p_value", "n_a", "n_b", "direction",
                ]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    writer.writeheader()
                    for row in feat_enr:
                        writer.writerow({k: row.get(k, "") for k in cols})

            per_feat = pair_result.get("per_feature_tests", {})
            if per_feat:
                csv_path = pair_dir / "feature_stats_tests.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "feature", "comparison", "U_statistic", "p_value",
                        "mean_a", "mean_b", "cohens_d", "effect_direction",
                    ])
                    for fname, ft in sorted(
                        per_feat.items(),
                        key=lambda x: x[1].get("p_value", 999) if np.isfinite(x[1].get("p_value", 999)) else 999,
                    ):
                        writer.writerow([
                            fname, ft.get("comparison", ""),
                            ft.get("U_statistic", ""), ft.get("p_value", ""),
                            ft.get("mean_a", ""), ft.get("mean_b", ""),
                            ft.get("cohens_d", ""), ft.get("effect_direction", ""),
                        ])

    logger.info("Pairwise comparison results saved to %s", base)
