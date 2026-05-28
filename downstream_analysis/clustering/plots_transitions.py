"""State-transition network diagrams for ethogram-derived cluster matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402

from .clustering_pipeline import ClusteringResult
from .plots import _cluster_cmap
from .transition_analysis import normalize_transition_matrix

logger = logging.getLogger(__name__)

_DPI = 140


def _matrix_for_display(matrix: np.ndarray, *, include_self_loops: bool) -> np.ndarray:
    """Return transition matrix for plotting; optionally drop and renormalize diagonal."""
    m = np.asarray(matrix, dtype=np.float64)
    if include_self_loops:
        return m
    m = m.copy()
    np.fill_diagonal(m, 0.0)
    return normalize_transition_matrix(m)


def _node_positions(n: int, radius: float = 1.0) -> np.ndarray:
    """Place *n* nodes evenly on a circle (first node at top)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _edge_rad(i: int, j: int, n: int) -> float:
    """Signed curvature so arrows between nearby nodes bend outward."""
    if i == j:
        return 0.0
    delta = (j - i) % n
    if delta <= n // 2:
        return 0.25
    return -0.25


def _draw_self_loop(
    ax,
    x: float,
    y: float,
    prob: float,
    *,
    node_radius: float,
    cmap,
    prob_max: float,
) -> None:
    """Draw a small self-loop above a node."""
    if prob <= 0:
        return

    norm = min(prob / max(prob_max, 1e-9), 1.0)
    color = cmap(norm)
    lw = 0.6 + 5.5 * norm
    loop_r = node_radius * 1.15
    arc = mpatches.Arc(
        (x, y + loop_r * 0.55),
        loop_r * 1.6,
        loop_r * 1.6,
        angle=0,
        theta1=200,
        theta2=-20,
        color=color,
        linewidth=lw,
        alpha=0.35 + 0.65 * norm,
        zorder=2,
    )
    ax.add_patch(arc)
    # Arrowhead at end of arc (approximate)
    tip_x = x + loop_r * 0.75 * np.cos(np.deg2rad(-20))
    tip_y = y + loop_r * 0.55 + loop_r * 0.8 * np.sin(np.deg2rad(-20))
    ax.annotate(
        "",
        xy=(tip_x, tip_y),
        xytext=(tip_x - 0.04, tip_y - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=0.35 + 0.65 * norm),
        zorder=3,
    )


def plot_transition_network(
    matrix: np.ndarray,
    n_clusters: int,
    title: str,
    outfile: Path,
    *,
    cluster_colors: list[Any] | None = None,
    prob_max: float | None = None,
    include_self_loops: bool = True,
) -> None:
    """Circular network diagram: nodes = clusters, edges = P(B | A).

    Edge linewidth and color encode transition probability (row-normalized).
    When ``include_self_loops`` is False, diagonal transitions are omitted and
    rows are renormalized over off-diagonal targets only.
    """
    if n_clusters <= 0 or matrix.shape != (n_clusters, n_clusters):
        return

    matrix = _matrix_for_display(matrix, include_self_loops=include_self_loops)

    colors = cluster_colors or _cluster_cmap(n_clusters)
    pos = _node_positions(n_clusters, radius=1.0)
    prob_max = float(prob_max if prob_max is not None else max(matrix.max(), 1e-9))
    cmap = cm.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    node_r = 0.11
    margin = 0.45

    # Edges (draw below nodes)
    for i in range(n_clusters):
        for j in range(n_clusters):
            prob = float(matrix[i, j])
            if prob <= 0:
                continue

            norm = min(prob / prob_max, 1.0)
            edge_color = cmap(norm)
            lw = 0.5 + 6.0 * norm
            alpha = 0.25 + 0.75 * norm

            if i == j:
                _draw_self_loop(
                    ax,
                    pos[i, 0],
                    pos[i, 1],
                    prob,
                    node_radius=node_r,
                    cmap=cmap,
                    prob_max=prob_max,
                )
                continue

            rad = _edge_rad(i, j, n_clusters)
            ax.annotate(
                "",
                xy=(pos[j, 0], pos[j, 1]),
                xytext=(pos[i, 0], pos[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=edge_color,
                    lw=lw,
                    alpha=alpha,
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=14,
                    shrinkB=14,
                ),
                zorder=1,
            )

    # Nodes
    for c in range(n_clusters):
        circle = plt.Circle(
            (pos[c, 0], pos[c, 1]),
            node_r,
            facecolor=colors[c],
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            pos[c, 0],
            pos[c, 1],
            f"C{c}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    ax.set_xlim(-1.0 - margin, 1.0 + margin)
    ax.set_ylim(-1.0 - margin, 1.0 + margin)
    ax.set_title(title, fontsize=11, pad=12)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=prob_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.55)
    cbar.set_label("P(next cluster | current cluster)")

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def plot_transition_heatmap(
    matrix: np.ndarray,
    n_clusters: int,
    title: str,
    outfile: Path,
    *,
    include_self_loops: bool = True,
) -> None:
    """Companion heatmap for the same transition matrix."""
    if n_clusters <= 0:
        return

    matrix = _matrix_for_display(matrix, include_self_loops=include_self_loops)

    fig, ax = plt.subplots(figsize=(max(5, n_clusters * 0.55), max(4.5, n_clusters * 0.5)))
    im = ax.imshow(matrix, vmin=0, vmax=max(matrix.max(), 1e-9), cmap="plasma", aspect="auto")
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels([f"C{c}" for c in range(n_clusters)], fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in range(n_clusters)], fontsize=8)
    ax.set_xlabel("Next cluster (B)")
    ax.set_ylabel("Current cluster (A)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(B | A)")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved %s", outfile)


def generate_transition_plots(
    result: ClusteringResult,
    comparison: dict[str, Any],
    plots_dir: Path,
    *,
    group_order: list[str] | None = None,
) -> None:
    """Generate one network diagram (and heatmap) per stratification group."""
    transitions = comparison.get("transition_matrices") or {}
    if not transitions:
        return

    out_dir = plots_dir / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_colors = _cluster_cmap(result.n_clusters)

    ordered_keys = list(group_order or [])
    for key in transitions:
        if key not in ordered_keys:
            ordered_keys.append(key)

    global_max = max(
        (max(row) for info in transitions.values() for row in info.get("matrix", [[0]])),
        default=0.0,
    )
    prob_max = max(global_max, 1e-9)

    for group_key in ordered_keys:
        info = transitions.get(group_key)
        if not info:
            continue

        matrix = np.array(info["matrix"], dtype=np.float64)
        label = info.get("label", group_key)
        n_trans = info.get("n_transitions", 0)
        n_videos = info.get("n_videos", 0)
        title = f"{label}\n(n={n_videos} videos, {n_trans} transitions)"

        safe_name = group_key.replace("/", "_").replace(" ", "_")
        for include_self_loops, suffix in (
            (True, "with_self_loops"),
            (False, "no_self_loops"),
        ):
            loop_label = "with self-loops" if include_self_loops else "no self-loops"
            plot_transition_network(
                matrix,
                result.n_clusters,
                f"{title}\n({loop_label})",
                out_dir / f"transition_network_{safe_name}_{suffix}.png",
                cluster_colors=cluster_colors,
                prob_max=prob_max,
                include_self_loops=include_self_loops,
            )
            plot_transition_heatmap(
                matrix,
                result.n_clusters,
                f"Transition matrix — {label} ({loop_label})",
                out_dir / f"transition_matrix_{safe_name}_{suffix}.png",
                include_self_loops=include_self_loops,
            )

    logger.info("Transition plots saved to %s", out_dir)
