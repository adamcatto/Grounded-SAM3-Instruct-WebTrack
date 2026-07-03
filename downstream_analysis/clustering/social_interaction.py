"""Social interaction time analysis.

Computes fraction of time each pair spends within various proximity cutoffs
(fraction of video diagonal), grouped by condition and role.  Uses frame-level
centroid distances from cached frame features.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ..figio import save_figure  # noqa: E402

from .dataset import WindowMetadata
from .multi_project_comparison import (
    _assign_condition_group,
    CONDITION_GROUP_LABELS,
    CONDITION_GROUP_ORDER,
    ROLE_CONDITION_LABELS,
    ROLE_CONDITION_ORDER,
    ROLE_MASK_BUILDERS,
)

logger = logging.getLogger(__name__)

_DPI = 140

# Centroid distance column in the (T, 21) frame feature array
_CDIST_COL = 14  # 2 * 7 (per-object) + 0 (first interaction feature)

DEFAULT_PROXIMITY_CUTOFFS = [0.02, 0.05, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Compute social interaction time
# ---------------------------------------------------------------------------

def compute_social_interaction_time(
    project_dirs: list[Path],
    metadata: list[WindowMetadata],
    *,
    cutoffs: list[float] | None = None,
) -> dict[str, Any]:
    """Compute proximity time fractions at multiple distance cutoffs.

    For each video, loads cached frame features, normalizes centroid distance
    by video diagonal, and computes the fraction of frames where the normalized
    distance is below each cutoff.

    Parameters
    ----------
    project_dirs : list of project directories (frame feature caches live here)
    metadata : per-window metadata (used to discover video IDs and their roles)
    cutoffs : proximity cutoffs as fraction of video diagonal

    Returns
    -------
    dict with keys: cutoffs, per_video, by_condition_group, by_role_condition
    """
    if cutoffs is None:
        cutoffs = list(DEFAULT_PROXIMITY_CUTOFFS)

    # Build video_id -> metadata mapping (first window's metadata per video)
    video_meta: dict[str, WindowMetadata] = {}
    for m in metadata:
        if m.video_id not in video_meta:
            video_meta[m.video_id] = m

    # Build video_id -> project_dir mapping by scanning for cached frame features
    video_to_cache: dict[str, Path] = {}
    for proj_dir in project_dirs:
        cache_dir = proj_dir / "analysis_of_tracking_data" / "clustering" / "features"
        if not cache_dir.is_dir():
            continue
        for npz_path in cache_dir.glob("frame_features_*.npz"):
            vid_id = npz_path.stem.replace("frame_features_", "")
            video_to_cache[vid_id] = npz_path

    per_video: dict[str, dict[str, Any]] = {}

    for vid_id, m in video_meta.items():
        cache_path = video_to_cache.get(vid_id)
        if cache_path is None:
            continue

        try:
            data = np.load(cache_path, allow_pickle=True)
            frame_features = data["frame_features"]
            # Get video diagonal for normalization
            if "video_diagonal" in data:
                diag = float(data["video_diagonal"])
            elif "width" in data and "height" in data:
                diag = float(np.hypot(float(data["width"]), float(data["height"])))
            else:
                logger.warning("No video dimensions in cache for %s, skipping", vid_id)
                continue
        except Exception as e:
            logger.warning("Failed to load frame features for %s: %s", vid_id, e)
            continue

        if diag < 1.0:
            continue

        # Extract centroid distance series and normalize
        cdist_raw = frame_features[:, _CDIST_COL]
        valid = np.isfinite(cdist_raw)
        if valid.sum() < 10:
            continue

        cdist_norm = cdist_raw / diag

        # Compute fraction within each cutoff
        fractions: dict[str, float] = {}
        for c in cutoffs:
            frac = float(np.nanmean(cdist_norm[valid] < c))
            fractions[str(c)] = frac

        cond_group = _assign_condition_group(m)

        per_video[vid_id] = {
            "video_name": m.video_name,
            "condition_group": cond_group,
            "experiment_name": m.experiment_name,
            "mouse_a_role": m.mouse_a_role,
            "mouse_a_housing": m.mouse_a_housing,
            "mouse_b_role": m.mouse_b_role,
            "mouse_b_housing": m.mouse_b_housing,
            "n_frames": int(valid.sum()),
            "fractions": fractions,
        }

    # Group by condition
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {str(c): [] for c in cutoffs}
    )
    for vid_data in per_video.values():
        g = vid_data["condition_group"]
        for c_str, frac in vid_data["fractions"].items():
            by_condition[g][c_str].append(frac)

    # Group by role (each video contributes to both roles present)
    by_role: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {str(c): [] for c in cutoffs}
    )
    # We use pair-level metadata, so assign the video to each role present
    for vid_id, vid_data in per_video.items():
        m = video_meta[vid_id]
        roles = _roles_for_video(m)
        for role in roles:
            for c_str, frac in vid_data["fractions"].items():
                by_role[role][c_str].append(frac)

    logger.info(
        "Social interaction time: %d videos, %d cutoffs",
        len(per_video), len(cutoffs),
    )

    return {
        "cutoffs": cutoffs,
        "per_video": per_video,
        "by_condition_group": dict(by_condition),
        "by_role_condition": dict(by_role),
    }


def _roles_for_video(m: WindowMetadata) -> list[str]:
    """Determine which role conditions apply to a video based on its metadata."""
    roles = []
    if m.experiment_name == "hab":
        roles.append("GH_littermate")
    elif m.experiment_name == "test_day":
        if m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH":
            roles.append("SH_resident")
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "intruder":
            roles.append("GH_intruder")
        elif m.mouse_b_housing == "GH" and m.mouse_b_role == "intruder":
            roles.append("GH_intruder")
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "resident":
            roles.append("GH_resident")
        elif m.mouse_b_housing == "GH" and m.mouse_b_role == "resident":
            roles.append("GH_resident")
    elif m.experiment_name == "sh_intruder":
        if m.mouse_a_housing == "SH" or m.mouse_b_housing == "SH":
            roles.append("SH_intruder")
        if m.mouse_a_housing == "GH" and m.mouse_a_role == "resident":
            roles.append("GH_resident")
        elif m.mouse_b_housing == "GH" and m.mouse_b_role == "resident":
            roles.append("GH_resident")
    return roles


# ---------------------------------------------------------------------------
# Plotting
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


def plot_social_interaction_time(
    social_data: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate social interaction time plots for each cutoff.

    Creates boxplots grouped by condition_group and by role_condition,
    one per cutoff, plus a summary line chart across cutoffs.

    All saved under ``plots_dir/social_interaction/``.
    """
    out_dir = plots_dir / "social_interaction"

    cutoffs = social_data.get("cutoffs", [])
    by_condition = social_data.get("by_condition_group", {})
    by_role = social_data.get("by_role_condition", {})

    # Per-cutoff boxplots by condition group
    for cutoff in cutoffs:
        c_str = str(cutoff)
        _plot_boxplot_for_cutoff(
            by_condition, c_str, cutoff,
            CONDITION_GROUP_ORDER, CONDITION_GROUP_LABELS, _CONDITION_COLORS,
            out_dir / f"proximity_by_condition_cutoff_{cutoff:.2f}.png",
            "Condition Group",
        )
        _plot_boxplot_for_cutoff(
            by_role, c_str, cutoff,
            ROLE_CONDITION_ORDER, ROLE_CONDITION_LABELS, _ROLE_COLORS,
            out_dir / f"proximity_by_role_cutoff_{cutoff:.2f}.png",
            "Role Condition",
        )

    # Summary line chart: mean fraction vs cutoff for each condition
    if cutoffs and by_condition:
        _plot_summary_line_chart(
            by_condition, cutoffs,
            CONDITION_GROUP_ORDER, CONDITION_GROUP_LABELS, _CONDITION_COLORS,
            out_dir / "proximity_summary_by_condition.png",
            "Condition Group",
        )
    if cutoffs and by_role:
        _plot_summary_line_chart(
            by_role, cutoffs,
            ROLE_CONDITION_ORDER, ROLE_CONDITION_LABELS, _ROLE_COLORS,
            out_dir / "proximity_summary_by_role.png",
            "Role Condition",
        )

    logger.info("Social interaction time plots saved to %s", out_dir)


def _plot_boxplot_for_cutoff(
    grouped_data: dict[str, dict[str, list[float]]],
    cutoff_str: str,
    cutoff_val: float,
    group_order: list[str],
    group_labels: dict[str, str],
    group_colors: dict[str, str],
    outfile: Path,
    group_type: str,
) -> None:
    """Boxplot of proximity fraction for a single cutoff, grouped by category."""
    present = [g for g in group_order if g in grouped_data and grouped_data[g].get(cutoff_str)]
    if len(present) < 2:
        return

    data = [grouped_data[g][cutoff_str] for g in present]
    colors = [group_colors.get(g, "#888888") for g in present]
    labels = [group_labels.get(g, g) for g in present]

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.5), 5))
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.6, showfliers=True,
        flierprops=dict(marker=".", markersize=3),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of Time Within Proximity")
    ax.set_title(
        f"Social Interaction Time (cutoff = {cutoff_val:.0%} of diagonal)\n"
        f"by {group_type}",
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)


def _plot_summary_line_chart(
    grouped_data: dict[str, dict[str, list[float]]],
    cutoffs: list[float],
    group_order: list[str],
    group_labels: dict[str, str],
    group_colors: dict[str, str],
    outfile: Path,
    group_type: str,
) -> None:
    """Line chart: mean proximity fraction vs cutoff for each group."""
    present = [g for g in group_order if g in grouped_data]
    if len(present) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in present:
        means = []
        sems = []
        for c in cutoffs:
            vals = grouped_data[g].get(str(c), [])
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
            else:
                means.append(0.0)
                sems.append(0.0)
        color = group_colors.get(g, "#888888")
        label = group_labels.get(g, g)
        ax.errorbar(
            cutoffs, means, yerr=sems,
            marker="o", color=color, label=label,
            linewidth=1.5, markersize=5, capsize=3,
        )

    ax.set_xlabel("Proximity Cutoff (fraction of video diagonal)")
    ax.set_ylabel("Mean Fraction of Time Within Proximity")
    ax.set_title(f"Social Interaction Time Summary by {group_type}")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, outfile, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outfile)
