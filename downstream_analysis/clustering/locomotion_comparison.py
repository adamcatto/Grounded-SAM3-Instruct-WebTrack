"""Compare resident locomotion (alone) vs resident behavior when paired.

Extracts 12-dim per-object sequence features from:
  - Locomotion videos (single mouse, Exp 4)
  - Interaction videos (resident mouse only, Exp 2/3)

Pairs them by (mouse_id, camera_view) for per-camera paired statistical tests.
Also loads existing locomotion pipeline metrics (path length).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from mask_store import VideoMaskStorage

from ..tracking_io import (
    VideoTrackContext,
    load_project_config,
    object_display_name,
    video_storage_dir,
)
from .config import ClusteringConfig
from .experiment_registry import (
    MouseIdentity,
    box_from_camera_view,
    experiment_name_from_project_dir,
    normalize_shave_name,
    parse_video_name,
    resolve_identities,
    resolve_single_identity,
)
from .frame_features import N_SINGLE_FEATURES, compute_single_mask_features
from .sequence_features import (
    N_PER_OBJECT_SEQ,
    _PER_OBJECT_NAMES,
    _object_sequence_features,
)
from ..locomotion import window_starts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature names for single-mouse comparison
# ---------------------------------------------------------------------------

SINGLE_MOUSE_FEATURE_NAMES = list(_PER_OBJECT_NAMES)  # 12 features


# ---------------------------------------------------------------------------
# Extract per-object frame features for a single mouse
# ---------------------------------------------------------------------------

def _extract_single_mouse_frame_features(
    ctx: VideoTrackContext,
    obj_key: str,
    cfg: ClusteringConfig,
) -> np.ndarray | None:
    """Extract (T, 7) frame features for a single object in a video.

    Returns array of shape (n_frames, 7) or None on failure.
    """
    start = int(ctx.config.get("start_frame") or 0)
    n = int(ctx.config["num_frames"])
    total_fr = max(0, n - start)

    ms = VideoMaskStorage(ctx.video_dir)
    frame_features = np.full((total_fr, N_SINGLE_FEATURES), np.nan, dtype=np.float64)

    for fi in range(start, n):
        masks = ms.load_masks_dense(fi)
        row = fi - start
        mask = masks.get(obj_key)
        if mask is not None:
            feats = compute_single_mask_features(mask)
            if feats is not None:
                frame_features[row] = feats

    return frame_features


def _frame_to_sequence_features(
    frame_features: np.ndarray,
    cfg: ClusteringConfig,
    video_diagonal: float,
    video_area: float,
) -> tuple[np.ndarray, list[int]]:
    """Convert single-mouse (T, 7) frame features into (n_windows, 12) sequence features.

    Uses the same sliding-window approach as the pair pipeline.
    """
    T = frame_features.shape[0]
    starts = window_starts(T, cfg.window_size, cfg.stride)
    if len(starts) == 0:
        return np.empty((0, N_PER_OBJECT_SEQ), dtype=np.float64), []

    # Normalize frame features to fraction of diagonal/area
    ff = frame_features.copy()
    if video_diagonal > 0:
        ff[:, 0] /= video_diagonal  # centroid_x
        ff[:, 1] /= video_diagonal  # centroid_y
    if video_area > 0:
        ff[:, 2] /= video_area  # area
        ff[:, 3] /= video_diagonal  # major_axis
        ff[:, 4] /= video_diagonal  # minor_axis

    # Build windowed features — pad to 21 columns (as _object_sequence_features expects)
    padded = np.full((T, 21), np.nan, dtype=np.float64)
    padded[:, :N_SINGLE_FEATURES] = ff  # columns 0-6 for object A

    seq_feats = np.full((len(starts), N_PER_OBJECT_SEQ), np.nan, dtype=np.float64)
    stat_thresh = cfg.stationary_speed_threshold

    for i, s in enumerate(starts):
        window = padded[s:s + cfg.window_size]
        seq_feats[i] = _object_sequence_features(
            window, 0, 1, 2, 5, 6, stat_thresh,
        )

    return seq_feats, starts


# ---------------------------------------------------------------------------
# Extract features for all videos in a project
# ---------------------------------------------------------------------------

def extract_locomotion_features(
    project_dir: Path,
    cfg: ClusteringConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract 12-dim per-object features for each single-mouse locomotion video.

    Returns dict keyed by mouse_id:
        {mouse_id: {
            'features': (n_windows, 12) array,
            'camera_view': str,
            'session': str,
            'video_name': str,
        }}
    """
    if cfg is None:
        cfg = ClusteringConfig()

    project_dir = Path(project_dir).resolve()
    config = load_project_config(project_dir)
    exp_name = experiment_name_from_project_dir(project_dir)
    results: dict[str, dict[str, Any]] = {}

    videos = config.get("videos", {})
    logger.info("Extracting locomotion features from %d videos in %s", len(videos), project_dir.name)

    for vid, v in videos.items():
        if not v.get("propagation_complete"):
            continue
        vdir = video_storage_dir(project_dir, vid)
        if vdir is None or not (vdir / "masks.sqlite").is_file():
            continue

        vname = v.get("name", "")
        parsed = parse_video_name(vname)
        if parsed is None:
            continue
        camera_view, session = parsed
        box = box_from_camera_view(camera_view)

        objects = v.get("objects", {})
        if len(objects) < 1:
            continue

        # Get the single tracked object
        obj_key = list(objects.keys())[0]
        obj_name = objects[obj_key].get("name", "")

        identity = resolve_single_identity(exp_name, session, box, obj_name)
        if identity is None:
            continue

        w = int(v.get("width") or 1)
        h = int(v.get("height") or 1)
        diag = float(np.hypot(w, h))
        area = float(w * h)

        ctx = VideoTrackContext(
            video_id=str(vid), video_name=vname,
            config=dict(v), video_dir=vdir,
        )

        ff = _extract_single_mouse_frame_features(ctx, obj_key, cfg)
        if ff is None:
            continue

        seq, starts = _frame_to_sequence_features(ff, cfg, diag, area)
        if seq.shape[0] == 0:
            continue

        key = f"{identity.mouse_id}_{camera_view}"
        results[key] = {
            "features": seq,
            "mouse_id": identity.mouse_id,
            "housing": identity.housing,
            "camera_view": camera_view,
            "session": session,
            "video_name": vname,
            "n_windows": seq.shape[0],
        }
        logger.info("  %s (%s, %s): %d windows", vname, identity.mouse_id, camera_view, seq.shape[0])

    logger.info("Locomotion features: %d mouse-camera pairs extracted", len(results))
    return results


def extract_resident_features(
    project_dir: Path,
    cfg: ClusteringConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract 12-dim per-object features for the RESIDENT mouse only in interaction videos.

    Returns dict keyed by '{mouse_id}_{camera_view}':
        {key: {'features': (n_windows, 12), 'mouse_id': str, ...}}
    """
    if cfg is None:
        cfg = ClusteringConfig()

    project_dir = Path(project_dir).resolve()
    config = load_project_config(project_dir)
    exp_name = experiment_name_from_project_dir(project_dir)
    results: dict[str, dict[str, Any]] = {}

    videos = config.get("videos", {})
    logger.info("Extracting resident features from %d videos in %s", len(videos), project_dir.name)

    for vid, v in videos.items():
        if not v.get("propagation_complete"):
            continue
        vdir = video_storage_dir(project_dir, vid)
        if vdir is None or not (vdir / "masks.sqlite").is_file():
            continue

        vname = v.get("name", "")
        parsed = parse_video_name(vname)
        if parsed is None:
            continue
        camera_view, session = parsed
        box = box_from_camera_view(camera_view)

        objects = v.get("objects", {})
        if len(objects) < 2:
            continue

        obj_keys = sorted(objects.keys())
        obj_names = [objects[k].get("name", "") for k in obj_keys]
        identities = resolve_identities(exp_name, session, box, obj_names[0], obj_names[1])
        if identities is None:
            continue

        id_a, id_b = identities

        # Find the resident
        if id_a.role == "resident":
            resident_key = obj_keys[0]
            resident_id = id_a
        elif id_b.role == "resident":
            resident_key = obj_keys[1]
            resident_id = id_b
        else:
            continue  # no resident (e.g. both littermates)

        w = int(v.get("width") or 1)
        h = int(v.get("height") or 1)
        diag = float(np.hypot(w, h))
        area = float(w * h)

        ctx = VideoTrackContext(
            video_id=str(vid), video_name=vname,
            config=dict(v), video_dir=vdir,
        )

        # Determine column offset based on which object is the resident
        # Object A is obj_keys[0], Object B is obj_keys[1]
        # In the mask storage, keys are "1", "2" etc.
        ff = _extract_single_mouse_frame_features(ctx, resident_key, cfg)
        if ff is None:
            continue

        seq, starts = _frame_to_sequence_features(ff, cfg, diag, area)
        if seq.shape[0] == 0:
            continue

        key = f"{resident_id.mouse_id}_{camera_view}"
        results[key] = {
            "features": seq,
            "mouse_id": resident_id.mouse_id,
            "housing": resident_id.housing,
            "camera_view": camera_view,
            "session": session,
            "video_name": vname,
            "n_windows": seq.shape[0],
        }
        logger.info("  %s (%s resident, %s): %d windows",
                     vname, resident_id.mouse_id, camera_view, seq.shape[0])

    logger.info("Resident features: %d mouse-camera pairs extracted", len(results))
    return results


# ---------------------------------------------------------------------------
# Locomotion pipeline metrics loader
# ---------------------------------------------------------------------------

def load_locomotion_metrics(project_dir: Path) -> dict[str, list[float]]:
    """Load existing locomotion pipeline CSV outputs (path length).

    Returns dict: {csv_stem: [normalized_chunk_path_length values]}
    """
    samples_dir = project_dir / "analysis_of_tracking_data" / "locomotion" / "samples"
    if not samples_dir.is_dir():
        logger.warning("No locomotion samples directory: %s", samples_dir)
        return {}

    results: dict[str, list[float]] = {}
    for csv_path in sorted(samples_dir.glob("*.csv")):
        values: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, val in row.items():
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
        if values:
            results[csv_path.stem] = values

    logger.info("Loaded locomotion metrics: %d files from %s", len(results), samples_dir)
    return results


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compare_locomotion_vs_paired(
    loc_features: dict[str, dict[str, Any]],
    paired_features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare locomotion (alone) vs paired (with intruder) for matched mice.

    Matches by key = '{mouse_id}_{camera_view}'.
    Returns per-feature paired tests (Wilcoxon signed-rank) and unpaired (Mann-Whitney).
    """
    # Find matching keys
    common_keys = sorted(set(loc_features) & set(paired_features))
    logger.info("Locomotion vs paired: %d matched mouse-camera pairs", len(common_keys))

    if len(common_keys) == 0:
        return {"note": "no matched pairs", "n_matched": 0}

    # Aggregate per-mouse mean features
    loc_means: list[np.ndarray] = []
    paired_means: list[np.ndarray] = []
    mouse_ids: list[str] = []
    housing_labels: list[str] = []

    for key in common_keys:
        loc_feat = loc_features[key]["features"]
        paired_feat = paired_features[key]["features"]
        loc_means.append(np.nanmean(loc_feat, axis=0))
        paired_means.append(np.nanmean(paired_feat, axis=0))
        mouse_ids.append(loc_features[key]["mouse_id"])
        housing_labels.append(loc_features[key]["housing"])

    loc_arr = np.array(loc_means)      # (n_pairs, 12)
    paired_arr = np.array(paired_means)  # (n_pairs, 12)
    n_pairs = loc_arr.shape[0]

    # Split by housing for subgroup analysis
    sh_mask = np.array([h == "SH" for h in housing_labels])
    gh_mask = np.array([h == "GH" for h in housing_labels])

    results: dict[str, Any] = {
        "n_matched": n_pairs,
        "n_SH": int(sh_mask.sum()),
        "n_GH": int(gh_mask.sum()),
        "mouse_ids": mouse_ids,
        "housing": housing_labels,
    }

    # Per-feature paired tests
    def _run_tests(
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        label: str,
    ) -> list[dict[str, Any]]:
        tests: list[dict[str, Any]] = []
        n = arr_a.shape[0]
        for fi, fname in enumerate(SINGLE_MOUSE_FEATURE_NAMES):
            va = arr_a[:, fi]
            vb = arr_b[:, fi]
            valid = np.isfinite(va) & np.isfinite(vb)
            va_c, vb_c = va[valid], vb[valid]
            if len(va_c) < 3:
                continue

            mean_loc = float(np.mean(va_c))
            mean_paired = float(np.mean(vb_c))

            # Wilcoxon signed-rank (paired)
            try:
                w_stat, w_p = wilcoxon(va_c, vb_c)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")

            # Mann-Whitney (unpaired, for reference)
            try:
                u_stat, u_p = mannwhitneyu(va_c, vb_c, alternative="two-sided")
            except ValueError:
                u_stat, u_p = float("nan"), float("nan")

            tests.append({
                "feature": fname,
                "mean_locomotion": mean_loc,
                "mean_paired": mean_paired,
                "diff": mean_paired - mean_loc,
                "wilcoxon_statistic": float(w_stat),
                "wilcoxon_p_value": float(w_p),
                "mannwhitney_U": float(u_stat),
                "mannwhitney_p_value": float(u_p),
                "n_valid": int(valid.sum()),
            })
        tests.sort(key=lambda x: x["wilcoxon_p_value"] if np.isfinite(x["wilcoxon_p_value"]) else 999)
        return tests

    results["all_mice"] = _run_tests(loc_arr, paired_arr, "all")

    if sh_mask.sum() >= 3:
        results["SH_mice"] = _run_tests(loc_arr[sh_mask], paired_arr[sh_mask], "SH")

    if gh_mask.sum() >= 3:
        results["GH_mice"] = _run_tests(loc_arr[gh_mask], paired_arr[gh_mask], "GH")

    # SH alone vs GH alone (locomotion only, unpaired)
    if sh_mask.sum() >= 3 and gh_mask.sum() >= 3:
        sh_vs_gh: list[dict[str, Any]] = []
        for fi, fname in enumerate(SINGLE_MOUSE_FEATURE_NAMES):
            va = loc_arr[sh_mask, fi]
            vb = loc_arr[gh_mask, fi]
            va_c = va[np.isfinite(va)]
            vb_c = vb[np.isfinite(vb)]
            if len(va_c) < 2 or len(vb_c) < 2:
                continue
            try:
                u_stat, u_p = mannwhitneyu(va_c, vb_c, alternative="two-sided")
            except ValueError:
                u_stat, u_p = float("nan"), float("nan")
            sh_vs_gh.append({
                "feature": fname,
                "mean_SH": float(np.mean(va_c)),
                "mean_GH": float(np.mean(vb_c)),
                "mannwhitney_U": float(u_stat),
                "mannwhitney_p_value": float(u_p),
            })
        sh_vs_gh.sort(key=lambda x: x["mannwhitney_p_value"] if np.isfinite(x["mannwhitney_p_value"]) else 999)
        results["SH_vs_GH_locomotion"] = sh_vs_gh

    # Raw data for boxplot generation
    results["loc_features"] = loc_arr.tolist()
    results["paired_features"] = paired_arr.tolist()
    results["feature_names"] = SINGLE_MOUSE_FEATURE_NAMES

    return results
