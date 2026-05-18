"""Sliding-window aggregation of frame features into sequence-level vectors.

Takes (T, 21) frame feature arrays and produces (n_windows, 33) sequence vectors.

Per-object (12 each, 24 total):
  energy, speed_mean, speed_std, speed_max,
  angular_velocity_mean, angular_velocity_std,
  frac_time_moving, area_mean, area_std, area_range,
  area_slope, mean_eccentricity

Interaction (9):
  approach_retreat_rate, parallel_movement_score, chase_metric, chase_lag,
  frac_close_proximity, mean_relative_heading,
  min_distance_mean, min_distance_std, overlap_frac_mean
"""

from __future__ import annotations

import numpy as np

from ..locomotion import window_starts
from .frame_features import N_SINGLE_FEATURES

# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------

_PER_OBJECT_NAMES = [
    "energy",
    "speed_mean",
    "speed_std",
    "speed_max",
    "angular_velocity_mean",
    "angular_velocity_std",
    "frac_time_moving",
    "area_mean",
    "area_std",
    "area_range",
    "area_slope",
    "mean_eccentricity",
]

_INTERACTION_SEQ_NAMES = [
    "approach_retreat_rate",
    "parallel_movement_score",
    "chase_metric",
    "chase_lag",
    "frac_close_proximity",
    "mean_relative_heading",
    "min_distance_mean",
    "min_distance_std",
    "overlap_frac_mean",
]

N_PER_OBJECT_SEQ = len(_PER_OBJECT_NAMES)       # 12
N_INTERACTION_SEQ = len(_INTERACTION_SEQ_NAMES)  # 9
N_SEQUENCE_FEATURES = 2 * N_PER_OBJECT_SEQ + N_INTERACTION_SEQ  # 33

SEQUENCE_FEATURE_NAMES: list[str] = (
    [f"a_{n}" for n in _PER_OBJECT_NAMES]
    + [f"b_{n}" for n in _PER_OBJECT_NAMES]
    + _INTERACTION_SEQ_NAMES
)


# ---------------------------------------------------------------------------
# Frame-level column indices (into the 21-element frame vector)
# ---------------------------------------------------------------------------

# Object A columns 0..6, Object B columns 7..13, Interaction columns 14..20
_A_CX, _A_CY, _A_AREA, _A_MAJOR, _A_MINOR, _A_ORIENT, _A_ECC = range(7)
_B_OFF = N_SINGLE_FEATURES
_B_CX = _B_OFF + 0
_B_CY = _B_OFF + 1
_B_AREA = _B_OFF + 2
_B_ORIENT = _B_OFF + 5
_B_ECC = _B_OFF + 6

_I_OFF = 2 * N_SINGLE_FEATURES
_I_CDIST = _I_OFF + 0
_I_MIN_CONTOUR = _I_OFF + 1
_I_OVERLAP = _I_OFF + 2
_I_REL_ANGLE = _I_OFF + 3
_I_HEAD_AB = _I_OFF + 4
_I_HEAD_BA = _I_OFF + 5
_I_COMBINED_AREA = _I_OFF + 6


# ---------------------------------------------------------------------------
# Per-object sequence feature computation
# ---------------------------------------------------------------------------

def _object_sequence_features(
    window: np.ndarray,
    cx_col: int,
    cy_col: int,
    area_col: int,
    orient_col: int,
    ecc_col: int,
    stationary_threshold: float,
) -> np.ndarray:
    """Compute 12 sequence features for one object within a window.

    Parameters
    ----------
    window : shape (W, 21) frame features for the window
    cx_col, cy_col, etc. : column indices for this object's features
    stationary_threshold : speed below this is "stationary" (already normalized)
    """
    out = np.full(N_PER_OBJECT_SEQ, np.nan, dtype=np.float64)

    cx = window[:, cx_col]
    cy = window[:, cy_col]
    areas = window[:, area_col]
    orients = window[:, orient_col]
    eccs = window[:, ecc_col]

    # Check for sufficient valid data
    valid = np.isfinite(cx) & np.isfinite(cy)
    if np.sum(valid) < 3:
        return out

    # Centroid displacements
    dx = np.diff(cx)
    dy = np.diff(cy)
    step_valid = valid[:-1] & valid[1:]
    speeds = np.where(step_valid, np.hypot(dx, dy), np.nan)
    speeds_clean = speeds[np.isfinite(speeds)]

    if len(speeds_clean) == 0:
        return out

    # 0: energy (sum of squared displacements)
    out[0] = float(np.nansum(dx[step_valid] ** 2 + dy[step_valid] ** 2))

    # 1-3: speed stats
    out[1] = float(np.mean(speeds_clean))
    out[2] = float(np.std(speeds_clean))
    out[3] = float(np.max(speeds_clean))

    # 4-5: angular velocity (absolute change in orientation per frame)
    orient_valid = np.isfinite(orients)
    if np.sum(orient_valid) >= 2:
        d_orient = np.diff(orients)
        # Wrap to [-pi, pi]
        d_orient = (d_orient + np.pi) % (2 * np.pi) - np.pi
        abs_d_orient = np.abs(d_orient)
        step_o_valid = orient_valid[:-1] & orient_valid[1:]
        av_clean = abs_d_orient[step_o_valid]
        if len(av_clean) > 0:
            out[4] = float(np.mean(av_clean))
            out[5] = float(np.std(av_clean))

    # 6: fraction time moving
    out[6] = float(np.mean(speeds_clean > stationary_threshold))

    # 7-9: area stats
    areas_clean = areas[np.isfinite(areas)]
    if len(areas_clean) > 0:
        out[7] = float(np.mean(areas_clean))
        out[8] = float(np.std(areas_clean))
        out[9] = float(np.ptp(areas_clean))

    # 10: area slope (linear trend)
    if len(areas_clean) >= 2:
        t = np.arange(len(areas_clean), dtype=np.float64)
        # Simple linear regression slope
        t_mean = t.mean()
        a_mean = areas_clean.mean()
        num = np.sum((t - t_mean) * (areas_clean - a_mean))
        den = np.sum((t - t_mean) ** 2)
        out[10] = float(num / den) if den > 1e-12 else 0.0

    # 11: mean eccentricity
    ecc_clean = eccs[np.isfinite(eccs)]
    if len(ecc_clean) > 0:
        out[11] = float(np.mean(ecc_clean))

    return out


# ---------------------------------------------------------------------------
# Interaction sequence features
# ---------------------------------------------------------------------------

def _interaction_sequence_features(
    window: np.ndarray,
    close_threshold: float,
    chase_max_lag: int,
) -> np.ndarray:
    """Compute 9 interaction sequence features within a window.

    Parameters
    ----------
    window : shape (W, 21)
    close_threshold : distance below which mice are "close" (already normalized)
    chase_max_lag : max frame lag for cross-correlation
    """
    out = np.full(N_INTERACTION_SEQ, np.nan, dtype=np.float64)

    cdist_series = window[:, _I_CDIST]
    min_cont = window[:, _I_MIN_CONTOUR]
    overlap = window[:, _I_OVERLAP]
    combined_area = window[:, _I_COMBINED_AREA]
    rel_heading = window[:, _I_HEAD_AB]  # heading A->B as proxy for mutual facing

    valid_i = np.isfinite(cdist_series)
    if np.sum(valid_i) < 3:
        return out

    # 0: approach/retreat rate = mean(diff(centroid_distance))
    #    negative = approaching, positive = retreating
    d_cdist = np.diff(cdist_series)
    step_valid = valid_i[:-1] & valid_i[1:]
    d_clean = d_cdist[step_valid]
    if len(d_clean) > 0:
        out[0] = float(np.mean(d_clean))

    # 1: parallel movement score
    #    correlation of velocity vectors of the two mice
    cx_a = window[:, _A_CX]
    cy_a = window[:, _A_CY]
    cx_b = window[:, _B_CX]
    cy_b = window[:, _B_CY]
    both_valid = (
        np.isfinite(cx_a) & np.isfinite(cy_a)
        & np.isfinite(cx_b) & np.isfinite(cy_b)
    )

    if np.sum(both_valid) >= 3:
        dx_a = np.diff(cx_a)
        dy_a = np.diff(cy_a)
        dx_b = np.diff(cx_b)
        dy_b = np.diff(cy_b)
        sv = both_valid[:-1] & both_valid[1:]

        if np.sum(sv) >= 2:
            # Dot product of velocity vectors, normalized
            dot = dx_a[sv] * dx_b[sv] + dy_a[sv] * dy_b[sv]
            mag_a = np.hypot(dx_a[sv], dy_a[sv])
            mag_b = np.hypot(dx_b[sv], dy_b[sv])
            denom = mag_a * mag_b
            # Only count steps where both are actually moving
            moving = denom > 1e-8
            if np.sum(moving) > 0:
                cos_sim = dot[moving] / denom[moving]
                out[1] = float(np.mean(cos_sim))

    # 2-3: chase metric (cross-correlation of centroid trajectories with lag)
    if np.sum(both_valid) >= 2 * chase_max_lag + 2:
        speed_a = np.hypot(np.diff(cx_a), np.diff(cy_a))
        speed_b = np.hypot(np.diff(cx_b), np.diff(cy_b))
        sv = both_valid[:-1] & both_valid[1:]
        sa = np.where(sv, speed_a, 0.0)
        sb = np.where(sv, speed_b, 0.0)

        # Normalize for cross-correlation
        sa_n = sa - np.mean(sa)
        sb_n = sb - np.mean(sb)
        norm = np.sqrt(np.sum(sa_n ** 2) * np.sum(sb_n ** 2))

        if norm > 1e-12:
            best_corr = -2.0
            best_lag = 0
            for lag in range(-chase_max_lag, chase_max_lag + 1):
                if lag == 0:
                    cc = np.sum(sa_n * sb_n) / norm
                elif lag > 0:
                    cc = np.sum(sa_n[lag:] * sb_n[:-lag]) / norm
                else:
                    cc = np.sum(sa_n[:lag] * sb_n[-lag:]) / norm
                if cc > best_corr:
                    best_corr = cc
                    best_lag = lag
            out[2] = float(best_corr)
            out[3] = float(best_lag)

    # 4: fraction of time in close proximity
    cdist_clean = cdist_series[valid_i]
    out[4] = float(np.mean(cdist_clean < close_threshold))

    # 5: mean relative heading
    head_ab = window[:, _I_HEAD_AB]
    head_ba = window[:, _I_HEAD_BA]
    head_valid = np.isfinite(head_ab) & np.isfinite(head_ba)
    if np.sum(head_valid) > 0:
        # Average of both headings: low = facing each other
        out[5] = float(np.mean((head_ab[head_valid] + head_ba[head_valid]) / 2.0))

    # 6-7: min contour distance stats
    mc_clean = min_cont[np.isfinite(min_cont)]
    if len(mc_clean) > 0:
        out[6] = float(np.mean(mc_clean))
        out[7] = float(np.std(mc_clean))

    # 8: mean overlap fraction (overlap / combined_area)
    ov_valid = np.isfinite(overlap) & np.isfinite(combined_area) & (combined_area > 0)
    if np.sum(ov_valid) > 0:
        out[8] = float(np.mean(overlap[ov_valid] / combined_area[ov_valid]))

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_sequence_features(
    frame_features: np.ndarray,
    *,
    window_size: int = 90,
    stride: int = 30,
    video_diagonal: float = 1.0,
    video_area: float = 1.0,
    stationary_speed_threshold: float = 0.001,
    close_proximity_threshold: float = 0.05,
    chase_max_lag: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate frame features into sequence-level vectors via sliding windows.

    Parameters
    ----------
    frame_features : shape (T, 21) -- frame-level features for one video
    window_size : frames per window
    stride : frame step between windows
    video_diagonal : sqrt(w^2 + h^2) for distance normalization
    video_area : w * h for area normalization
    stationary_speed_threshold : normalized speed threshold
    close_proximity_threshold : normalized distance threshold

    Returns
    -------
    features : float64 (n_windows, 33)
    starts : int32 (n_windows,) -- frame indices of window starts
    """
    t_len = frame_features.shape[0]
    starts = window_starts(t_len, window_size, stride)

    if len(starts) == 0:
        return (
            np.empty((0, N_SEQUENCE_FEATURES), dtype=np.float64),
            np.empty(0, dtype=np.int32),
        )

    # Pre-normalize frame features for invariance:
    # distances -> / diagonal, areas -> / video_area, centroids -> / (w, h)
    ff = frame_features.copy()
    diag = max(video_diagonal, 1.0)
    va = max(video_area, 1.0)

    # Centroids: normalize by dimensions (approximate: divide by diagonal)
    ff[:, _A_CX] /= diag
    ff[:, _A_CY] /= diag
    ff[:, _B_CX] /= diag
    ff[:, _B_CY] /= diag

    # Areas
    ff[:, _A_AREA] /= va
    ff[:, _A_AREA + N_SINGLE_FEATURES] /= va  # B area

    # Major/minor axis lengths
    ff[:, _A_MAJOR] /= diag
    ff[:, _A_MINOR] /= diag
    ff[:, _A_MAJOR + N_SINGLE_FEATURES] /= diag
    ff[:, _A_MINOR + N_SINGLE_FEATURES] /= diag

    # Interaction distances
    ff[:, _I_CDIST] /= diag
    ff[:, _I_MIN_CONTOUR] /= diag

    # Interaction areas
    ff[:, _I_OVERLAP] /= va
    ff[:, _I_COMBINED_AREA] /= va

    out = np.full((len(starts), N_SEQUENCE_FEATURES), np.nan, dtype=np.float64)

    for i, s in enumerate(starts):
        w = ff[s : s + window_size]

        obj_a = _object_sequence_features(
            w, _A_CX, _A_CY, _A_AREA, _A_ORIENT, _A_ECC,
            stationary_threshold=stationary_speed_threshold,
        )
        obj_b = _object_sequence_features(
            w, _B_CX, _B_CY, _B_AREA, _B_ORIENT, _B_ECC,
            stationary_threshold=stationary_speed_threshold,
        )
        inter = _interaction_sequence_features(
            w,
            close_threshold=close_proximity_threshold,
            chase_max_lag=chase_max_lag,
        )

        out[i, :N_PER_OBJECT_SEQ] = obj_a
        out[i, N_PER_OBJECT_SEQ : 2 * N_PER_OBJECT_SEQ] = obj_b
        out[i, 2 * N_PER_OBJECT_SEQ :] = inter

    return out, np.array(starts, dtype=np.int32)
