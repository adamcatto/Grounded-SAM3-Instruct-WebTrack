"""Per-frame feature extraction from binary segmentation masks.

Each frame produces a 21-element vector:
  [obj_A(7), obj_B(7), interaction(7)]

Per-mask features (7):
  centroid_x, centroid_y, area, major_axis_length, minor_axis_length,
  orientation, eccentricity

Interaction features (7):
  centroid_distance, min_contour_distance, overlap_area,
  relative_body_angle, heading_a_to_b, heading_b_to_a, combined_area
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SINGLE_FEATURES = 7
N_INTERACTION_FEATURES = 7
N_FRAME_FEATURES = 2 * N_SINGLE_FEATURES + N_INTERACTION_FEATURES  # 21

SINGLE_FEATURE_NAMES = [
    "centroid_x",
    "centroid_y",
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "eccentricity",
]

INTERACTION_FEATURE_NAMES = [
    "centroid_distance",
    "min_contour_distance",
    "overlap_area",
    "relative_body_angle",
    "heading_a_to_b",
    "heading_b_to_a",
    "combined_area",
]

FRAME_FEATURE_NAMES = (
    [f"a_{n}" for n in SINGLE_FEATURE_NAMES]
    + [f"b_{n}" for n in SINGLE_FEATURE_NAMES]
    + INTERACTION_FEATURE_NAMES
)


# ---------------------------------------------------------------------------
# Single-mask features
# ---------------------------------------------------------------------------

def compute_single_mask_features(mask: np.ndarray) -> np.ndarray | None:
    """Compute 7 geometric features from a binary mask.

    Returns float64 array of shape (7,) or None if mask is empty.
    """
    ys, xs = np.nonzero(mask)
    n_pixels = len(xs)
    if n_pixels == 0:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())
    area = float(n_pixels)

    # PCA of mask pixel coordinates for body axes
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - np.array([[cx, cy]])
    cov = (centered.T @ centered) / max(n_pixels, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending order; largest is last
    major_val = max(eigvals[-1], 0.0)
    minor_val = max(eigvals[0], 0.0)
    major_len = np.sqrt(major_val)
    minor_len = np.sqrt(minor_val)

    # Orientation: angle of major axis (eigvec for largest eigenvalue)
    major_vec = eigvecs[:, -1]
    orientation = float(np.arctan2(major_vec[1], major_vec[0]))

    # Eccentricity
    if major_len > 1e-8:
        ratio = minor_len / major_len
        eccentricity = float(np.sqrt(max(1.0 - ratio * ratio, 0.0)))
    else:
        eccentricity = 0.0

    return np.array(
        [cx, cy, area, major_len, minor_len, orientation, eccentricity],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Contour extraction and sampling
# ---------------------------------------------------------------------------

def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels of a binary mask.

    Returns float64 array of shape (N, 2) as (x, y) coordinates.
    Uses erosion: boundary = mask AND NOT eroded(mask).
    """
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask.astype(bool))
    boundary = mask.astype(bool) & ~eroded
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        # Fallback: if erosion leaves nothing, use all mask pixels
        ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _sample_contour(contour_pts: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample n points from contour array (N, 2)."""
    if len(contour_pts) <= n:
        return contour_pts
    indices = np.linspace(0, len(contour_pts) - 1, n, dtype=int)
    return contour_pts[indices]


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def compute_interaction_features(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    *,
    contour_sample_n: int = 200,
) -> np.ndarray | None:
    """Compute 7 interaction features between two masks.

    Parameters
    ----------
    mask_a, mask_b : binary masks (H, W)
    feats_a, feats_b : single-mask feature arrays from compute_single_mask_features
    contour_sample_n : number of contour points to sample for min-distance

    Returns float64 array of shape (7,) or None if either mask is empty.
    """
    if feats_a is None or feats_b is None:
        return None

    cx_a, cy_a, area_a, _, _, orient_a, _ = feats_a
    cx_b, cy_b, area_b, _, _, orient_b, _ = feats_b

    # 1. Centroid distance
    centroid_dist = float(np.hypot(cx_b - cx_a, cy_b - cy_a))

    # 2. Minimum contour distance (sampled)
    cont_a = _sample_contour(_boundary_pixels(mask_a), contour_sample_n)
    cont_b = _sample_contour(_boundary_pixels(mask_b), contour_sample_n)
    if len(cont_a) > 0 and len(cont_b) > 0:
        dists = cdist(cont_a, cont_b)
        min_contour_dist = float(dists.min())
    else:
        min_contour_dist = centroid_dist

    # 3. Overlap area
    overlap = float(np.count_nonzero(mask_a.astype(bool) & mask_b.astype(bool)))

    # 4. Relative body angle: |orient_a - orient_b| wrapped to [0, pi/2]
    angle_diff = abs(orient_a - orient_b) % np.pi
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff
    relative_body_angle = float(angle_diff)

    # 5-6. Heading: angle from one's centroid to the other, relative to body axis
    # heading = 0 means facing the other mouse; pi means facing away
    dx_ab = cx_b - cx_a
    dy_ab = cy_b - cy_a
    angle_to_b = np.arctan2(dy_ab, dx_ab)
    heading_a_to_b = float(_wrap_angle(angle_to_b - orient_a))

    dx_ba = cx_a - cx_b
    dy_ba = cy_a - cy_b
    angle_to_a = np.arctan2(dy_ba, dx_ba)
    heading_b_to_a = float(_wrap_angle(angle_to_a - orient_b))

    # 7. Combined area
    combined_area = float(area_a + area_b)

    return np.array(
        [
            centroid_dist,
            min_contour_dist,
            overlap,
            relative_body_angle,
            heading_a_to_b,
            heading_b_to_a,
            combined_area,
        ],
        dtype=np.float64,
    )


def _wrap_angle(a: float) -> float:
    """Wrap angle to [0, pi] (absolute heading offset)."""
    a = a % (2 * np.pi)
    if a > np.pi:
        a = 2 * np.pi - a
    return a


# ---------------------------------------------------------------------------
# Combined frame feature vector
# ---------------------------------------------------------------------------

def extract_frame_features(
    masks: dict[str, np.ndarray],
    obj_a_key: str,
    obj_b_key: str,
    *,
    contour_sample_n: int = 200,
) -> np.ndarray:
    """Extract the full 21-element feature vector for one frame.

    Returns float64 array of shape (21,). NaN-filled if a mask is missing/empty.
    """
    out = np.full(N_FRAME_FEATURES, np.nan, dtype=np.float64)

    mask_a = masks.get(obj_a_key)
    mask_b = masks.get(obj_b_key)

    feats_a = None
    feats_b = None

    if mask_a is not None:
        feats_a = compute_single_mask_features(mask_a)
    if mask_b is not None:
        feats_b = compute_single_mask_features(mask_b)

    if feats_a is not None:
        out[:N_SINGLE_FEATURES] = feats_a
    if feats_b is not None:
        out[N_SINGLE_FEATURES : 2 * N_SINGLE_FEATURES] = feats_b

    if mask_a is not None and mask_b is not None:
        inter = compute_interaction_features(
            mask_a,
            mask_b,
            feats_a,
            feats_b,
            contour_sample_n=contour_sample_n,
        )
        if inter is not None:
            out[2 * N_SINGLE_FEATURES :] = inter

    return out
