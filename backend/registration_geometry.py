"""Geometry for partial-view square calibration with radial lens distortion."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

EDGE_NAMES = ("top", "right", "bottom", "left")


def camera_matrix_for_size(width: int, height: int) -> np.ndarray:
    focal = float(max(width, height))
    return np.asarray(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix).reshape(-1, 2)


def fit_line_tls(points: np.ndarray) -> np.ndarray:
    """Line ax+by+c=0 with unit (a,b), fit by total least squares."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    direction = vh[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    return np.asarray([normal[0], normal[1], -normal @ center], dtype=np.float64)


def intersect_lines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.cross(a, b)
    if abs(float(p[2])) < 1e-9:
        raise ValueError("Adjacent fitted floor edges are nearly parallel in the image")
    return p[:2] / p[2]


def fit_edge_registration(
    edge_points_normalized: dict[str, list[list[float]]],
    width: int,
    height: int,
    target_size: int,
) -> dict:
    """Fit k1/k2 via plumb lines, extrapolate corners, then solve homography."""
    width, height = int(width), int(height)
    groups: dict[str, np.ndarray] = {}
    for name in EDGE_NAMES:
        raw = np.asarray(edge_points_normalized.get(name) or [], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1:] != (2,) or len(raw) < 2:
            raise ValueError(f"Edge '{name}' needs at least two points")
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Edge '{name}' contains invalid coordinates")
        raw = np.clip(raw, 0.0, 1.0)
        groups[name] = np.column_stack([raw[:, 0] * (width - 1), raw[:, 1] * (height - 1)])
    if sum(len(v) >= 3 for v in groups.values()) < 2:
        raise ValueError("At least two edges need three or more points to estimate lens curvature")

    camera_matrix = camera_matrix_for_size(width, height)
    diagonal = float(np.hypot(width, height))

    def residual(coeffs: np.ndarray) -> np.ndarray:
        distortion = np.asarray([coeffs[0], coeffs[1], 0.0, 0.0, 0.0], dtype=np.float64)
        out: list[np.ndarray] = []
        for points in groups.values():
            corrected = undistort_points(points, camera_matrix, distortion)
            line = fit_line_tls(corrected)
            out.append((corrected @ line[:2] + line[2]) / diagonal)
        # Weak regularizer avoids extreme coefficients on nearly straight/noisy edges.
        out.append(np.asarray([1e-3 * coeffs[0], 1e-3 * coeffs[1]]))
        return np.concatenate(out)

    fit = least_squares(
        residual,
        x0=np.zeros(2, dtype=np.float64),
        bounds=(np.asarray([-1.5, -1.5]), np.asarray([1.5, 1.5])),
        loss="soft_l1",
        f_scale=1e-3,
        max_nfev=1000,
    )
    distortion = np.asarray([fit.x[0], fit.x[1], 0.0, 0.0, 0.0], dtype=np.float64)
    corrected_groups = {
        name: undistort_points(points, camera_matrix, distortion)
        for name, points in groups.items()
    }
    lines = {name: fit_line_tls(points) for name, points in corrected_groups.items()}
    corners = np.stack(
        [
            intersect_lines(lines["top"], lines["left"]),
            intersect_lines(lines["top"], lines["right"]),
            intersect_lines(lines["bottom"], lines["right"]),
            intersect_lines(lines["bottom"], lines["left"]),
        ]
    ).astype(np.float32)
    size = max(2, int(target_size))
    target = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, target)
    rms = float(np.sqrt(np.mean(np.square(residual(fit.x)[:-2]))) * diagonal)
    return {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "undistorted_corners": corners.astype(float).tolist(),
        "homography": homography.astype(float).tolist(),
        "straightness_rms_pixels": rms,
        "fit_success": bool(fit.success),
    }
