"""Geometry for partial-view square calibration with radial lens distortion."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

EDGE_NAMES = ("top", "right", "bottom", "left")


def bounded_mesh_points(
    width: int,
    height: int,
    entry: dict,
    target_size: int,
    columns: int = 41,
    rows: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Control mesh: exact on the floor, bounded affine toward the frame boundary."""
    width, height = max(2, int(width)), max(2, int(height))
    x = np.linspace(0.0, width - 1.0, max(3, int(columns)))
    y = np.linspace(0.0, height - 1.0, max(3, int(rows)))
    source = np.asarray([(px, py) for py in y for px in x], dtype=np.float64)
    corners = np.asarray(entry["source_corners"], dtype=np.float64).reshape(4, 2)
    target_max = float(max(2, int(target_size)) - 1)
    target = np.asarray(
        [[0, 0], [target_max, 0], [target_max, target_max], [0, target_max]],
        dtype=np.float64,
    )
    visible = (
        (corners[:, 0] >= 0)
        & (corners[:, 0] <= width - 1)
        & (corners[:, 1] >= 0)
        & (corners[:, 1] <= height - 1)
    )
    source = np.vstack([source, corners[visible]])
    source = np.unique(np.round(source, decimals=7), axis=0)

    projective = np.asarray(entry["homography"], dtype=np.float64).reshape(3, 3)
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projected_h = homogeneous @ projective.T
    projected = projected_h[:, :2] / projected_h[:, 2, None]

    affine_coefficients = np.linalg.lstsq(
        np.column_stack([corners, np.ones(4)]),
        target,
        rcond=None,
    )[0]
    affine = np.vstack([affine_coefficients.T, [0.0, 0.0, 1.0]])
    affine_points = (homogeneous @ affine.T)[:, :2]

    signed_distance = np.asarray(
        [
            cv2.pointPolygonTest(corners.astype(np.float32), tuple(point), True)
            for point in source
        ],
        dtype=np.float64,
    )
    transition = 0.45 * float(np.hypot(width, height))
    floor_weight = np.clip(
        1.0 - np.maximum(0.0, -signed_distance) / max(transition, 1.0),
        0.0,
        1.0,
    )
    floor_weight = floor_weight * floor_weight * (3.0 - 2.0 * floor_weight)

    corner_h = np.column_stack([corners, np.ones(4)]) @ projective.T
    reference_denominator = max(float(np.min(np.abs(corner_h[:, 2]))), 1e-9)
    denominator = np.abs(projected_h[:, 2])
    stable = np.clip(
        (denominator - 0.15 * reference_denominator) / (0.35 * reference_denominator),
        0.0,
        1.0,
    )
    stable = stable * stable * (3.0 - 2.0 * stable)
    same_side = np.sign(projected_h[:, 2]) == np.sign(np.median(corner_h[:, 2]))
    weight = floor_weight * stable * same_side
    destination = weight[:, None] * projected + (1.0 - weight[:, None]) * affine_points
    if not np.all(np.isfinite(destination)):
        raise ValueError("Bounded registration mesh contains non-finite coordinates")
    return source, destination


def expanded_registration_canvas(
    videos: dict[str, dict],
    entries: dict[str, dict],
    target_size: int,
) -> dict:
    """Return one shared finite canvas containing every nonlinear-warped frame."""
    transformed_boundaries: list[np.ndarray] = []
    for vid, video in videos.items():
        entry = entries.get(vid) or {}
        if not entry.get("registered") or not entry.get("homography") or not entry.get("source_corners"):
            continue
        source, destination = bounded_mesh_points(
            int(video.get("width") or 1),
            int(video.get("height") or 1),
            entry,
            target_size,
        )
        _validated_mesh_triangles(source, destination, vid)
        transformed_boundaries.append(destination)

    if not transformed_boundaries:
        size = max(2, int(target_size))
        return {
            "version": 2,
            "method": "bounded_full_frame_mesh_with_floor_homography",
            "canvas_width": size,
            "canvas_height": size,
            "canvas_offset": [0.0, 0.0],
            "warp_mode": "bounded_full_frame_mesh",
        }

    all_points = np.concatenate(transformed_boundaries)
    floor_max = float(max(2, int(target_size)) - 1)
    minimum = np.minimum(np.floor(all_points.min(axis=0)), [0.0, 0.0])
    maximum = np.maximum(np.ceil(all_points.max(axis=0)), [floor_max, floor_max])
    width, height = (maximum - minimum + 1.0).astype(int)
    maximum_dimension = max(8192, int(target_size) * 8)
    if width > maximum_dimension or height > maximum_dimension:
        raise ValueError(
            f"Expanded registration canvas would be {width}x{height}; "
            "check the floor corners/edges for an unstable homography"
        )
    return {
        "version": 2,
        "method": "bounded_full_frame_mesh_with_floor_homography",
        "canvas_width": int(width),
        "canvas_height": int(height),
        "canvas_offset": [float(-minimum[0]), float(-minimum[1])],
        "warp_mode": "bounded_full_frame_mesh",
    }


def canvas_homography(homography: list[list[float]], offset: list[float] | None) -> np.ndarray:
    """Compose a canonical floor homography with the shared canvas translation."""
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    dx, dy = (offset or [0.0, 0.0])[:2]
    translation = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return translation @ matrix


def bounded_mesh_remap(
    width: int,
    height: int,
    entry: dict,
    target_size: int,
    canvas_width: int,
    canvas_height: int,
    canvas_offset: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build OpenCV inverse-remap arrays for the bounded full-frame mesh."""
    source, destination = bounded_mesh_points(width, height, entry, target_size)
    destination = destination + np.asarray(canvas_offset or [0.0, 0.0], dtype=np.float64)
    triangles = _validated_mesh_triangles(source, destination, "video")
    map_x = np.full((canvas_height, canvas_width), -1.0, dtype=np.float32)
    map_y = np.full((canvas_height, canvas_width), -1.0, dtype=np.float32)
    for indices in triangles:
        src_triangle = source[indices].astype(np.float32)
        dst_triangle = destination[indices].astype(np.float32)
        x, y, w, h = cv2.boundingRect(dst_triangle)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas_width, x + w), min(canvas_height, y + h)
        if x0 >= x1 or y0 >= y1:
            continue
        inverse = cv2.getAffineTransform(dst_triangle, src_triangle)
        grid_x, grid_y = np.meshgrid(
            np.arange(x0, x1, dtype=np.float32),
            np.arange(y0, y1, dtype=np.float32),
        )
        local_x = inverse[0, 0] * grid_x + inverse[0, 1] * grid_y + inverse[0, 2]
        local_y = inverse[1, 0] * grid_x + inverse[1, 1] * grid_y + inverse[1, 2]
        polygon = np.rint(dst_triangle - [x0, y0]).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 1)
        region_x = map_x[y0:y1, x0:x1]
        region_y = map_y[y0:y1, x0:x1]
        selected = mask.astype(bool)
        region_x[selected] = local_x[selected]
        region_y[selected] = local_y[selected]
    return map_x, map_y


def _validated_mesh_triangles(
    source: np.ndarray,
    destination: np.ndarray,
    label: str,
) -> np.ndarray:
    triangles = Delaunay(source).simplices

    def doubled_area(points: np.ndarray) -> np.ndarray:
        first = points[:, 1] - points[:, 0]
        second = points[:, 2] - points[:, 0]
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

    source_area = doubled_area(source[triangles])
    destination_area = doubled_area(destination[triangles])
    folded = np.count_nonzero(source_area * destination_area <= 1e-7)
    if folded:
        raise ValueError(
            f"Nonlinear registration mesh for {label} folds {folded} triangle(s); "
            "check the inferred floor corners"
        )
    return triangles


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
