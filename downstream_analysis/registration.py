"""Project-level camera registration utilities.

Registration matrices map source video pixels into a common square coordinate
system. Masks use nearest-neighbor resampling so object identities and binary
areas remain discrete.
"""

from __future__ import annotations

from typing import Any
import json
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import Delaunay


@lru_cache(maxsize=32)
def _undistort_maps(
    width: int,
    height: int,
    camera_json: str,
    distortion_json: str,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(json.loads(camera_json), dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(json.loads(distortion_json), dtype=np.float64)
    return cv2.initUndistortRectifyMap(
        camera, distortion, None, camera, (width, height), cv2.CV_32FC1,
    )


def registration_for_video(
    project_config: dict[str, Any],
    video_id: str,
) -> dict[str, Any] | None:
    registration = project_config.get("registration") or {}
    entry = (registration.get("videos") or {}).get(str(video_id))
    if not entry or not entry.get("registered") or not entry.get("homography"):
        return None
    return {
        **entry,
        "source_width": int((project_config.get("videos") or {}).get(str(video_id), {}).get("width") or 1),
        "source_height": int((project_config.get("videos") or {}).get(str(video_id), {}).get("height") or 1),
        "target_size": int(registration.get("target_size") or 1000),
        "canvas_width": int(registration.get("canvas_width") or registration.get("target_size") or 1000),
        "canvas_height": int(registration.get("canvas_height") or registration.get("target_size") or 1000),
        "canvas_offset": list(registration.get("canvas_offset") or [0.0, 0.0]),
        "warp_mode": registration.get("warp_mode"),
    }


def registration_signature(registration: dict[str, Any] | None) -> str:
    """Stable cache key; empty string represents legacy/unregistered geometry."""
    if not registration:
        return ""
    return json.dumps(
        {
            "target_size": int(registration["target_size"]),
            "canvas_width": int(registration.get("canvas_width") or registration["target_size"]),
            "canvas_height": int(registration.get("canvas_height") or registration["target_size"]),
            "canvas_offset": registration.get("canvas_offset") or [0.0, 0.0],
            "warp_mode": registration.get("warp_mode"),
            "source_width": registration.get("source_width"),
            "source_height": registration.get("source_height"),
            "source_corners": registration.get("source_corners"),
            "homography": registration["homography"],
            "camera_matrix": registration.get("camera_matrix"),
            "distortion_coefficients": registration.get("distortion_coefficients"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def warp_mask(
    mask: np.ndarray,
    registration: dict[str, Any] | None,
) -> np.ndarray:
    """Warp a binary mask into the canonical square; return unchanged if unregistered."""
    dense = np.squeeze(np.asarray(mask))
    if not registration:
        return dense
    size = max(2, int(registration["target_size"]))
    canvas_width = max(2, int(registration.get("canvas_width") or size))
    canvas_height = max(2, int(registration.get("canvas_height") or size))
    source = dense.astype(np.uint8)
    if registration.get("camera_matrix") and registration.get("distortion_coefficients"):
        h, w = source.shape[:2]
        map_x, map_y = _undistort_maps(
            w,
            h,
            json.dumps(registration["camera_matrix"], separators=(",", ":")),
            json.dumps(registration["distortion_coefficients"], separators=(",", ":")),
        )
        source = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    if registration.get("warp_mode") == "bounded_full_frame_mesh":
        map_x, map_y = _cached_bounded_mesh_maps(
            source.shape[1],
            source.shape[0],
            json.dumps(registration, sort_keys=True, separators=(",", ":")),
        )
        warped = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpPerspective(
            source,
            _canvas_homography(registration),
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return warped.astype(bool)


def warp_masks(
    masks: dict[str, np.ndarray],
    registration: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if not registration:
        return masks
    return {str(key): warp_mask(mask, registration) for key, mask in masks.items()}


def transform_xy(
    points_xy: np.ndarray,
    registration: dict[str, Any] | None,
) -> np.ndarray:
    """Transform Nx2 source-pixel points into the canonical square."""
    points = np.asarray(points_xy, dtype=np.float64)
    if not registration or points.size == 0:
        return points.copy()
    if registration.get("camera_matrix") and registration.get("distortion_coefficients"):
        camera = np.asarray(registration["camera_matrix"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(registration["distortion_coefficients"], dtype=np.float64)
        points = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            camera,
            distortion,
            P=camera,
        ).reshape(-1, 2)
    if registration.get("warp_mode") == "bounded_full_frame_mesh":
        transformed = _bounded_transform_points(
            points,
            int(registration.get("source_width") or 1),
            int(registration.get("source_height") or 1),
            registration,
        )
        return transformed + np.asarray(registration.get("canvas_offset") or [0.0, 0.0])
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        _canvas_homography(registration),
    ).reshape(-1, 2)


def _canvas_homography(registration: dict[str, Any]) -> np.ndarray:
    if registration.get("warp_mode") == "affine_full_frame":
        corners = np.asarray(registration["source_corners"], dtype=np.float64).reshape(4, 2)
        size = max(2, int(registration["target_size"]))
        target = np.asarray(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float64,
        )
        coefficients = np.linalg.lstsq(
            np.column_stack([corners, np.ones(4)]),
            target,
            rcond=None,
        )[0]
        matrix = np.vstack([coefficients.T, [0.0, 0.0, 1.0]])
    else:
        matrix = np.asarray(registration["homography"], dtype=np.float64).reshape(3, 3)
    dx, dy = (registration.get("canvas_offset") or [0.0, 0.0])[:2]
    translation = np.asarray(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translation @ matrix


def _bounded_transform_points(
    points: np.ndarray,
    width: int,
    height: int,
    registration: dict[str, Any],
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    corners = np.asarray(registration["source_corners"], dtype=np.float64).reshape(4, 2)
    size = max(2, int(registration["target_size"]))
    target = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projective = np.asarray(registration["homography"], dtype=np.float64).reshape(3, 3)
    projected_h = homogeneous @ projective.T
    projected = projected_h[:, :2] / projected_h[:, 2, None]
    coefficients = np.linalg.lstsq(
        np.column_stack([corners, np.ones(4)]),
        target,
        rcond=None,
    )[0]
    affine = np.vstack([coefficients.T, [0.0, 0.0, 1.0]])
    affine_points = (homogeneous @ affine.T)[:, :2]
    signed = np.asarray(
        [cv2.pointPolygonTest(corners.astype(np.float32), tuple(point), True) for point in source],
        dtype=np.float64,
    )
    transition = 0.45 * float(np.hypot(width, height))
    floor_weight = np.clip(1.0 - np.maximum(0.0, -signed) / max(transition, 1.0), 0.0, 1.0)
    floor_weight = floor_weight * floor_weight * (3.0 - 2.0 * floor_weight)
    corner_h = np.column_stack([corners, np.ones(4)]) @ projective.T
    reference = max(float(np.min(np.abs(corner_h[:, 2]))), 1e-9)
    denominator = np.abs(projected_h[:, 2])
    stable = np.clip((denominator - 0.15 * reference) / (0.35 * reference), 0.0, 1.0)
    stable = stable * stable * (3.0 - 2.0 * stable)
    same_side = np.sign(projected_h[:, 2]) == np.sign(np.median(corner_h[:, 2]))
    weight = floor_weight * stable * same_side
    return weight[:, None] * projected + (1.0 - weight[:, None]) * affine_points


@lru_cache(maxsize=16)
def _cached_bounded_mesh_maps(
    width: int,
    height: int,
    registration_json: str,
) -> tuple[np.ndarray, np.ndarray]:
    registration = json.loads(registration_json)
    columns, rows = 41, 31
    x = np.linspace(0.0, width - 1.0, columns)
    y = np.linspace(0.0, height - 1.0, rows)
    source = np.asarray([(px, py) for py in y for px in x], dtype=np.float64)
    corners = np.asarray(registration["source_corners"], dtype=np.float64).reshape(4, 2)
    visible = (
        (corners[:, 0] >= 0)
        & (corners[:, 0] <= width - 1)
        & (corners[:, 1] >= 0)
        & (corners[:, 1] <= height - 1)
    )
    source = np.unique(np.round(np.vstack([source, corners[visible]]), 7), axis=0)
    destination = _bounded_transform_points(source, width, height, registration)
    destination += np.asarray(registration.get("canvas_offset") or [0.0, 0.0])
    canvas_width = max(2, int(registration.get("canvas_width") or registration["target_size"]))
    canvas_height = max(2, int(registration.get("canvas_height") or registration["target_size"]))
    map_x = np.full((canvas_height, canvas_width), -1.0, dtype=np.float32)
    map_y = np.full((canvas_height, canvas_width), -1.0, dtype=np.float32)
    for indices in Delaunay(source).simplices:
        src_triangle = source[indices].astype(np.float32)
        dst_triangle = destination[indices].astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(dst_triangle)
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(canvas_width, bx + bw), min(canvas_height, by + bh)
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
        selected = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(selected, polygon, 1)
        selected_bool = selected.astype(bool)
        map_x[y0:y1, x0:x1][selected_bool] = local_x[selected_bool]
        map_y[y0:y1, x0:x1][selected_bool] = local_y[selected_bool]
    return map_x, map_y
