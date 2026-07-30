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
        "target_size": int(registration.get("target_size") or 1000),
    }


def registration_signature(registration: dict[str, Any] | None) -> str:
    """Stable cache key; empty string represents legacy/unregistered geometry."""
    if not registration:
        return ""
    return json.dumps(
        {
            "target_size": int(registration["target_size"]),
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
    matrix = np.asarray(registration["homography"], dtype=np.float64).reshape(3, 3)
    size = max(2, int(registration["target_size"]))
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
    warped = cv2.warpPerspective(
        source,
        matrix,
        (size, size),
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
    matrix = np.asarray(registration["homography"], dtype=np.float64).reshape(3, 3)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
