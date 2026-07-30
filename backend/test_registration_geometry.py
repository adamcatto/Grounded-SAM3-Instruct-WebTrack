from __future__ import annotations

import cv2
import numpy as np

from registration_geometry import (
    camera_matrix_for_size,
    bounded_mesh_points,
    bounded_mesh_remap,
    expanded_registration_canvas,
    fit_edge_registration,
)


def test_partial_curved_edges_recover_radial_distortion_and_hidden_corner() -> None:
    width, height = 640, 480
    camera = camera_matrix_for_size(width, height)
    focal = camera[0, 0]
    cx, cy = camera[0, 2], camera[1, 2]
    k1, k2 = -0.25, 0.08
    corners = np.asarray([[100, 80], [540, 105], [520, 410], [75, 390]], dtype=float)

    def distort(points: np.ndarray) -> np.ndarray:
        x = (points[:, 0] - cx) / focal
        y = (points[:, 1] - cy) / focal
        radius2 = x * x + y * y
        scale = 1 + k1 * radius2 + k2 * radius2 * radius2
        return np.column_stack([focal * x * scale + cx, focal * y * scale + cy])

    def sample_edge(a: np.ndarray, b: np.ndarray, count: int = 12) -> np.ndarray:
        t = np.linspace(0, 1, count)[:, None]
        return distort(a[None, :] + t * (b - a)[None, :])

    sampled = {
        "top": sample_edge(corners[0], corners[1]),
        "right": sample_edge(corners[1], corners[2]),
        "bottom": sample_edge(corners[3], corners[2]),
        "left": sample_edge(corners[0], corners[3]),
    }
    normalized = {
        name: np.column_stack([points[:, 0] / (width - 1), points[:, 1] / (height - 1)]).tolist()
        for name, points in sampled.items()
    }

    fitted = fit_edge_registration(normalized, width, height, 1000)

    np.testing.assert_allclose(fitted["undistorted_corners"], corners, atol=0.2)
    assert fitted["straightness_rms_pixels"] < 0.1


def test_expanded_canvas_contains_full_frames_with_one_shared_offset() -> None:
    videos = {
        "left": {"width": 100, "height": 80},
        "right": {"width": 100, "height": 80},
    }
    corners = np.asarray([[20, 10], [79, 10], [79, 69], [20, 69]], dtype=np.float32)
    target = np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target).tolist()
    entries = {
        "left": {"registered": True, "homography": matrix, "source_corners": corners.tolist()},
        "right": {"registered": True, "homography": matrix, "source_corners": corners.tolist()},
    }

    canvas = expanded_registration_canvas(videos, entries, target_size=100)

    assert canvas["warp_mode"] == "bounded_full_frame_mesh"
    assert canvas["canvas_width"] > 100
    assert canvas["canvas_height"] > 100
    source, destination = bounded_mesh_points(100, 80, entries["left"], 100)
    shifted = destination + np.asarray(canvas["canvas_offset"])
    assert float(shifted.min()) >= 0.0
    assert float(shifted[:, 0].max()) <= canvas["canvas_width"] - 1
    assert float(shifted[:, 1].max()) <= canvas["canvas_height"] - 1


def test_bounded_mesh_supports_only_two_visible_inferred_corners() -> None:
    corners = np.asarray([[-30, 20], [80, 20], [80, 90], [-30, 90]], dtype=np.float32)
    target = np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
    entry = {
        "registered": True,
        "source_corners": corners.tolist(),
        "homography": cv2.getPerspectiveTransform(corners, target).tolist(),
    }
    videos = {"camera": {"width": 100, "height": 100}}
    entries = {"camera": entry}

    canvas = expanded_registration_canvas(videos, entries, 100)
    map_x, map_y = bounded_mesh_remap(
        100,
        100,
        entry,
        100,
        canvas["canvas_width"],
        canvas["canvas_height"],
        canvas["canvas_offset"],
    )

    assert map_x.shape == (canvas["canvas_height"], canvas["canvas_width"])
    assert map_y.shape == map_x.shape
    assert int(np.count_nonzero(map_x >= 0)) > 9000
