from __future__ import annotations

import numpy as np

from registration_geometry import camera_matrix_for_size, fit_edge_registration


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

