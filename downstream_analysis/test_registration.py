from __future__ import annotations

import cv2
import numpy as np

from .registration import transform_xy, warp_mask


def test_transform_xy_maps_source_quad_to_target_square() -> None:
    source = np.asarray([[20, 10], [90, 20], [80, 90], [10, 80]], dtype=np.float32)
    target = np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    registration = {"homography": matrix.tolist(), "target_size": 100}

    actual = transform_xy(source, registration)

    np.testing.assert_allclose(actual, target, atol=1e-4)


def test_warp_mask_returns_canonical_square_binary_mask() -> None:
    source = np.asarray([[20, 10], [90, 20], [80, 90], [10, 80]], dtype=np.int32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillConvexPoly(mask, source, 1)
    target = np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), target)

    warped = warp_mask(mask, {"homography": matrix.tolist(), "target_size": 100})

    assert warped.shape == (100, 100)
    assert warped.dtype == np.bool_
    assert float(warped.mean()) > 0.95


def test_unregistered_mask_is_unchanged() -> None:
    mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(warp_mask(mask, None), mask)


def test_distance_and_path_length_are_rotation_invariant() -> None:
    xy = np.asarray([[10.0, 20.0], [15.0, 24.0], [12.0, 31.0]])
    quarter_turn = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    rotated = xy @ quarter_turn.T

    original_steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    rotated_steps = np.linalg.norm(np.diff(rotated, axis=0), axis=1)

    np.testing.assert_allclose(original_steps, rotated_steps)


def test_morphology_pixel_delta_is_exactly_reversible() -> None:
    before = np.zeros((20, 20), dtype=bool)
    before[4:16, 4:16] = True
    before[9, 9] = False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    after = cv2.morphologyEx(before.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    added = np.flatnonzero(after & ~before)
    removed = np.flatnonzero(before & ~after)

    undone = after.reshape(-1).copy()
    undone[added] = False
    undone[removed] = True

    np.testing.assert_array_equal(undone.reshape(before.shape), before)


def test_polygon_convex_hull_fills_concavity() -> None:
    vertices = np.asarray([[2, 2], [17, 2], [9, 8], [17, 17], [2, 17]], dtype=np.int32)
    hull = cv2.convexHull(vertices.reshape(-1, 1, 2))
    mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 1)

    assert mask[9, 9] == 1
    assert cv2.contourArea(hull) > cv2.contourArea(vertices.reshape(-1, 1, 2))


def test_polygon_border_coordinates_are_clipped() -> None:
    vertices = np.asarray([[-1e-6, 0.5], [0.5, -1e-6], [1.000001, 1.0]])
    clipped = np.clip(vertices, 0.0, 1.0)

    assert float(clipped.min()) == 0.0
    assert float(clipped.max()) == 1.0
