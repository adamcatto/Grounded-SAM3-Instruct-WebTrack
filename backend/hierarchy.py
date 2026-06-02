"""
Sub-object containment helpers (pure numpy / cv2, no SAM dependency).

Objects form a tree via each object's optional ``parent_id`` field.  A child mask
must always be contained within its parent's mask.  These helpers enforce that
("hard-clip / drop"), build the small point-blob masks used by ``kind == "point"``
sub-objects, and sample negative seed points just outside a parent for guiding
segmentation children inward.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is a hard dep elsewhere
    cv2 = None

from project_manager import iter_objects_topdown


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.squeeze(np.asarray(mask)) > 0


def apply_containment(frame_masks: dict, objects: dict) -> dict:
    """Clip every child mask to its parent, cascading top-down.

    *frame_masks* maps obj_id (str) -> binary mask (H, W).  *objects* is the video's
    object dict (each value may carry a ``parent_id``).  Returns a new dict of bool
    masks with each child intersected with its (already-clipped) parent.  If a
    parent has no mask on this frame, the child is dropped (empty mask) — strict
    hierarchy.  Objects without a parent are passed through unchanged.
    """
    out: dict = {}
    for oid in iter_objects_topdown(objects):
        oid = str(oid)
        if oid not in frame_masks:
            continue
        m = _as_bool(frame_masks[oid])
        parent_id = objects.get(oid, {}).get("parent_id")
        if parent_id is not None:
            parent_id = str(parent_id)
            parent_m = out.get(parent_id)
            if parent_m is None or not parent_m.any():
                # Parent absent/empty on this frame → child cannot be contained.
                m = np.zeros_like(m, dtype=bool)
            else:
                if parent_m.shape != m.shape:
                    parent_m = _resize_bool(parent_m, m.shape)
                m = m & parent_m
        out[oid] = m
    # Preserve any masks for ids not present in `objects` (defensive; e.g. stale keys).
    for oid, m in frame_masks.items():
        if str(oid) not in out:
            out[str(oid)] = _as_bool(m)
    return out


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized > 0
    # Fallback nearest-neighbour without cv2.
    ys = (np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
    xs = (np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
    return mask[np.ix_(ys, xs)] > 0


def parent_bbox_px(parent_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Pixel-space (x, y, w, h) bounding box of a binary mask. (0,0,0,0) if empty."""
    m = _as_bool(parent_mask)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def point_blob_mask(center_xy_norm, parent_mask: np.ndarray, frac: float = 0.06) -> np.ndarray:
    """Filled disk centred at *center_xy_norm* (normalized [0,1] (x, y)), radius
    ``frac * min(parent_bbox_w, parent_bbox_h)``, intersected with the parent.

    Returns a bool mask the same shape as *parent_mask*.  Empty if the parent is
    empty.  Radius is clamped to at least 2 px so the blob never vanishes.
    """
    parent = _as_bool(parent_mask)
    h, w = parent.shape[:2]
    if not parent.any():
        return np.zeros((h, w), dtype=bool)
    _, _, bw, bh = parent_bbox_px(parent)
    radius = max(2, int(round(frac * min(bw, bh))))
    cx = int(round(float(center_xy_norm[0]) * w))
    cy = int(round(float(center_xy_norm[1]) * h))
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)
    blob = np.zeros((h, w), dtype=np.uint8)
    if cv2 is not None:
        cv2.circle(blob, (cx, cy), radius, 1, thickness=-1)
        blob_b = blob > 0
    else:
        yy, xx = np.ogrid[:h, :w]
        blob_b = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return blob_b & parent


def centroid_norm(mask: np.ndarray) -> tuple[float, float] | None:
    """Center-of-mass of a binary mask as normalized (x, y); None if empty."""
    m = _as_bool(mask)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    h, w = m.shape[:2]
    return (float(xs.mean()) / w, float(ys.mean()) / h)


def negative_seed_points(parent_mask: np.ndarray, k: int = 8, ring_px: int = 6) -> list[list[float]]:
    """Sample up to *k* normalized [x, y] points on a thin ring just OUTSIDE the
    parent mask.  Used as label-0 prompts to push a segmentation child inward.
    Returns [] if the parent is empty or cv2 is unavailable.
    """
    parent = _as_bool(parent_mask).astype(np.uint8)
    if not parent.any() or cv2 is None or k <= 0:
        return []
    h, w = parent.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    dilated = cv2.dilate(parent, kernel, iterations=1)
    ring = (dilated > 0) & (parent == 0)
    ys, xs = np.where(ring)
    if len(xs) == 0:
        return []
    # Evenly subsample the ring coordinates.
    idx = np.linspace(0, len(xs) - 1, num=min(k, len(xs))).astype(int)
    return [[float(xs[i]) / w, float(ys[i]) / h] for i in idx]
