"""Anchor frame list for whole-video batch propagation (shared by server and project metadata)."""

from __future__ import annotations

STREAM_BATCH_SIZE = 1000  # frames per propagation batch / anchor interval

# After the first ANCHOR_MANUAL_PREFIX_COUNT anchors are committed in the UI,
# the remainder of the anchor frames can be filled by sequential mask propagation
# (see server anchor-remainder SSE endpoint).
ANCHOR_MANUAL_PREFIX_COUNT = 5


def compute_anchor_frames(start_frame: int, num_frames: int, batch_size: int = STREAM_BATCH_SIZE) -> list[int]:
    """Compute anchor frames for bidirectional batch propagation.

    The first anchor is at start_frame so the user labels the very beginning of
    the video. Subsequent anchors sit at the midpoint of each following batch
    window. The last frame is always an anchor when it is not already on the grid.
    """
    last = num_frames - 1
    anchors = list(range(start_frame, num_frames, batch_size))
    if not anchors:
        anchors = [last]
    elif anchors[-1] != last:
        anchors.append(last)
    return anchors


def is_anchor_labeling_complete(
    video: dict,
    batch_size: int = STREAM_BATCH_SIZE,
) -> bool:
    """True if every computed anchor frame index appears in annotated_anchors."""
    start = int(video.get("start_frame") or 0)
    num_frames = int(video["num_frames"])
    required = compute_anchor_frames(start, num_frames, batch_size)
    annotated = set(video.get("annotated_anchors") or [])
    return bool(required) and all(a in annotated for a in required)
