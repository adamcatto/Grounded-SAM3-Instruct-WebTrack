"""Anchor frame list for whole-video batch propagation (shared by server and project metadata)."""

from __future__ import annotations

STREAM_BATCH_SIZE = 1000  # default frames per propagation batch / anchor interval
ANCHOR_BATCH_SIZE_MIN = 10
ANCHOR_BATCH_SIZE_MAX = 10_000


def normalize_anchor_batch_size(value: object, default: int = STREAM_BATCH_SIZE) -> int:
    """Clamp and validate per-video anchor / propagation batch size."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(ANCHOR_BATCH_SIZE_MIN, min(ANCHOR_BATCH_SIZE_MAX, n))


def video_anchor_batch_size(video: dict, default: int = STREAM_BATCH_SIZE) -> int:
    return normalize_anchor_batch_size(video.get("anchor_batch_size"), default)


def anchor_labeling_has_started(video: dict, batch_size: int | None = None) -> bool:
    """True once any anchor frame has been committed or point-labeled."""
    if video.get("annotated_anchors"):
        return True
    bs = batch_size if batch_size is not None else video_anchor_batch_size(video)
    start = int(video.get("start_frame") or 0)
    num_frames = int(video.get("num_frames") or 0)
    if num_frames <= 0:
        return False
    anchor_set = set(compute_anchor_frames(start, num_frames, bs))
    for obj_prompts in (video.get("point_prompts") or {}).values():
        if not isinstance(obj_prompts, dict):
            continue
        for frame_key in obj_prompts:
            try:
                if int(frame_key) in anchor_set:
                    return True
            except (ValueError, TypeError):
                pass
    return False


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
    batch_size: int | None = None,
) -> bool:
    """True if every computed anchor frame has evidence of labeling.

    A frame counts as labeled if it appears in ``annotated_anchors`` OR if at
    least one object has point prompts saved for that frame.  This handles the
    case where the commit didn't persist but the annotation data was saved.
    """
    start = int(video.get("start_frame") or 0)
    num_frames = int(video["num_frames"])
    bs = batch_size if batch_size is not None else video_anchor_batch_size(video)
    required = compute_anchor_frames(start, num_frames, bs)

    annotated = set(video.get("annotated_anchors") or [])

    # Also count frames that have point prompts for any object
    for obj_prompts in (video.get("point_prompts") or {}).values():
        if isinstance(obj_prompts, dict):
            for frame_key in obj_prompts:
                try:
                    annotated.add(int(frame_key))
                except (ValueError, TypeError):
                    pass

    return bool(required) and all(a in annotated for a in required)
