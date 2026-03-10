"""
Identity Tracker for video object segmentation.

Tracks per-object mask trajectory statistics, detects confusion windows
(frames where similar-looking objects are in proximity), computes adaptive
seeding strategies, and detects identity swaps post-confusion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Mask statistics ──────────────────────────────────────────────────────────

@dataclass
class MaskStats:
    centroid: tuple[float, float]        # normalized (cx, cy) in [0, 1]
    area: float                           # fraction of frame pixels
    bbox: tuple[float, float, float, float]  # normalized (x, y, w, h)
    iou_with_prev: float = 1.0           # IoU against previous frame mask


def compute_mask_stats(
    mask: np.ndarray,
    prev_mask: Optional[np.ndarray] = None,
) -> Optional[MaskStats]:
    """Compute statistics for a binary mask. Returns None if mask is empty."""
    m = np.squeeze(mask).astype(bool)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape
    ys, xs = np.where(m)
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
    area = float(m.sum()) / (h * w)

    iou = 1.0
    if prev_mask is not None:
        iou = compute_iou(m, np.squeeze(prev_mask).astype(bool))

    return MaskStats(centroid=(cx, cy), area=area, bbox=bbox, iou_with_prev=iou)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def hungarian_match(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    Simple greedy Hungarian-like matching (min cost assignment).
    For small N (≤10 instances), this greedy approach is sufficient.
    Returns list of (row_idx, col_idx) assignments.
    """
    import copy
    n_rows, n_cols = cost_matrix.shape
    mat = copy.deepcopy(cost_matrix)
    assigned_rows = set()
    assigned_cols = set()
    assignments = []
    for _ in range(min(n_rows, n_cols)):
        # Find the global minimum in unassigned cells
        best_cost = float("inf")
        best_r, best_c = -1, -1
        for r in range(n_rows):
            if r in assigned_rows:
                continue
            for c in range(n_cols):
                if c in assigned_cols:
                    continue
                if mat[r, c] < best_cost:
                    best_cost = mat[r, c]
                    best_r, best_c = r, c
        if best_r == -1:
            break
        assignments.append((best_r, best_c))
        assigned_rows.add(best_r)
        assigned_cols.add(best_c)
    return assignments


# ─── Per-object mask trajectory ───────────────────────────────────────────────

class MaskTrajectory:
    """
    Rolling-window statistics for one SAM object.
    Detects anomalous frames (large jumps in position/area/continuity).
    """

    WINDOW_K = 30  # frames in rolling window

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self._history: deque[tuple[int, MaskStats]] = deque(maxlen=self.WINDOW_K)
        self.last_clean_mask: Optional[np.ndarray] = None
        self.last_clean_frame: int = -1
        self.anomaly_scores: dict[int, float] = {}  # frame_idx → score

    def update(self, frame_idx: int, stats: MaskStats):
        self._history.append((frame_idx, stats))

    def compute_anomaly_score(self, new_stats: MaskStats) -> float:
        """
        Compute anomaly score for a new set of mask stats relative to recent history.
        Returns 0 if history is too short.
        """
        if len(self._history) < 3:
            return 0.0

        cxs = np.array([s.centroid[0] for _, s in self._history])
        cys = np.array([s.centroid[1] for _, s in self._history])
        areas = np.array([s.area for _, s in self._history])

        # Centroid z-score
        std_cx = float(np.std(cxs)) + 1e-4
        std_cy = float(np.std(cys)) + 1e-4
        std_area = float(np.std(areas)) + 1e-4
        mean_cx = float(np.mean(cxs))
        mean_cy = float(np.mean(cys))
        mean_area = float(np.mean(areas))

        dz_cx = abs(new_stats.centroid[0] - mean_cx) / std_cx
        dz_cy = abs(new_stats.centroid[1] - mean_cy) / std_cy
        centroid_z = float(np.sqrt(dz_cx ** 2 + dz_cy ** 2))
        area_z = abs(new_stats.area - mean_area) / std_area
        iou_drop = max(0.0, 1.0 - new_stats.iou_with_prev)

        # Weighted combination
        score = centroid_z * 0.4 + area_z * 0.3 + iou_drop * 3.0
        return score

    def record_frame(
        self,
        frame_idx: int,
        stats: MaskStats,
        mask: np.ndarray,
        anomaly_threshold: float,
    ):
        """Update trajectory, record anomaly score, and refresh last_clean_mask."""
        score = self.compute_anomaly_score(stats)
        self.update(frame_idx, stats)
        self.anomaly_scores[frame_idx] = score
        if score < anomaly_threshold:
            self.last_clean_mask = mask.copy()
            self.last_clean_frame = frame_idx
        return score


# ─── Text similarity ──────────────────────────────────────────────────────────

def _word_jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def compute_text_similarity(text_a: str, text_b: str, predictor=None) -> float:
    """
    Compute similarity between two text descriptions.
    Tries SAM3 VETextEncoder first; falls back to word-Jaccard.
    """
    if not text_a or not text_b:
        return 0.0
    if predictor is not None:
        try:
            import torch
            import torch.nn.functional as F
            enc = getattr(predictor, "model", predictor)
            text_encoder = getattr(enc, "text_encoder", None)
            if text_encoder is not None:
                with torch.inference_mode():
                    emb_a = text_encoder.encode(text_a).float().mean(dim=0, keepdim=True)
                    emb_b = text_encoder.encode(text_b).float().mean(dim=0, keepdim=True)
                    sim = F.cosine_similarity(emb_a, emb_b).item()
                return float(sim)
        except Exception as e:
            logger.debug(f"SAM3 text embedding failed ({e}), falling back to word overlap")
    return _word_jaccard(text_a, text_b)


# ─── Occlusion window ─────────────────────────────────────────────────────────

@dataclass
class OcclusionWindow:
    """Tracks a pair of objects through an occlusion/overlap event."""
    pair: tuple[int, int]                                     # (a, b), a < b
    onset_frame: int
    end_frame: int = -1                                       # -1 = still active
    velocity: dict = field(default_factory=dict)              # {obj_id: (dvx, dvy)} normalized/frame
    onset_masks: dict = field(default_factory=dict)           # {obj_id: uint8 (H,W)}
    ghost_masks_by_frame: dict = field(default_factory=dict)  # {frame_idx: {obj_id: mask}}
    post_frames: list = field(default_factory=list)           # [(frame_idx, {obj_id: mask})]
    identity_map: Optional[dict] = None                       # {observed_id: true_id}; None=unresolved, {}=no swap
    corrected: bool = False


# ─── Identity Tracker ─────────────────────────────────────────────────────────

class IdentityTracker:
    """
    Tracks all objects during video propagation.
    Computes confusion scores, detects swaps, and provides adaptive seeding strategy.
    """

    # Default thresholds (can be overridden per-instance via tracking_params)
    DEFAULT_TRACKING_PARAMS = {
        "min_iou_threshold": 0.15,
        "max_area_ratio": 5.0,
        "max_centroid_jump": 0.25,
        "consecutive_reject_limit": 5,
        # Entry confirmation window: after an object has been absent for
        # `entry_absence_threshold` consecutive frames, its next non-empty
        # prediction is treated as a *candidate* rather than a final mask.
        # Only after `entry_confirm_window` consecutive detections are all
        # candidate frames retroactively committed to disk.
        "entry_absence_threshold": 5,
        "entry_confirm_window": 5,
    }

    def __init__(
        self,
        object_ids: list[int],          # SAM obj_ids to track
        descriptions: dict[int, str],   # SAM obj_id → text description
        anomaly_threshold: float = 2.5,
        confusion_threshold: float = 0.4,
        confusion_reset_threshold: float = 0.7,
        text_similarity_threshold: float = 0.3,
        sam_predictor=None,
        tracking_params: Optional[dict] = None,
    ):
        self.object_ids = object_ids
        self.descriptions = descriptions
        self.anomaly_threshold = anomaly_threshold
        self.confusion_threshold = confusion_threshold
        self.confusion_reset_threshold = confusion_reset_threshold
        self.text_similarity_threshold = text_similarity_threshold

        # Initialize tracking params from defaults, then override with provided params
        self._tracking_params = {**self.DEFAULT_TRACKING_PARAMS}
        if tracking_params:
            self.update_tracking_params(tracking_params)

        # Per-object trajectories
        self.trajectories: dict[int, MaskTrajectory] = {
            oid: MaskTrajectory(oid) for oid in object_ids
        }

        # Pairwise text similarity matrix
        self.text_similarity: dict[tuple[int, int], float] = {}
        for i, a in enumerate(object_ids):
            for b in object_ids[i + 1:]:
                sim = compute_text_similarity(
                    descriptions.get(a, ""),
                    descriptions.get(b, ""),
                    predictor=sam_predictor,
                )
                self.text_similarity[(a, b)] = sim
                self.text_similarity[(b, a)] = sim
                logger.info(
                    f"Text similarity ({a!r} vs {b!r}): "
                    f"{descriptions.get(a)!r} ~ {descriptions.get(b)!r} = {sim:.3f}"
                )

        # Per-frame tracking data
        self.confusion_scores: dict[int, float] = {}
        self.anomaly_scores_by_obj: dict[int, dict[int, float]] = {
            oid: {} for oid in object_ids
        }
        self._prev_masks: dict[int, np.ndarray] = {}
        
        # Track temporal consistency rejections
        # {frame_idx: {obj_id: reason}}
        self.temporal_rejections: dict[int, dict[int, str]] = {}

        # Store seed annotation points for fallback seeding
        # {obj_id: [(frame_idx, points, labels), ...]}
        self.seed_annotations: dict[int, list[tuple[int, list, list]]] = {
            oid: [] for oid in object_ids
        }

        # ── Entry confirmation window state ───────────────────────────────────
        # Tracks how many consecutive frames each object has had NO raw SAM
        # output.  When this count is ≥ entry_absence_threshold and SAM
        # suddenly produces a mask, that mask becomes a *candidate* instead
        # of being saved immediately.
        self._no_mask_run: dict[int, int] = {}
        # Buffer of (frame_idx, mask) pairs collected during the entry window.
        # Absent once reset when the window is confirmed or aborted.
        self._entry_candidates: dict[int, list[tuple[int, np.ndarray]]] = {}

        # ── Occlusion / ghost propagation state ──────────────────────────────
        self.occlusion_windows: dict[tuple[int, int], OcclusionWindow] = {}
        # Onset triggers (any one is sufficient)
        self.ONSET_PROXIMITY = 0.12      # normalized centroid distance
        self.ONSET_AREA_RATIO = 0.85     # union_area / sum_of_individual_areas
        self.ONSET_TOUCH_MARGIN = 0.02   # fraction of min(H,W) for dilation touch-check
        # Velocity estimation
        self.VELOCITY_WINDOW = 8         # history frames for velocity estimate
        self.MAX_GHOST_FRAMES = 90       # clamp linear extrapolation after this many frames
        # Post-divergence identity resolution
        self.POST_DIVERGE_WAIT = 4       # stable SAM frames before resolving identity
        self.DIVERGE_WIDTH_FRAC = 0.10   # min centroid separation as fraction of avg mask width

    def set_seed_annotations(self, obj_id: int, annotations: list[tuple[int, list, list]]):
        """Register original hand-labeled point prompts for fallback seeding."""
        self.seed_annotations[obj_id] = annotations

    # ── Occlusion / Ghost Propagation ────────────────────────────────────────

    def _estimate_velocity(self, obj_id: int, onset_frame: int) -> tuple[float, float]:
        """
        Estimate per-frame displacement velocity from trajectory history before onset.
        Returns normalized (dvx, dvy) per frame in [0,1] coords.
        Falls back to (0,0) if history is too short or noisy.
        """
        traj = self.trajectories.get(obj_id)
        if traj is None:
            return (0.0, 0.0)
        relevant = [(f, s) for f, s in traj._history if f < onset_frame]
        relevant = relevant[-self.VELOCITY_WINDOW:]
        if len(relevant) < 2:
            return (0.0, 0.0)

        displacements = []
        weights = []
        for i in range(1, len(relevant)):
            f0, s0 = relevant[i - 1]
            f1, s1 = relevant[i]
            dt = max(1, f1 - f0)
            dvx = (s1.centroid[0] - s0.centroid[0]) / dt
            dvy = (s1.centroid[1] - s0.centroid[1]) / dt
            displacements.append((dvx, dvy))
            weights.append(float(i))  # linear ramp: more recent = higher weight

        mags = [float(np.sqrt(dx * dx + dy * dy)) for dx, dy in displacements]
        if len(mags) >= 2:
            mean_mag = float(np.mean(mags))
            std_mag = float(np.std(mags))
            # Remove outlier steps
            filtered = [
                (d, w) for d, m, w in zip(displacements, mags, weights)
                if m <= mean_mag + 2.0 * std_mag
            ]
            if filtered:
                displacements, weights = zip(*filtered)
                displacements = list(displacements)
                weights = list(weights)
                mags = [float(np.sqrt(dx * dx + dy * dy)) for dx, dy in displacements]
                # Noise guard: if std is high relative to mean, stay put
                if len(mags) >= 2 and float(np.mean(mags)) > 0:
                    if float(np.std(mags)) > 0.5 * float(np.mean(mags)):
                        return (0.0, 0.0)

        total_w = sum(weights)
        vx = sum(w * d[0] for w, d in zip(weights, displacements)) / total_w
        vy = sum(w * d[1] for w, d in zip(weights, displacements)) / total_w

        # Cap at 0.05 per frame (5% of frame dimension)
        mag = float(np.sqrt(vx * vx + vy * vy))
        if mag > 0.05:
            scale = 0.05 / mag
            vx, vy = vx * scale, vy * scale

        return (vx, vy)

    def _generate_ghost_mask(
        self,
        onset_mask: np.ndarray,
        velocity: tuple[float, float],
        frames_elapsed: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Translate onset_mask by velocity * min(frames_elapsed, MAX_GHOST_FRAMES) pixels.
        Returns uint8 (H, W) mask clipped to frame bounds.
        """
        H, W = frame_shape
        vx, vy = velocity
        clamped = min(frames_elapsed, self.MAX_GHOST_FRAMES)
        shift_x = int(round(vx * clamped * W))
        shift_y = int(round(vy * clamped * H))

        if shift_x == 0 and shift_y == 0:
            return onset_mask.copy()

        ghost = np.zeros((H, W), dtype=np.uint8)
        ys, xs = np.where(onset_mask.astype(bool))
        dst_xs = xs + shift_x
        dst_ys = ys + shift_y
        valid = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
        ghost[dst_ys[valid], dst_xs[valid]] = 1
        return ghost

    def _open_occlusion_window(
        self,
        frame_idx: int,
        pair: tuple[int, int],
        sam_masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> None:
        """Initialize an OcclusionWindow for the given pair at the given frame."""
        a, b = pair
        window = OcclusionWindow(pair=pair, onset_frame=frame_idx)
        for obj_id in (a, b):
            window.velocity[obj_id] = self._estimate_velocity(obj_id, frame_idx)
            traj = self.trajectories.get(obj_id)
            clean = traj.last_clean_mask if traj else None
            if clean is not None:
                window.onset_masks[obj_id] = clean.copy()
            elif sam_masks.get(obj_id) is not None:
                window.onset_masks[obj_id] = sam_masks[obj_id].copy()
        self.occlusion_windows[pair] = window
        logger.info(
            f"Occlusion onset frame {frame_idx}: pair {pair}, "
            f"velocities {window.velocity}"
        )

    def detect_overlap_onset(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> set[tuple[int, int]]:
        """
        Detect pairs of similar objects that are entering an occlusion event.
        Returns set of (a, b) pairs with a < b.
        """
        H, W = frame_shape
        margin_px = max(3, int(min(H, W) * self.ONSET_TOUCH_MARGIN))
        kernel = np.ones((margin_px, margin_px), np.uint8)
        triggered: set[tuple[int, int]] = set()

        for (a, b), sim in self.text_similarity.items():
            if a >= b:
                continue
            if sim < self.text_similarity_threshold:
                continue
            # Skip pairs already in an active window
            key = (a, b)
            if key in self.occlusion_windows and self.occlusion_windows[key].end_frame == -1:
                continue

            mask_a = sam_masks.get(a)
            mask_b = sam_masks.get(b)
            if mask_a is None or mask_b is None:
                continue
            stats_a = compute_mask_stats(mask_a)
            stats_b = compute_mask_stats(mask_b)
            if stats_a is None or stats_b is None:
                continue

            # Condition 1: centroids within ONSET_PROXIMITY
            dx = stats_a.centroid[0] - stats_b.centroid[0]
            dy = stats_a.centroid[1] - stats_b.centroid[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            proximity_triggered = dist < self.ONSET_PROXIMITY

            # Condition 2: masks overlapping (union < sum of individual areas)
            ma = mask_a.astype(bool)
            mb = mask_b.astype(bool)
            union_px = float((ma | mb).sum())
            sum_px = float(ma.sum()) + float(mb.sum())
            area_ratio = union_px / sum_px if sum_px > 0 else 1.0
            area_triggered = area_ratio < self.ONSET_AREA_RATIO

            # Condition 3: masks nearly touching (dilated masks intersect)
            dilated_a = cv2.dilate(mask_a, kernel)
            dilated_b = cv2.dilate(mask_b, kernel)
            touch_triggered = bool((dilated_a.astype(bool) & dilated_b.astype(bool)).any())

            if proximity_triggered or area_triggered or touch_triggered:
                triggered.add(key)
                logger.debug(
                    f"Frame {frame_idx}: pair {key} onset "
                    f"(prox={proximity_triggered}, area={area_triggered}, touch={touch_triggered})"
                )

        return triggered

    def _get_ghost_masks_for_frame(
        self,
        frame_idx: int,
        frame_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        """
        Return ghost masks for all objects currently in active occlusion windows.
        If an object participates in multiple pairs, ghosts are unioned.
        """
        ghost_accum: dict[int, np.ndarray] = {}
        for pair, window in self.occlusion_windows.items():
            if window.end_frame != -1:
                continue  # window already closed
            a, b = pair
            frames_elapsed = frame_idx - window.onset_frame
            for obj_id in (a, b):
                onset_mask = window.onset_masks.get(obj_id)
                if onset_mask is None:
                    continue
                vel = window.velocity.get(obj_id, (0.0, 0.0))
                ghost = self._generate_ghost_mask(onset_mask, vel, frames_elapsed, frame_shape)
                # Cache in window
                if frame_idx not in window.ghost_masks_by_frame:
                    window.ghost_masks_by_frame[frame_idx] = {}
                window.ghost_masks_by_frame[frame_idx][obj_id] = ghost
                # Accumulate (union across pairs)
                if obj_id in ghost_accum:
                    ghost_accum[obj_id] = np.logical_or(ghost_accum[obj_id], ghost).astype(np.uint8)
                else:
                    ghost_accum[obj_id] = ghost
        return ghost_accum

    def detect_divergence(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> set[tuple[int, int]]:
        """
        Detect pairs whose SAM masks have cleanly separated after an occlusion window.
        Returns pairs to close.
        """
        diverged: set[tuple[int, int]] = set()
        for pair, window in self.occlusion_windows.items():
            if window.end_frame != -1:
                continue
            a, b = pair
            mask_a = sam_masks.get(a)
            mask_b = sam_masks.get(b)
            if mask_a is None or mask_b is None:
                continue
            stats_a = compute_mask_stats(mask_a)
            stats_b = compute_mask_stats(mask_b)
            if stats_a is None or stats_b is None:
                continue
            avg_width = (stats_a.bbox[2] + stats_b.bbox[2]) / 2.0
            threshold = max(0.05, avg_width * self.DIVERGE_WIDTH_FRAC)
            dx = stats_a.centroid[0] - stats_b.centroid[0]
            dy = stats_a.centroid[1] - stats_b.centroid[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            iou = compute_iou(mask_a, mask_b)
            if dist > threshold and iou < 0.05:
                diverged.add(pair)
                logger.info(
                    f"Frame {frame_idx}: pair {pair} diverged "
                    f"(dist={dist:.3f} > {threshold:.3f}, iou={iou:.3f})"
                )
        return diverged

    def collect_post_diverge_frame(
        self,
        frame_idx: int,
        pair: tuple[int, int],
        sam_masks: dict[int, np.ndarray],
    ) -> None:
        """Accumulate stable SAM frames after divergence for identity resolution."""
        window = self.occlusion_windows.get(pair)
        if window is None or window.end_frame == -1 or window.identity_map is not None:
            return
        a, b = pair
        mask_a = sam_masks.get(a)
        mask_b = sam_masks.get(b)
        if mask_a is None or mask_b is None:
            return
        # Only accumulate if both masks are non-empty
        if not mask_a.any() or not mask_b.any():
            return
        window.post_frames.append((frame_idx, {a: mask_a.copy(), b: mask_b.copy()}))

    def resolve_post_overlap_identity(self, pair: tuple[int, int]) -> Optional[dict]:
        """
        After POST_DIVERGE_WAIT stable frames, determine true identity mapping.
        Returns {observed_id: true_id} or None if not enough data.
        An empty dict {} means no swap detected.
        """
        window = self.occlusion_windows.get(pair)
        if window is None or window.end_frame == -1:
            return None
        if len(window.post_frames) < self.POST_DIVERGE_WAIT:
            return None

        a, b = pair
        frames_to_use = window.post_frames[-self.POST_DIVERGE_WAIT:]

        masks_a = [f[a] for _, f in frames_to_use if a in f and f[a].any()]
        masks_b = [f[b] for _, f in frames_to_use if b in f and f[b].any()]
        if not masks_a or not masks_b:
            return None

        avg_mask_a = (np.mean(np.stack(masks_a, axis=0), axis=0) > 0.5).astype(np.uint8)
        avg_mask_b = (np.mean(np.stack(masks_b, axis=0), axis=0) > 0.5).astype(np.uint8)

        # Expected positions: last ghost at window.end_frame, else onset_masks
        ghost_at_end = window.ghost_masks_by_frame.get(window.end_frame, {})
        expected_a = ghost_at_end.get(a) if ghost_at_end.get(a) is not None else window.onset_masks.get(a)
        expected_b = ghost_at_end.get(b) if ghost_at_end.get(b) is not None else window.onset_masks.get(b)

        if expected_a is None or expected_b is None:
            return None

        # 2×2 IoU cost matrix: rows = expected {a, b}, cols = observed {a, b}
        cost = np.array([
            [1.0 - compute_iou(expected_a, avg_mask_a), 1.0 - compute_iou(expected_a, avg_mask_b)],
            [1.0 - compute_iou(expected_b, avg_mask_a), 1.0 - compute_iou(expected_b, avg_mask_b)],
        ])
        assignments = hungarian_match(cost)
        # assignments: [(expected_idx, observed_idx)]
        id_list = [a, b]
        identity_map: dict[int, int] = {}
        for expected_idx, observed_idx in assignments:
            true_id = id_list[expected_idx]
            observed_id = id_list[observed_idx]
            if true_id != observed_id:
                identity_map[observed_id] = true_id

        window.identity_map = identity_map
        if identity_map:
            logger.info(f"Identity swap resolved for pair {pair}: {identity_map}")
        else:
            logger.info(f"No identity swap detected for pair {pair} post-occlusion")
        return window.identity_map

    def _com_translated_fallback(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
        display: dict[int, np.ndarray],
        tracking: dict[int, np.ndarray],
        rejections: dict[int, str],
        frame_shape: tuple[int, int],
    ) -> None:
        """
        Phase 1.5: For similar nearby pairs where a mask has a drastic area change
        (< 0.5× or > 2× vs previous frame), translate both previous-frame masks
        toward the combined center-of-mass of the current SAM predictions.

        This catches the case where SAM collapses two nearby objects into a single
        merged blob (or drops one to zero) without an explicit occlusion window open.
        Modifies display/tracking/rejections in-place.
        """
        H, W = frame_shape
        AREA_LOW = 0.5
        AREA_HIGH = 2.0
        PROXIMITY = 0.30  # normalized centroid distance in prev frame

        for (a, b), sim in self.text_similarity.items():
            if a >= b:
                continue
            if sim < self.text_similarity_threshold:
                continue
            # Skip pairs already in an active ghost propagation window
            key = (min(a, b), max(a, b))
            if key in self.occlusion_windows and self.occlusion_windows[key].end_frame == -1:
                continue

            prev_a = self._prev_masks.get(a)
            prev_b = self._prev_masks.get(b)
            if prev_a is None or prev_b is None:
                continue
            stats_pa = compute_mask_stats(prev_a)
            stats_pb = compute_mask_stats(prev_b)
            if stats_pa is None or stats_pb is None:
                continue

            # Check proximity in previous frame
            dx = stats_pa.centroid[0] - stats_pb.centroid[0]
            dy = stats_pa.centroid[1] - stats_pb.centroid[1]
            if float(np.sqrt(dx * dx + dy * dy)) > PROXIMITY:
                continue

            # Check area ratios for both objects (using current display masks)
            def _area(m: Optional[np.ndarray]) -> float:
                return float(m.astype(bool).sum()) if m is not None else 0.0

            prev_area_a = _area(prev_a)
            prev_area_b = _area(prev_b)
            curr_area_a = _area(display.get(a))
            curr_area_b = _area(display.get(b))

            ratio_a = curr_area_a / prev_area_a if prev_area_a > 0 else float("inf")
            ratio_b = curr_area_b / prev_area_b if prev_area_b > 0 else float("inf")
            bad_a = ratio_a < AREA_LOW or ratio_a > AREA_HIGH or curr_area_a == 0
            bad_b = ratio_b < AREA_LOW or ratio_b > AREA_HIGH or curr_area_b == 0
            if not (bad_a or bad_b):
                continue

            # Combined CoM in previous frame (area-weighted)
            total_prev = prev_area_a + prev_area_b
            if total_prev == 0:
                continue
            prev_com_x = (stats_pa.centroid[0] * prev_area_a + stats_pb.centroid[0] * prev_area_b) / total_prev
            prev_com_y = (stats_pa.centroid[1] * prev_area_a + stats_pb.centroid[1] * prev_area_b) / total_prev

            # Combined CoM in current frame — from raw SAM output (non-empty masks)
            sam_a = sam_masks.get(a)
            sam_b = sam_masks.get(b)
            s_a = compute_mask_stats(sam_a) if sam_a is not None else None
            s_b = compute_mask_stats(sam_b) if sam_b is not None else None

            curr_wts: list[float] = []
            curr_cxs: list[float] = []
            curr_cys: list[float] = []
            if s_a is not None:
                w = _area(sam_a)
                if w > 0:
                    curr_wts.append(w); curr_cxs.append(s_a.centroid[0]); curr_cys.append(s_a.centroid[1])
            if s_b is not None:
                w = _area(sam_b)
                if w > 0:
                    curr_wts.append(w); curr_cxs.append(s_b.centroid[0]); curr_cys.append(s_b.centroid[1])
            if not curr_wts:
                continue  # Both SAM outputs empty — can't determine direction

            total_curr = sum(curr_wts)
            curr_com_x = sum(wt * cx for wt, cx in zip(curr_wts, curr_cxs)) / total_curr
            curr_com_y = sum(wt * cy for wt, cy in zip(curr_wts, curr_cys)) / total_curr

            # Translation vector
            shift_x = int(round((curr_com_x - prev_com_x) * W))
            shift_y = int(round((curr_com_y - prev_com_y) * H))

            logger.info(
                f"Frame {frame_idx}: pair ({a},{b}) CoM-translated "
                f"shift=({shift_x}px,{shift_y}px), bad_a={bad_a}, bad_b={bad_b}"
            )

            for obj_id, prev_m in ((a, prev_a), (b, prev_b)):
                translated = np.zeros((H, W), dtype=np.uint8)
                ys, xs = np.where(prev_m.astype(bool))
                dst_xs = xs + shift_x
                dst_ys = ys + shift_y
                valid = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
                translated[dst_ys[valid], dst_xs[valid]] = 1
                display[obj_id] = translated
                tracking[obj_id] = translated
                existing = rejections.get(obj_id, "")
                rejections[obj_id] = (existing + "|com_translated").lstrip("|")

    def get_retroactive_corrections(
        self, pair: tuple[int, int]
    ) -> list[tuple[int, dict[int, np.ndarray]]]:
        """
        Return corrected mask dicts for all frames in the occlusion window.
        Caller is responsible for writing these to disk.
        Returns [(frame_idx, {obj_id: corrected_mask}), ...].
        Marks the window as corrected.
        """
        window = self.occlusion_windows.get(pair)
        if window is None or window.identity_map is None or window.corrected:
            return []
        if window.end_frame == -1:
            return []  # window still active

        a, b = pair
        swap = bool(window.identity_map)  # non-empty means a swap occurred

        corrections: list[tuple[int, dict[int, np.ndarray]]] = []
        for frame_idx in range(window.onset_frame, window.end_frame + 1):
            ghost_frame = window.ghost_masks_by_frame.get(frame_idx)
            if not ghost_frame:
                continue
            corrected: dict[int, np.ndarray] = {}
            if swap:
                # Swap the ghost mask keys based on identity_map
                for obj_id, ghost in ghost_frame.items():
                    true_id = window.identity_map.get(obj_id, obj_id)
                    corrected[true_id] = ghost
            else:
                corrected = dict(ghost_frame)
            corrections.append((frame_idx, corrected))

        window.corrected = True
        return corrections

    # ── Tracking Params Management ───────────────────────────────────────────

    def update_tracking_params(self, params: dict) -> None:
        """Update tracking parameters dynamically."""
        for key, value in params.items():
            if key in self.DEFAULT_TRACKING_PARAMS:
                self._tracking_params[key] = value
                logger.info(f"Updated tracking param {key} = {value}")

    def get_tracking_params(self) -> dict:
        """Get current tracking parameters."""
        return self._tracking_params.copy()

    @property
    def min_iou_threshold(self) -> float:
        return self._tracking_params["min_iou_threshold"]

    @property
    def max_area_ratio(self) -> float:
        return self._tracking_params["max_area_ratio"]

    @property
    def max_centroid_jump(self) -> float:
        return self._tracking_params["max_centroid_jump"]

    @property
    def consecutive_reject_limit(self) -> int:
        return int(self._tracking_params["consecutive_reject_limit"])

    @property
    def entry_absence_threshold(self) -> int:
        return int(self._tracking_params.get("entry_absence_threshold", 5))

    @property
    def entry_confirm_window(self) -> int:
        return int(self._tracking_params.get("entry_confirm_window", 5))

    # ── Temporal Consistency Validation ──────────────────────────────────────

    def validate_temporal_consistency(
        self,
        obj_id: int,
        new_mask: np.ndarray,
        prev_mask: Optional[np.ndarray],
    ) -> tuple[bool, str, float]:
        """
        Validate that a new mask is temporally consistent with the previous frame.
        
        Returns:
            (is_valid, reason, confidence_score)
            - is_valid: True if mask passes consistency checks
            - reason: Description of why it failed (empty if valid)
            - confidence_score: 0-1 score, lower = more suspicious
        """
        if prev_mask is None:
            # No previous mask to compare - accept
            return True, "", 1.0
        
        new_stats = compute_mask_stats(new_mask, prev_mask)
        prev_stats = compute_mask_stats(prev_mask)
        
        if new_stats is None:
            return False, "empty_mask", 0.0
        if prev_stats is None:
            # Previous mask was empty, accept new one
            return True, "", 1.0
        
        # Check IoU
        iou = new_stats.iou_with_prev
        if iou < self.min_iou_threshold:
            return False, f"low_iou:{iou:.3f}", iou
        
        # Check area ratio
        if prev_stats.area > 0:
            area_ratio = new_stats.area / prev_stats.area
            if area_ratio > self.max_area_ratio or area_ratio < 1.0 / self.max_area_ratio:
                return False, f"area_jump:{area_ratio:.2f}x", max(0, 1.0 - abs(np.log(area_ratio)) / 2)
        
        # Check centroid distance
        dx = new_stats.centroid[0] - prev_stats.centroid[0]
        dy = new_stats.centroid[1] - prev_stats.centroid[1]
        centroid_dist = np.sqrt(dx * dx + dy * dy)
        if centroid_dist > self.max_centroid_jump:
            return False, f"centroid_jump:{centroid_dist:.3f}", max(0, 1.0 - centroid_dist / 0.5)
        
        # Compute confidence based on all factors
        iou_conf = min(1.0, iou / 0.5)  # Full confidence at IoU >= 0.5
        area_conf = 1.0 if prev_stats.area == 0 else max(0, 1.0 - abs(np.log(new_stats.area / prev_stats.area)) / 2)
        centroid_conf = max(0, 1.0 - centroid_dist / self.max_centroid_jump)
        confidence = (iou_conf * 0.5 + area_conf * 0.25 + centroid_conf * 0.25)
        
        return True, "", confidence

    def detect_overlap_confusion(self, sam_masks: dict[int, np.ndarray]) -> set[int]:
        """
        Detect objects involved in mask-level overlap confusion.

        Two similar objects whose SAM masks have high inter-object IoU, or where one
        mask has ballooned far beyond its trajectory-average area, are considered
        confused.  Returns the set of affected obj_ids.
        """
        confused: set[int] = set()
        for (a, b), sim in self.text_similarity.items():
            if a >= b or sim < self.text_similarity_threshold:
                continue
            mask_a = sam_masks.get(a)
            mask_b = sam_masks.get(b)
            if mask_a is None or mask_b is None:
                continue

            # Inter-object mask IoU
            iou = compute_iou(
                np.squeeze(mask_a).astype(bool),
                np.squeeze(mask_b).astype(bool),
            )
            if iou > 0.15:
                confused.add(a)
                confused.add(b)
                continue

            # Check for area explosion against trajectory history
            for oid, mask in ((a, mask_a), (b, mask_b)):
                traj = self.trajectories.get(oid)
                if traj is None or len(traj._history) < 5:
                    continue
                stats = compute_mask_stats(mask)
                if stats is None:
                    continue
                mean_area = float(np.mean([s.area for _, s in traj._history]))
                if mean_area > 0 and stats.area > mean_area * 3.0:
                    confused.add(a)
                    confused.add(b)
                    break

        return confused

    def validate_mask_against_trajectory(
        self,
        obj_id: int,
        mask: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Validate a mask against the object's rolling trajectory window.
        Catches area explosions and centroid drift relative to recent history.
        Returns (is_valid, reason).
        """
        traj = self.trajectories.get(obj_id)
        if traj is None or len(traj._history) < 5:
            return True, ""

        stats = compute_mask_stats(mask)
        if stats is None:
            return False, "empty_mask"

        areas = [s.area for _, s in traj._history]
        cxs   = [s.centroid[0] for _, s in traj._history]
        cys   = [s.centroid[1] for _, s in traj._history]

        mean_area = float(np.mean(areas))
        mean_cx   = float(np.mean(cxs))
        mean_cy   = float(np.mean(cys))

        # Area explosion: more than 3× the rolling average
        if mean_area > 0 and stats.area > mean_area * 3.0:
            return False, f"overlap_area_explosion:{stats.area/mean_area:.1f}x"

        # Centroid drift: more than 40% of frame diagonal from trajectory mean
        dx = stats.centroid[0] - mean_cx
        dy = stats.centroid[1] - mean_cy
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist > 0.40:
            return False, f"overlap_centroid_drift:{dist:.3f}"

        return True, ""

    def validate_and_filter_masks(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
        force_accept: bool = False,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, str],
        dict[int, list[tuple[int, np.ndarray]]],
    ]:
        """
        Validate SAM output masks for temporal consistency and filter/fallback as needed.

        Args:
            frame_idx: Current frame index
            sam_masks: SAM's output masks {obj_id: mask}
            force_accept: If True, accept all masks regardless of validation

        Returns:
            (display_masks, tracking_masks, rejection_reasons, retroactive_masks)
            - display_masks:  masks to save to disk (SAM output when valid, temporal
                              fallback otherwise)
            - tracking_masks: masks to carry forward as tracking state — uses last
                              clean mask for confused objects so overlap frames don't
                              corrupt future predictions
            - rejection_reasons: obj_id → reason string for any rejection
            - retroactive_masks: {obj_id: [(frame_idx, mask), ...]} for objects whose
                              entry window just confirmed — caller must retroactively
                              save these past-frame masks to disk
        """
        display: dict[int, np.ndarray] = {}
        tracking: dict[int, np.ndarray] = {}
        rejections: dict[int, str] = {}
        # Retroactive masks for objects whose entry window just confirmed.
        # Each entry is a list of (frame_idx, mask) that the caller must
        # save to disk because they were withheld during the window.
        retroactive: dict[int, list[tuple[int, np.ndarray]]] = {}

        # Track consecutive rejections per object
        if not hasattr(self, '_consecutive_rejects'):
            self._consecutive_rejects: dict[int, int] = {}

        # ── Phase 0: update raw-absence counters and abort stale entry windows ─
        # Count BEFORE temporal validation so we know the pre-frame absence run.
        prev_no_mask_run = dict(self._no_mask_run)
        for oid in self.object_ids:
            raw = sam_masks.get(oid)
            if raw is None or not raw.astype(bool).any():
                self._no_mask_run[oid] = self._no_mask_run.get(oid, 0) + 1
                # An absent frame during an entry window aborts it
                if oid in self._entry_candidates:
                    logger.info(
                        f"Frame {frame_idx}: obj {oid} entry window aborted "
                        f"(absent after {len(self._entry_candidates[oid])} candidates)"
                    )
                    self._entry_candidates.pop(oid)
            # else: don't reset yet — reset happens when mask is confirmed/committed

        # ── Phase 1: temporal consistency validation ──────────────────────────
        for oid in self.object_ids:
            new_mask = sam_masks.get(oid)
            prev_mask = self._prev_masks.get(oid)

            # Objects inside an entry confirmation window must not be compared
            # against a stale prev_mask (which may be from a wrong historical
            # detection at a completely different position).  Treat them as if
            # no previous mask exists so the check always passes.
            if oid in self._entry_candidates:
                prev_mask = None

            if new_mask is None:
                if prev_mask is not None:
                    display[oid] = prev_mask.copy()
                    tracking[oid] = prev_mask.copy()
                    rejections[oid] = "no_sam_output"
                continue

            if force_accept:
                display[oid] = new_mask
                tracking[oid] = new_mask
                self._consecutive_rejects[oid] = 0
                continue

            # Empty mask: always fall back — never force-accept a zero-area mask.
            # These don't count toward consecutive_rejects so they don't
            # eventually trigger the force-accept escape hatch.
            if not new_mask.astype(bool).any():
                if prev_mask is not None:
                    display[oid] = prev_mask.copy()
                    tracking[oid] = prev_mask.copy()
                    rejections[oid] = "empty_mask"
                    logger.debug(f"Frame {frame_idx}, obj {oid}: empty mask, using prev")
                continue

            is_valid, reason, _ = self.validate_temporal_consistency(oid, new_mask, prev_mask)

            if is_valid:
                display[oid] = new_mask
                tracking[oid] = new_mask
                self._consecutive_rejects[oid] = 0
            else:
                consec = self._consecutive_rejects.get(oid, 0) + 1
                self._consecutive_rejects[oid] = consec

                if consec >= self.consecutive_reject_limit:
                    logger.warning(
                        f"Frame {frame_idx}, obj {oid}: accepting after {consec} consecutive rejects "
                        f"(reason: {reason})"
                    )
                    display[oid] = new_mask
                    tracking[oid] = new_mask
                    self._consecutive_rejects[oid] = 0
                elif prev_mask is not None:
                    logger.info(
                        f"Frame {frame_idx}, obj {oid}: rejected ({reason}), using prev mask"
                    )
                    display[oid] = prev_mask.copy()
                    tracking[oid] = prev_mask.copy()
                    rejections[oid] = reason
                else:
                    logger.warning(
                        f"Frame {frame_idx}, obj {oid}: rejected ({reason}) but no fallback, accepting"
                    )
                    display[oid] = new_mask
                    tracking[oid] = new_mask

        # ── Phase 1.3: Entry confirmation window ──────────────────────────────
        # After a long absence (≥ entry_absence_threshold consecutive frames with
        # no raw SAM output), the next detection is treated as a *candidate*
        # rather than committed immediately.  Only after entry_confirm_window
        # consecutive candidate frames does the entry become final.
        #
        # This prevents false positives (SAM briefly predicts a hallucination at
        # the old annotation location) and also avoids the stale-prev_mask
        # rejection loop where a real re-entry gets rejected because _prev_masks
        # still holds a mask from an unrelated earlier detection.
        if not force_accept:
            for oid in list(display.keys()):
                mask = display[oid]
                if mask is None or not mask.astype(bool).any():
                    continue  # empty / fallback — no entry logic needed

                # Was this object absent long enough to trigger an entry window?
                run_before = prev_no_mask_run.get(oid, 0)
                in_window = oid in self._entry_candidates
                if run_before >= self.entry_absence_threshold or in_window:
                    # Add this frame to the candidate buffer
                    if not in_window:
                        self._entry_candidates[oid] = []
                        logger.info(
                            f"Frame {frame_idx}: obj {oid} entry window started "
                            f"(absent {run_before} frames, "
                            f"need {self.entry_confirm_window} to confirm)"
                        )
                    self._entry_candidates[oid].append((frame_idx, mask.copy()))
                    num_candidates = len(self._entry_candidates[oid])

                    if num_candidates >= self.entry_confirm_window:
                        # Confirmed! Commit current frame and queue past frames.
                        past = self._entry_candidates.pop(oid)
                        # Past frames (all but the last, which is the current frame)
                        # must be written to disk by the caller.
                        if len(past) > 1:
                            retroactive[oid] = past[:-1]
                        self._no_mask_run[oid] = 0
                        self._consecutive_rejects[oid] = 0
                        logger.info(
                            f"Frame {frame_idx}: obj {oid} entry CONFIRMED "
                            f"({num_candidates} consecutive frames); "
                            f"retroactive={len(past) - 1} frames"
                        )
                        # Keep current frame in display/tracking (already set above)
                    else:
                        # Still accumulating — withhold this frame from disk.
                        # Do NOT update _prev_masks (update_frame won't see it).
                        del display[oid]
                        if oid in tracking:
                            del tracking[oid]
                        rejections[oid] = (
                            f"entry_candidate:{num_candidates}/{self.entry_confirm_window}"
                        )
                        logger.debug(
                            f"Frame {frame_idx}: obj {oid} entry candidate "
                            f"{num_candidates}/{self.entry_confirm_window}"
                        )
                else:
                    # Not coming from a long absence — normal tracking, reset run.
                    self._no_mask_run[oid] = 0
                    self._consecutive_rejects[oid] = 0

        # ── Phase 1.5: CoM-translation fallback for drastic area changes ─────
        # Runs before Phase 2 so that the corrected masks feed into overlap
        # detection. Skipped for pairs already in an active ghost window.
        if not force_accept and frame_shape is not None and len(self.text_similarity) > 0:
            self._com_translated_fallback(
                frame_idx, sam_masks, display, tracking, rejections, frame_shape
            )

        # ── Phase 2: overlap-confusion validation ─────────────────────────────
        # Only run when there are at least 2 similar objects tracked.
        if len(self.text_similarity) > 0 and not force_accept:
            confused_ids = self.detect_overlap_confusion(sam_masks)
            for oid in confused_ids:
                new_mask = sam_masks.get(oid)
                if new_mask is None:
                    continue
                traj_ok, traj_reason = self.validate_mask_against_trajectory(oid, new_mask)
                if not traj_ok:
                    # Keep the SAM prediction in display (so user sees what happened)
                    # but fall back to last clean mask for tracking carry-forward.
                    clean = self.trajectories[oid].last_clean_mask
                    if clean is not None:
                        tracking[oid] = clean.copy()
                        rejection_key = f"overlap+{traj_reason}"
                        rejections[oid] = rejection_key
                        logger.info(
                            f"Frame {frame_idx}, obj {oid}: overlap confusion ({traj_reason}), "
                            f"tracking falls back to last clean mask (frame "
                            f"{self.trajectories[oid].last_clean_frame})"
                        )
                    # display mask is already set from phase 1 — leave it as-is

        # ── Phase 3: Overlap onset detection ──────────────────────────────────
        if not force_accept and frame_shape is not None and len(self.text_similarity) > 0:
            for pair in self.detect_overlap_onset(frame_idx, sam_masks, frame_shape):
                key = (min(pair), max(pair))
                if key not in self.occlusion_windows or self.occlusion_windows[key].end_frame != -1:
                    self._open_occlusion_window(frame_idx, key, sam_masks, frame_shape)

        # ── Phase 4: Ghost mask substitution during active occlusion windows ──
        if not force_accept and frame_shape is not None:
            ghost_masks = self._get_ghost_masks_for_frame(frame_idx, frame_shape)
            for oid, ghost in ghost_masks.items():
                display[oid] = ghost
                tracking[oid] = ghost
                existing = rejections.get(oid, "")
                rejections[oid] = (existing + "|ghost").lstrip("|")

        # ── Phase 5: Divergence check + post-diverge frame collection ─────────
        if not force_accept and frame_shape is not None:
            # Use raw sam_masks for divergence detection (not ghost-substituted display)
            for pair in self.detect_divergence(frame_idx, sam_masks, frame_shape):
                key = (min(pair), max(pair))
                if key in self.occlusion_windows:
                    self.occlusion_windows[key].end_frame = frame_idx
            # Collect stable SAM frames for closed-but-unresolved windows
            for key, window in self.occlusion_windows.items():
                if window.end_frame != -1 and window.identity_map is None:
                    self.collect_post_diverge_frame(frame_idx, key, sam_masks)
                    # Auto-resolve once we have enough frames
                    if len(window.post_frames) >= self.POST_DIVERGE_WAIT:
                        self.resolve_post_overlap_identity(key)

        # Record rejections for uncertainty reporting
        if rejections:
            self.temporal_rejections[frame_idx] = {oid: r for oid, r in rejections.items()}

        return display, tracking, rejections, retroactive

    def update_frame(
        self,
        frame_idx: int,
        masks_dict: dict[int, np.ndarray],   # SAM obj_id → binary mask (H,W)
        bboxes_dict: dict[int, list],         # SAM obj_id → [x,y,w,h,score]
    ) -> dict[int, float]:
        """
        Process one frame's worth of outputs.
        Returns per-object anomaly scores.
        """
        anomaly_scores = {}
        for oid in self.object_ids:
            mask = masks_dict.get(oid)
            prev = self._prev_masks.get(oid)
            if mask is None:
                self.anomaly_scores_by_obj[oid][frame_idx] = 0.0
                continue

            stats = compute_mask_stats(mask, prev)
            if stats is None:
                self.anomaly_scores_by_obj[oid][frame_idx] = 0.0
                continue

            score = self.trajectories[oid].record_frame(
                frame_idx, stats, mask, self.anomaly_threshold
            )
            self.anomaly_scores_by_obj[oid][frame_idx] = score
            anomaly_scores[oid] = score

        # Compute pair-wise confusion score
        conf_score = self._compute_confusion_score(frame_idx, masks_dict)
        self.confusion_scores[frame_idx] = conf_score

        # Update previous masks
        self._prev_masks = {
            oid: mask.copy()
            for oid, mask in masks_dict.items()
            if mask is not None
        }

        return anomaly_scores

    def _compute_confusion_score(
        self,
        frame_idx: int,
        masks_dict: dict[int, np.ndarray],
    ) -> float:
        """
        Compute the frame-level confusion score: max over all similar-object pairs
        of (proximity × text_similarity).
        """
        max_score = 0.0
        for (a, b), sim in self.text_similarity.items():
            if a >= b:
                continue  # only process each pair once
            if sim < self.text_similarity_threshold:
                continue
            mask_a = masks_dict.get(a)
            mask_b = masks_dict.get(b)
            if mask_a is None or mask_b is None:
                continue
            stats_a = compute_mask_stats(mask_a)
            stats_b = compute_mask_stats(mask_b)
            if stats_a is None or stats_b is None:
                continue

            # Centroid distance (normalized)
            dx = stats_a.centroid[0] - stats_b.centroid[0]
            dy = stats_a.centroid[1] - stats_b.centroid[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            proximity = max(1.0 - dist / 0.4, 0.0)

            # IoU between the two masks
            iou = compute_iou(
                np.squeeze(mask_a).astype(bool),
                np.squeeze(mask_b).astype(bool),
            )
            proximity = max(proximity, min(iou, 1.0))

            pair_score = proximity * sim
            if pair_score > max_score:
                max_score = pair_score

        return min(max_score, 1.0)

    def get_confusion_windows(self) -> list[dict]:
        """
        Merge consecutive frames with confusion_score ≥ threshold into windows.
        Returns list of {start, end, obj_ids, avg_score}.
        """
        if not self.confusion_scores:
            return []
        frames = sorted(self.confusion_scores)
        windows = []
        in_window = False
        w_start = 0
        w_scores = []

        for f in frames:
            score = self.confusion_scores[f]
            if score >= self.confusion_threshold:
                if not in_window:
                    in_window = True
                    w_start = f
                    w_scores = [score]
                else:
                    w_scores.append(score)
            else:
                if in_window:
                    in_window = False
                    obj_ids = self._confused_objects_in_window(w_start, f - 1)
                    if obj_ids:
                        windows.append({
                            "start": w_start,
                            "end": f - 1,
                            "obj_ids": [str(oid) for oid in obj_ids],
                            "avg_score": float(np.mean(w_scores)),
                        })
                    w_scores = []

        if in_window and w_scores:
            last_f = frames[-1]
            obj_ids = self._confused_objects_in_window(w_start, last_f)
            if obj_ids:
                windows.append({
                    "start": w_start,
                    "end": last_f,
                    "obj_ids": [str(oid) for oid in obj_ids],
                    "avg_score": float(np.mean(w_scores)),
                })

        return windows

    def _confused_objects_in_window(self, w_start: int, w_end: int) -> list[int]:
        """Return SAM obj_ids that were confused within the given frame range."""
        obj_ids = set()
        for (a, b), sim in self.text_similarity.items():
            if a >= b or sim < self.text_similarity_threshold:
                continue
            # Check if any frame in the window has high confusion for this pair
            for f in range(w_start, w_end + 1):
                score = self.confusion_scores.get(f, 0.0)
                if score >= self.confusion_threshold:
                    obj_ids.add(a)
                    obj_ids.add(b)
                    break
        return sorted(obj_ids)

    def get_adaptive_seeds(self, confusion_score: float) -> dict[int, tuple]:
        """
        Return seeding strategy for each object based on the given confusion score.

        Returns dict of:
          obj_id → ("mask", mask_array)          when confusion < 0.3
          obj_id → ("both", mask_array, seed_pts) when 0.3 ≤ confusion < 0.7
          obj_id → ("annotations", seed_pts)      when confusion ≥ 0.7
        where seed_pts = [(frame_idx, points, labels), ...]
        """
        strategy = {}
        for oid in self.object_ids:
            traj = self.trajectories[oid]
            clean_mask = traj.last_clean_mask
            seed_pts = self.seed_annotations.get(oid, [])

            if confusion_score < 0.3:
                if clean_mask is not None:
                    strategy[oid] = ("mask", clean_mask)
                elif seed_pts:
                    strategy[oid] = ("annotations", seed_pts)
            elif confusion_score < 0.7:
                if clean_mask is not None and seed_pts:
                    strategy[oid] = ("both", clean_mask, seed_pts)
                elif clean_mask is not None:
                    strategy[oid] = ("mask", clean_mask)
                elif seed_pts:
                    strategy[oid] = ("annotations", seed_pts)
            else:
                if seed_pts:
                    strategy[oid] = ("annotations", seed_pts)
                elif clean_mask is not None:
                    # Nothing else available; use clean mask as last resort
                    strategy[oid] = ("mask", clean_mask)

        return strategy

    def assign_instances(
        self,
        prev_masks: dict[int, np.ndarray],
        cur_masks: dict[int, np.ndarray],
        group_ids: list[int],
    ) -> dict[int, int]:
        """
        Run Hungarian matching to maintain consistent instance identities within
        a multi-instance group across frames.

        Returns dict mapping: current SAM output slot → assigned persistent slot.
        (Identity is preserved by maximizing IoU with previous-frame masks.)
        """
        if not group_ids or len(group_ids) == 1:
            return {gid: gid for gid in group_ids}

        prev_list = [prev_masks.get(gid) for gid in group_ids]
        cur_list = [cur_masks.get(gid) for gid in group_ids]

        n = len(group_ids)
        # Cost = 1 - IoU (minimize cost = maximize overlap)
        cost_matrix = np.ones((n, n), dtype=float)
        for i, prev in enumerate(prev_list):
            for j, cur in enumerate(cur_list):
                if prev is not None and cur is not None:
                    cost_matrix[i, j] = 1.0 - compute_iou(
                        np.squeeze(prev).astype(bool),
                        np.squeeze(cur).astype(bool),
                    )

        assignments = hungarian_match(cost_matrix)
        # assignments: list of (prev_slot_idx, cur_slot_idx)
        result = {}
        for prev_idx, cur_idx in assignments:
            prev_id = group_ids[prev_idx]
            cur_id = group_ids[cur_idx]
            result[cur_id] = prev_id  # current slot → persistent identity
        return result

    def detect_identity_swap(
        self,
        window_start: int,
        window_end: int,
        post_window_masks: dict[int, np.ndarray],  # obj_id → mask from post-window frames
        seed_masks: dict[int, np.ndarray],          # obj_id → mask from seed frame
    ) -> tuple[Optional[dict[int, int]], int]:
        """
        Detect whether identities of objects swapped during a confusion window.

        Compares spatial position of post-window masks against seed-frame mask
        positions. If object A's post-window mask is closer to B's seed region
        than A's, a swap is suspected.

        Returns:
          - swap_dict: {obj_id: swapped_with_obj_id} (empty if no swap detected)
          - swap_onset: estimated frame where swap first occurred (window_start if unsure)
        """
        # Only consider object pairs with significant text similarity
        candidate_pairs = [
            (a, b) for (a, b), sim in self.text_similarity.items()
            if a < b and sim >= self.text_similarity_threshold
        ]
        if not candidate_pairs:
            return {}, window_start

        swaps = {}
        for (a, b) in candidate_pairs:
            mask_a_post = post_window_masks.get(a)
            mask_b_post = post_window_masks.get(b)
            mask_a_seed = seed_masks.get(a)
            mask_b_seed = seed_masks.get(b)

            if any(m is None for m in [mask_a_post, mask_b_post, mask_a_seed, mask_b_seed]):
                continue

            # IoU of post-window A with seed regions
            iou_a_vs_a = compute_iou(
                np.squeeze(mask_a_post).astype(bool),
                np.squeeze(mask_a_seed).astype(bool),
            )
            iou_a_vs_b = compute_iou(
                np.squeeze(mask_a_post).astype(bool),
                np.squeeze(mask_b_seed).astype(bool),
            )
            iou_b_vs_b = compute_iou(
                np.squeeze(mask_b_post).astype(bool),
                np.squeeze(mask_b_seed).astype(bool),
            )
            iou_b_vs_a = compute_iou(
                np.squeeze(mask_b_post).astype(bool),
                np.squeeze(mask_a_seed).astype(bool),
            )

            # Swap if cross-IoU exceeds same-identity IoU by a margin
            SWAP_MARGIN = 0.15
            a_swapped = (iou_a_vs_b - iou_a_vs_a) > SWAP_MARGIN
            b_swapped = (iou_b_vs_a - iou_b_vs_b) > SWAP_MARGIN

            if a_swapped and b_swapped:
                swaps[a] = b
                swaps[b] = a
                logger.info(
                    f"Identity swap detected: obj {a} ↔ {b} "
                    f"(iou_a_vs_a={iou_a_vs_a:.2f}, iou_a_vs_b={iou_a_vs_b:.2f}, "
                    f"iou_b_vs_b={iou_b_vs_b:.2f}, iou_b_vs_a={iou_b_vs_a:.2f})"
                )

        if not swaps:
            return {}, window_start

        # Estimate swap onset via binary search on confusion scores within window
        swap_onset = self._estimate_swap_onset(window_start, window_end, swaps)
        return swaps, swap_onset

    def _estimate_swap_onset(
        self,
        window_start: int,
        window_end: int,
        swaps: dict[int, int],
    ) -> int:
        """
        Find the first frame in the window where trajectory anomaly scores were
        high for the swapped objects — this is the likely swap onset.
        """
        swapped_ids = list(swaps.keys())
        frames = sorted(
            f for f in range(window_start, window_end + 1)
            if f in self.confusion_scores
        )
        if not frames:
            return window_start

        # Find first frame where all swapped objects had high anomaly scores
        for f in frames:
            scores = [
                self.anomaly_scores_by_obj.get(oid, {}).get(f, 0.0)
                for oid in swapped_ids
            ]
            if all(s > self.anomaly_threshold * 0.6 for s in scores):
                return f

        return window_start

    def build_uncertainty_json(
        self,
        confusion_windows: Optional[list[dict]] = None,
    ) -> dict:
        """Build the uncertainty.json payload for storage."""
        if confusion_windows is None:
            confusion_windows = self.get_confusion_windows()

        per_frame = {}
        for f, score in self.confusion_scores.items():
            entry: dict = {"confusion_score": round(score, 4)}

            # Per-object anomaly scores for this frame
            per_object = {}
            for oid in self.object_ids:
                anomaly = self.anomaly_scores_by_obj.get(oid, {}).get(f, 0.0)
                per_object[str(oid)] = {"anomaly_score": round(float(anomaly), 4)}
            if per_object:
                entry["per_object"] = per_object

            if score >= self.confusion_threshold:
                confused = self._confused_objects_in_window(f, f)
                if confused:
                    entry["confused_objects"] = [str(oid) for oid in confused]
            # Add temporal rejection info if any
            if f in self.temporal_rejections:
                entry["temporal_rejections"] = {
                    str(oid): reason for oid, reason in self.temporal_rejections[f].items()
                }
            per_frame[str(f)] = entry

        sim_matrix = {
            f"{a}_{b}": round(sim, 4)
            for (a, b), sim in self.text_similarity.items()
            if a < b
        }
        
        # Count total rejections
        total_rejections = sum(len(r) for r in self.temporal_rejections.values())

        return {
            "per_frame": per_frame,
            "confusion_windows": confusion_windows,
            "similarity_matrix": sim_matrix,
            "temporal_rejection_count": total_rejections,
        }

    def build_correction_records(
        self,
        confusion_windows: list[dict],
        post_window_masks_by_window: dict,  # window_idx → {obj_id: mask}
        seed_masks: dict[int, np.ndarray],
    ) -> list[dict]:
        """
        For each confusion window, run swap detection and build correction records.
        Returns list of correction dicts ready for storage.
        """
        records = []
        for w_idx, window in enumerate(confusion_windows):
            post_masks = post_window_masks_by_window.get(w_idx, {})
            if not post_masks:
                continue

            # Convert str keys to int for detect_identity_swap
            post_int = {int(k): v for k, v in post_masks.items()}

            swaps, swap_onset = self.detect_identity_swap(
                window["start"],
                window["end"],
                post_int,
                seed_masks,
            )
            if not swaps:
                continue

            # Only emit one record per unique pair
            seen = set()
            for a, b in swaps.items():
                pair = (min(a, b), max(a, b))
                if pair in seen:
                    continue
                seen.add(pair)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                records.append({
                    "id": ts,
                    "status": "pending",
                    "window_start": window["start"],
                    "window_end": window["end"],
                    "swap_onset": swap_onset,
                    "obj_id_a": str(pair[0]),
                    "obj_id_b": str(pair[1]),
                    "reason": "spatial_crossover",
                    "avg_confusion_score": round(window["avg_score"], 4),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

        return records
