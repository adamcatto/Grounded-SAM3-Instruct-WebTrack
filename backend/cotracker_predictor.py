"""Lazy CoTracker3 point-pose inference using Meta's Hugging Face checkpoint."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MODEL_REPO = "facebook/cotracker3"
MODEL_FILE = "scaled_offline.pth"
DEFAULT_MODEL_PATH = Path("/opt/models/cotracker3/scaled_offline.pth")
DEFAULT_SHAPE_REGULARIZATION_STRENGTH = 0.9


class CoTracker3Predictor:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.RLock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint: Path | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def checkpoint_path(self) -> Path:
        configured = os.environ.get("COTRACKER3_CHECKPOINT", "").strip()
        candidate = Path(configured).expanduser() if configured else DEFAULT_MODEL_PATH
        if candidate.is_file():
            return candidate.resolve()
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)).resolve()

    def ensure_loaded(self):
        with self._lock:
            if self._model is not None:
                return self._model
            checkpoint = self.checkpoint_path()
            model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                pretrained=False,
                trust_repo=True,
            )
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            model.model.load_state_dict(state)
            model = model.to(self._device).eval()
            self._model = model
            self._checkpoint = checkpoint
            return model

    def status(self) -> dict:
        path = None
        try:
            path = self.checkpoint_path()
        except Exception:
            pass
        return {
            "model": "cotracker3_offline",
            "repo": MODEL_REPO,
            "checkpoint": str(path) if path else None,
            "checkpoint_ready": bool(path and path.is_file()),
            "loaded": self.loaded,
            "device": self.device,
            "shape_regularization": {
                "type": "appearance_guided_per_object_similarity",
                "strength": float(os.environ.get(
                    "COTRACKER3_POSE_SHAPE_STRENGTH",
                    str(DEFAULT_SHAPE_REGULARIZATION_STRENGTH),
                )),
                "minimum_reference_scale": 0.75,
                "maximum_scale_change_per_frame": 0.025,
                "orientation_source": "local_foreground_pca_with_temporal_sign",
            },
        }

    def extract_feature_map(self, source_path: str, frame_idx: int) -> np.ndarray:
        """Extract the normalized CoTracker convolutional map for one video frame."""
        model = self.ensure_loaded()
        capture = cv2.VideoCapture(source_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_idx}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self._device)
        tensor = F.interpolate(
            tensor, tuple(model.interp_shape), mode="bilinear", align_corners=True,
        )
        tensor = 2 * (tensor / 255.0) - 1.0
        with self._lock, torch.inference_mode():
            features = model.model.fnet(tensor)
            features = F.normalize(features.float(), dim=1)
        return features[0].detach().cpu().numpy().astype(np.float16)

    @staticmethod
    def correlate_memories(
        target_features: np.ndarray,
        memories: list[tuple[np.ndarray, np.ndarray, float | np.ndarray]],
    ) -> np.ndarray:
        """Locate points in a target feature map using weighted memory descriptors.

        Each memory tuple contains (feature_map[C,H,W], normalized_xy[N,2], weight).
        """
        target = np.asarray(target_features, dtype=np.float32)
        _, height, width = target.shape
        count = len(memories[0][1])
        scores = np.zeros((count, height, width), dtype=np.float32)
        weight_sum = np.zeros(count, dtype=np.float32)
        for feature_map, points, weight in memories:
            fmap = np.asarray(feature_map, dtype=np.float32)
            points = np.asarray(points, dtype=np.float32)
            valid = ~np.isnan(points).any(axis=1)
            weights = np.broadcast_to(np.asarray(weight, dtype=np.float32), (count,))
            active = valid & (weights > 0)
            if not active.any():
                continue
            indices = np.flatnonzero(active)
            xs = np.clip(np.rint(points[indices, 0] * (fmap.shape[2] - 1)), 0, fmap.shape[2] - 1).astype(int)
            ys = np.clip(np.rint(points[indices, 1] * (fmap.shape[1] - 1)), 0, fmap.shape[1] - 1).astype(int)
            descriptors = fmap[:, ys, xs].T
            descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
            scores[indices] += weights[indices, None, None] * np.einsum("nc,chw->nhw", descriptors, target)
            weight_sum[indices] += weights[indices]
        if np.any(weight_sum <= 0):
            raise ValueError("Every pose point requires at least one visible, positive-weight memory")
        flat = scores.reshape(count, -1).argmax(axis=1)
        y, x = np.divmod(flat, width)
        return np.column_stack([
            x.astype(np.float32) / max(1, width - 1),
            y.astype(np.float32) / max(1, height - 1),
        ])

    @staticmethod
    def _read_clip(source_path: str, start_frame: int, end_frame: int) -> np.ndarray:
        capture = cv2.VideoCapture(source_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: list[np.ndarray] = []
        try:
            for _ in range(start_frame, end_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(frames) != end_frame - start_frame + 1:
            raise RuntimeError(
                f"Decoded {len(frames)} of {end_frame - start_frame + 1} requested frames"
            )
        return np.stack(frames)

    @staticmethod
    def regularize_pose_tracks(
        tracks: np.ndarray,
        visibility: np.ndarray,
        reference_points: np.ndarray,
        point_groups: list[list[int]],
        state: dict | None = None,
        strength: float = 0.9,
        max_scale_step: float = 0.025,
        max_rotation_step: float = np.deg2rad(20.0),
        skip_first: bool = False,
        frame_images: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Regularize each object's landmarks toward a rotation-invariant reference shape.

        Translation remains unconstrained. Rotation and scale follow the raw tracks, but
        their frame-to-frame changes are bounded; landmark residuals are blended with the
        closest similarity-transformed reference configuration.
        """
        output = np.asarray(tracks, dtype=np.float32).copy()
        visible = np.asarray(visibility, dtype=bool)
        reference = np.asarray(reference_points, dtype=np.float32)
        state = state or {"scales": {}, "angles": {}, "angular_velocities": {}}
        scales = state.setdefault("scales", {})
        angles = state.setdefault("angles", {})
        angular_velocities = state.setdefault("angular_velocities", {})
        strength = float(np.clip(strength, 0.0, 1.0))
        start_index = 1 if skip_first else 0

        for frame_idx in range(start_index, len(output)):
            for group_idx, group in enumerate(point_groups):
                indices = np.asarray(group, dtype=int)
                if len(indices) < 2:
                    continue
                ref = reference[indices]
                raw = output[frame_idx, indices]
                finite = np.isfinite(raw).all(axis=1)
                if finite.sum() >= 2:
                    ref_mean = ref[finite].mean(axis=0)
                    raw_mean = raw[finite].mean(axis=0)
                    ref_centered = ref[finite] - ref_mean
                    raw_centered = raw[finite] - raw_mean
                    covariance = ref_centered.T @ raw_centered
                    u, singular, vt = np.linalg.svd(covariance)
                    rotation = u @ vt
                    if np.linalg.det(rotation) < 0:
                        u[:, -1] *= -1
                        rotation = u @ vt
                    denominator = float(np.square(ref_centered).sum())
                    fitted_scale = float(singular.sum() / max(denominator, 1e-8))
                    fitted_angle = float(np.arctan2(rotation[0, 1], rotation[0, 0]))
                else:
                    ref_mean = ref.mean(axis=0)
                    raw_mean = raw[np.isfinite(raw).all(axis=1)].mean(axis=0) if np.isfinite(raw).all(axis=1).any() else ref_mean
                    fitted_scale = float(scales.get(group_idx, 1.0))
                    fitted_angle = float(angles.get(group_idx, 0.0))

                previous_scale = float(scales.get(group_idx, 1.0))
                previous_angle = float(angles.get(group_idx, 0.0))
                previous_velocity = float(angular_velocities.get(group_idx, 0.0))
                image_orientation_available = False
                if frame_images is not None and frame_idx < len(frame_images):
                    reference_axis = ref[-1] - ref[0]
                    reference_span = float(np.linalg.norm(reference_axis))
                    image_axis = CoTracker3Predictor._foreground_major_axis(
                        frame_images[frame_idx],
                        raw_mean,
                        reference_span,
                    )
                    if image_axis is not None and reference_span > 1e-6:
                        reference_angle = float(np.arctan2(reference_axis[1], reference_axis[0]))
                        image_angle = float(np.arctan2(image_axis[1], image_axis[0]))
                        image_transform = image_angle - reference_angle
                        predicted_angle = previous_angle + previous_velocity
                        candidates = (image_transform, image_transform + np.pi)
                        fitted_angle = min(
                            candidates,
                            key=lambda candidate: abs(
                                (candidate - predicted_angle + np.pi) % (2 * np.pi) - np.pi
                            ),
                        )
                        image_orientation_available = True
                # A collapsed landmark constellation has an ill-conditioned
                # orientation and can flip as points cross. Continue the last
                # trustworthy angular motion instead of accepting that flip.
                orientation_reliable = (
                    image_orientation_available
                    or fitted_scale >= max(0.55, previous_scale * 0.65)
                )
                scale = float(np.clip(
                    fitted_scale,
                    max(0.75, previous_scale * (1.0 - max_scale_step)),
                    min(2.0, previous_scale * (1.0 + max_scale_step)),
                ))
                if orientation_reliable:
                    angle_delta = (fitted_angle - previous_angle + np.pi) % (2 * np.pi) - np.pi
                    angle_delta = float(np.clip(angle_delta, -max_rotation_step, max_rotation_step))
                    angular_velocity = 0.75 * previous_velocity + 0.25 * angle_delta
                else:
                    angle_delta = previous_velocity
                    angular_velocity = previous_velocity
                angle = previous_angle + float(np.clip(angle_delta, -max_rotation_step, max_rotation_step))
                rotation = np.asarray([
                    [np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)],
                ], dtype=np.float32)
                structured = raw_mean + scale * ((ref - ref_mean) @ rotation)
                blend = np.where(visible[frame_idx, indices, None], strength, 1.0)
                output[frame_idx, indices] = (1.0 - blend) * raw + blend * structured
                scales[group_idx] = scale
                angles[group_idx] = angle
                angular_velocities[group_idx] = angular_velocity
        return output, state

    @staticmethod
    def _foreground_major_axis(
        frame_rgb: np.ndarray,
        center_xy: np.ndarray,
        reference_span: float,
    ) -> np.ndarray | None:
        """Estimate an oriented animal axis from local foreground appearance.

        The connected component nearest the tracked centroid supplies an
        unoriented PCA axis. Its longer, thinner extension is treated as the
        tail side, resolving the 180-degree ambiguity for elongated animals.
        """
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        height, width = frame.shape[:2]
        radius = int(np.clip(reference_span * 1.35, 48, min(height, width) * 0.35))
        cx = int(np.clip(round(float(center_xy[0])), 0, width - 1))
        cy = int(np.clip(round(float(center_xy[1])), 0, height - 1))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        crop = frame[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 16:
            return None
        border = np.concatenate([
            crop[0], crop[-1], crop[:, 0], crop[:, -1],
        ], axis=0).astype(np.float32)
        background = np.median(border, axis=0)
        difference = np.linalg.norm(crop.astype(np.float32) - background, axis=2)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        scaled = np.clip(difference, 0, 255).astype(np.uint8)
        threshold, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if threshold < 10:
            mask = (scaled >= 10).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        local_center = np.asarray([cx - x0, cy - y0], dtype=np.float32)
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 40:
                continue
            distance = float(np.linalg.norm(centroids[label] - local_center))
            candidates.append((distance, -area, label))
        if not candidates:
            return None
        distance, _negative_area, label = min(candidates)
        if distance > radius * 0.8:
            return None
        yy, xx = np.nonzero(labels == label)
        points = np.column_stack([xx, yy]).astype(np.float32)
        if len(points) < 40:
            return None
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(points) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] < max(16.0, eigenvalues[0] * 1.8):
            return None
        axis = eigenvectors[:, -1].astype(np.float32)
        centered_on_median = points - np.median(points, axis=0)
        projections = centered_on_median @ axis
        perpendicular = centered_on_median @ np.asarray([-axis[1], axis[0]])
        q10, q35, q65, q90 = np.quantile(projections, [0.10, 0.35, 0.65, 0.90])
        negative_side = perpendicular[(projections >= q10) & (projections <= q35)]
        positive_side = perpendicular[(projections >= q65) & (projections <= q90)]
        negative_width = (
            float(np.quantile(negative_side, 0.9) - np.quantile(negative_side, 0.1))
            if len(negative_side) >= 10 else 0.0
        )
        positive_width = (
            float(np.quantile(positive_side, 0.9) - np.quantile(positive_side, 0.1))
            if len(positive_side) >= 10 else 0.0
        )
        # The rump / tail-base side is typically wider than the tapered snout
        # side, so orient the major axis from snout toward tail base.
        if positive_width < negative_width:
            axis *= -1
        return axis / max(float(np.linalg.norm(axis)), 1e-8)

    def track(
        self,
        source_path: str,
        start_frame: int,
        num_next_frames: int,
        points_xy: np.ndarray,
        point_groups: list[list[int]] | None = None,
        reference_points_xy: np.ndarray | None = None,
        on_progress: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track N points, publishing cumulative results after every inferred chunk."""
        if num_next_frames < 1:
            raise ValueError("num_next_frames must be at least 1")
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if not len(points):
            raise ValueError("At least one labeled pose part is required")
        model = self.ensure_loaded()
        all_tracks = [points.copy()]
        all_visibility = [np.ones(len(points), dtype=bool)]
        cursor = int(start_frame)
        remaining = int(num_next_frames)
        query_points = points.copy()
        reference_points = (
            np.asarray(reference_points_xy, dtype=np.float32).reshape(-1, 2)
            if reference_points_xy is not None else points.copy()
        )
        groups = point_groups or [list(range(len(points)))]
        regularization_state: dict = {"scales": {}, "angles": {}, "angular_velocities": {}}
        regularization_strength = float(os.environ.get(
            "COTRACKER3_POSE_SHAPE_STRENGTH",
            str(DEFAULT_SHAPE_REGULARIZATION_STRENGTH),
        ))
        with self._lock, torch.inference_mode():
            while remaining > 0:
                steps = min(remaining, 59)
                frames = self._read_clip(source_path, cursor, cursor + steps)
                video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self._device)
                queries = np.column_stack([np.zeros(len(query_points)), query_points])
                query_tensor = torch.from_numpy(queries.astype(np.float32))[None].to(self._device)
                tracks, visibility = model(video, queries=query_tensor)
                chunk_tracks = tracks[0].detach().cpu().numpy()
                chunk_visibility = visibility[0].detach().cpu().numpy().astype(bool)
                next_query_points = chunk_tracks[-1].copy()
                chunk_tracks, regularization_state = self.regularize_pose_tracks(
                    chunk_tracks,
                    chunk_visibility,
                    reference_points,
                    groups,
                    state=regularization_state,
                    strength=regularization_strength,
                    skip_first=True,
                    frame_images=frames,
                )
                all_tracks.extend(chunk_tracks[1:])
                all_visibility.extend(chunk_visibility[1:])
                if on_progress is not None:
                    on_progress(np.asarray(all_tracks), np.asarray(all_visibility))
                # Preserve CoTracker's own point queries for localization. The
                # shape prior regularizes published anatomy but does not invent
                # image evidence for the next clip.
                query_points = next_query_points
                cursor += steps
                remaining -= steps
                del video, query_tensor, tracks, visibility
        return np.asarray(all_tracks), np.asarray(all_visibility)


cotracker3 = CoTracker3Predictor()
