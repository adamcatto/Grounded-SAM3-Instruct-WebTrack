"""Lazy CoTracker3 point-pose inference using Meta's Hugging Face checkpoint."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_REPO = "facebook/cotracker3"
MODEL_FILE = "scaled_offline.pth"
DEFAULT_MODEL_PATH = Path("/opt/models/cotracker3/scaled_offline.pth")


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
        }

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

    def track(
        self,
        source_path: str,
        start_frame: int,
        num_next_frames: int,
        points_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track N points from start through the next requested frames, chunking at 60."""
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
                all_tracks.extend(chunk_tracks[1:])
                all_visibility.extend(chunk_visibility[1:])
                query_points = chunk_tracks[-1]
                cursor += steps
                remaining -= steps
                del video, query_tensor, tracks, visibility
        return np.asarray(all_tracks), np.asarray(all_visibility)


cotracker3 = CoTracker3Predictor()
