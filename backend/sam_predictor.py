"""
SAM video predictor wrapper with SAM3 → SAM2 fallback.
Manages sessions keyed by (project_id, video_id).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SAM3_CHECKPOINT = Path("/opt/software/SAM3WebTrack/pretrained_models/sam3.pt")
SAM2_CHECKPOINT = Path("/opt/software/SAM3WebTrack/pretrained_models/sam2.1_hiera_large.pt")
SAM2_CONFIG = "sam2.1_hiera_l"  # sam2 config name for the large model

# Lazy-loaded predictor singleton
_predictor = None
_model_name: str = "none"


def _get_predictor():
    """
    Try to load SAM3 first.  If the checkpoint is missing or the import fails,
    fall back to SAM2.  Returns the predictor singleton.
    """
    global _predictor, _model_name
    if _predictor is not None:
        return _predictor

    # ── Attempt 1: SAM3 ──────────────────────────────────────────────────────
    if SAM3_CHECKPOINT.exists():
        try:
            logger.info(f"Loading SAM3 model from {SAM3_CHECKPOINT} ...")
            from sam3.model_builder import build_sam3_video_predictor
            _predictor = build_sam3_video_predictor(checkpoint_path=str(SAM3_CHECKPOINT))
            _model_name = "sam3"
            logger.info("SAM3 model loaded successfully.")
            return _predictor
        except Exception as e:
            logger.warning(f"SAM3 load failed ({e}), will try SAM2 fallback.")

    # ── Attempt 2: SAM2 ──────────────────────────────────────────────────────
    if SAM2_CHECKPOINT.exists():
        try:
            logger.info(f"Loading SAM2 model from {SAM2_CHECKPOINT} ...")
            from sam2.build_sam import build_sam2_video_predictor
            _predictor = build_sam2_video_predictor(
                config_file=SAM2_CONFIG,
                ckpt_path=str(SAM2_CHECKPOINT),
            )
            _model_name = "sam2"
            logger.info("SAM2 model loaded successfully (fallback).")
            return _predictor
        except Exception as e:
            logger.error(f"SAM2 load also failed: {e}")

    raise FileNotFoundError(
        "No SAM model checkpoint found.  Looked for:\n"
        f"  SAM3: {SAM3_CHECKPOINT}\n"
        f"  SAM2: {SAM2_CHECKPOINT}\n"
        "Run:  python scripts/download_model.py"
    )


class SAMPredictor:
    """
    Manages SAM video sessions for multiple projects/videos.
    Transparently handles SAM3 or SAM2 backends.
    """

    def __init__(self):
        self._sessions: dict[tuple, str] = {}
        self._sam2_states: dict[tuple, object] = {}  # SAM2 inference states
        # Maps (pid, vid) → sorted list of real frame indices loaded into the SAM session.
        # SAM's internal frame index is the position in this sorted list, NOT the real video
        # frame index.  Preview frames are named by real index (e.g. 001902.jpg), so we
        # must translate before every handle_request call.
        self._frame_maps: dict[tuple, list[int]] = {}

    # ── Status helpers ────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if a checkpoint file is available on disk."""
        return SAM3_CHECKPOINT.exists() or SAM2_CHECKPOINT.exists()

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded into memory."""
        return _predictor is not None

    def model_name(self) -> str:
        """Return which model is currently loaded (sam3 / sam2 / none)."""
        return _model_name

    def ensure_loaded(self):
        """Eagerly load the model into memory (blocks until done)."""
        _get_predictor()

    # ── Frame index translation ──────────────────────────────────────────────

    def _build_frame_map(self, pid: str, vid: str, frames_dir: str):
        """
        Scan *frames_dir* for *.jpg files and record their sorted real frame
        indices.  SAM assigns its own sequential 0-based index by sorting the
        filenames alphabetically, so position-in-sorted-list == SAM frame idx.
        """
        jpg_files = sorted(Path(frames_dir).glob("*.jpg"))
        real_indices = []
        for f in jpg_files:
            try:
                real_indices.append(int(f.stem))
            except ValueError:
                pass
        self._frame_maps[(pid, vid)] = real_indices
        logger.debug(
            f"Frame map built for {pid}/{vid}: {len(real_indices)} frames "
            f"(first={real_indices[:3] if real_indices else []}, "
            f"last={real_indices[-3:] if real_indices else []})"
        )

    def to_real_idx(self, pid: str, vid: str, sam_idx: int) -> int:
        """
        Translate SAM's internal 0-based sequential frame index back to the
        real video frame index (i.e. the number in the jpg filename).
        Falls back to sam_idx itself if no map is recorded.
        """
        frame_list = self._frame_maps.get((pid, vid))
        if frame_list and sam_idx < len(frame_list):
            return frame_list[sam_idx]
        return sam_idx

    def _to_sam_idx(self, pid: str, vid: str, real_idx: int) -> int:
        """
        Translate a real video frame index to SAM's internal 0-based sequential
        index (position in the sorted list of loaded jpg files).
        Raises ValueError if the frame is not in the current session.
        """
        frame_list = self._frame_maps.get((pid, vid))
        if not frame_list:
            # No mapping recorded (e.g. SAM2 path or mapping not yet built).
            return real_idx
        try:
            return frame_list.index(real_idx)
        except ValueError:
            raise ValueError(
                f"Frame {real_idx} is not loaded in the current SAM session "
                f"({len(frame_list)} frames; "
                f"first few: {frame_list[:5]}, last few: {frame_list[-5:]}). "
                "Re-initialize the session after extracting the frame."
            )

    # ── Session management ──────────────────────────────────────────────────

    def init_session(self, pid: str, vid: str, frames_dir: str) -> str:
        """Initialize (or re-initialize) a SAM session for a video."""
        self.close_session(pid, vid)
        predictor = _get_predictor()

        if _model_name == "sam2":
            import torch
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(video_path=str(frames_dir))
            session_id = f"sam2_{pid}_{vid}"
            self._sessions[(pid, vid)] = session_id
            self._sam2_states[(pid, vid)] = inference_state
        else:
            # SAM3 uses its own handle_request protocol
            resp = predictor.handle_request({
                "type": "start_session",
                "resource_path": str(frames_dir),
            })
            session_id = resp["session_id"]
            self._sessions[(pid, vid)] = session_id

        # Record which real frame indices are loaded so we can translate to SAM's
        # internal sequential indices later.
        self._build_frame_map(pid, vid, frames_dir)

        logger.info(f"Session started ({_model_name}) for {pid}/{vid}: {session_id}")
        return session_id

    def get_session_id(self, pid: str, vid: str) -> Optional[str]:
        return self._sessions.get((pid, vid))

    def close_session(self, pid: str, vid: str):
        session_id = self._sessions.pop((pid, vid), None)
        self._sam2_states.pop((pid, vid), None)
        self._frame_maps.pop((pid, vid), None)
        if session_id is not None:
            if _model_name != "sam2":
                try:
                    _get_predictor().handle_request({
                        "type": "close_session",
                        "session_id": session_id,
                    })
                except Exception as e:
                    logger.warning(f"Error closing session {session_id}: {e}")
            logger.info(f"Session closed: {session_id}")

    def reset_session(self, pid: str, vid: str):
        session_id = self.get_session_id(pid, vid)
        if session_id is None:
            raise ValueError(f"No active session for {pid}/{vid}")

        if _model_name == "sam2":
            import torch
            state = self._sam2_states.get((pid, vid))
            if state is not None:
                _get_predictor().reset_state(state)
        else:
            _get_predictor().handle_request({
                "type": "reset_session",
                "session_id": session_id,
            })

    # ── Point prompts ────────────────────────────────────────────────────────

    def add_points(
        self,
        pid: str,
        vid: str,
        frame_idx: int,
        obj_id: int,
        points: list,   # [[x_norm, y_norm], ...]
        labels: list,   # [1, 0, ...]
    ) -> dict:
        """
        Add/update point prompts for an object on a frame.
        Returns a unified dict: {frame_idx: {out_obj_ids, out_binary_masks, out_boxes_xywh, out_probs}}
        """
        session_id = self.get_session_id(pid, vid)
        if session_id is None:
            raise ValueError(f"No active session for {pid}/{vid}. Call init_session first.")

        predictor = _get_predictor()

        if _model_name == "sam2":
            import torch, numpy as np
            state = self._sam2_states[(pid, vid)]
            sam_frame_idx = self._to_sam_idx(pid, vid, frame_idx)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                video_h = state["video_height"]
                video_w = state["video_width"]
                abs_points = [[x * video_w, y * video_h] for x, y in points]
                frame_out_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=sam_frame_idx,
                    obj_id=obj_id,
                    points=abs_points,
                    labels=labels,
                    normalize_coords=True,
                )
            # Convert to unified format
            masks_np = [(m > 0.0).cpu().numpy() for m in masks]
            return {
                frame_idx: {
                    "out_obj_ids": list(obj_ids) if not isinstance(obj_ids, list) else obj_ids,
                    "out_binary_masks": masks_np,
                    "out_boxes_xywh": [],
                    "out_probs": [],
                }
            }
        else:
            import torch
            # Translate real video frame index → SAM's internal sequential index
            sam_frame_idx = self._to_sam_idx(pid, vid, frame_idx)
            # SAM3's _build_tracker_output asserts that cached_frame_outputs[frame_idx]
            # exists (normally populated by propagation).  For interactive first-click
            # annotation before propagation, pre-seed the entry with an empty dict so
            # the tracker can create a brand-new object and fill in its mask.
            state = predictor._ALL_INFERENCE_STATES[session_id]["state"]
            if sam_frame_idx not in state["cached_frame_outputs"]:
                state["cached_frame_outputs"][sam_frame_idx] = {}
            # SAM3 uses the "add_prompt" request type with points in relative coords
            points_tensor = torch.tensor(points, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                resp = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": sam_frame_idx,
                    "obj_id": obj_id,
                    "points": points_tensor,
                    "point_labels": labels_tensor,
                })
            # resp = {"frame_index": sam_frame_idx, "outputs": {out_obj_ids, out_binary_masks, ...}}
            outputs = resp.get("outputs", {})
            return {frame_idx: outputs}  # key by real frame_idx for callers

    def add_mask_prompt(
        self,
        pid: str,
        vid: str,
        frame_idx: int,
        obj_id: int,
        mask: object,  # numpy array (H, W)
    ) -> dict:
        """Add a mask as a prompt (used for chunk handoff)."""
        session_id = self.get_session_id(pid, vid)
        if session_id is None:
            raise ValueError(f"No active session for {pid}/{vid}.")
        predictor = _get_predictor()

        if _model_name == "sam2":
            import torch
            state = self._sam2_states[(pid, vid)]
            sam_frame_idx = self._to_sam_idx(pid, vid, frame_idx)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                frame_out_idx, obj_ids, masks = predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=sam_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
            masks_np = [(m > 0.0).cpu().numpy() for m in masks]
            return {
                frame_idx: {
                    "out_obj_ids": list(obj_ids) if not isinstance(obj_ids, list) else obj_ids,
                    "out_binary_masks": masks_np,
                }
            }
        else:
            import torch
            import numpy as np
            # SAM3 doesn't expose add_new_mask via handle_request.
            # Use center-of-mass of the mask as a point prompt for handoff.
            mask_np = np.squeeze(mask)
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0:
                logger.warning(f"Empty mask for obj {obj_id} on frame {frame_idx}, skipping")
                return {}
            # Compute center of mass in relative coords (0-1)
            h, w = mask_np.shape
            cx = float(xs.mean()) / w
            cy = float(ys.mean()) / h
            # Translate real video frame index → SAM's internal sequential index
            sam_frame_idx = self._to_sam_idx(pid, vid, frame_idx)
            # Pre-seed cached_frame_outputs so SAM3's assertion passes (same as add_points)
            state = predictor._ALL_INFERENCE_STATES[session_id]["state"]
            if sam_frame_idx not in state["cached_frame_outputs"]:
                state["cached_frame_outputs"][sam_frame_idx] = {}
            points_tensor = torch.tensor([[cx, cy]], dtype=torch.float32)
            labels_tensor = torch.tensor([1], dtype=torch.int32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                resp = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": sam_frame_idx,
                    "obj_id": obj_id,
                    "points": points_tensor,
                    "point_labels": labels_tensor,
                })
            outputs = resp.get("outputs", {})
            return {frame_idx: outputs}  # key by real frame_idx for callers

    # ── Object management ────────────────────────────────────────────────────

    def remove_object(self, pid: str, vid: str, obj_id: int):
        session_id = self.get_session_id(pid, vid)
        if session_id is None:
            return

        if _model_name == "sam2":
            # SAM2 doesn't have a direct remove_object; we note it for our records.
            # The object can be effectively ignored during propagation.
            logger.info(f"SAM2: object {obj_id} removal noted (session will be reset on next propagation if needed)")
        else:
            _get_predictor().handle_request({
                "type": "remove_object",
                "session_id": session_id,
                "obj_id": obj_id,
            })

    def clear_object_prompts(self, pid: str, vid: str, obj_id: int):
        """Clear all points for an object (remove it, then it can be re-added)."""
        self.remove_object(pid, vid, obj_id)

    # ── Propagation ──────────────────────────────────────────────────────────

    def propagate_stream(
        self,
        pid: str,
        vid: str,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        propagation_direction: str = "both",
    ):
        """
        Generator that yields per-frame propagation results.
        Each item: {frame_index: int, outputs: {out_obj_ids, out_binary_masks, out_boxes_xywh}}

        If start_frame_idx / max_frame_num_to_track are given, propagation is
        limited to a sub-range of the video (used for batched processing).
        """
        session_id = self.get_session_id(pid, vid)
        if session_id is None:
            raise ValueError(f"No active session for {pid}/{vid}.")

        predictor = _get_predictor()

        if _model_name == "sam2":
            import torch
            state = self._sam2_states[(pid, vid)]
            # Translate real frame indices → SAM's internal sequential indices
            sam_start = self._to_sam_idx(pid, vid, start_frame_idx) if start_frame_idx is not None else None
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                    state,
                    start_frame_idx=sam_start,
                    max_frame_num_to_track=max_frame_num_to_track,
                ):
                    masks_np = [(m > 0.0).cpu().numpy() for m in masks]
                    yield {
                        "frame_index": frame_idx,
                        "outputs": {
                            frame_idx: {
                                "out_obj_ids": list(obj_ids) if not isinstance(obj_ids, list) else obj_ids,
                                "out_binary_masks": masks_np,
                                "out_boxes_xywh": [],
                                "out_probs": [],
                            }
                        },
                    }
        else:
            import torch
            # Translate real frame indices → SAM's internal sequential indices.
            # With temp-dir-per-batch, SAM's internal index 0 is NOT necessarily
            # real frame 0 — it's the first jpg in the temp dir.
            sam_start = self._to_sam_idx(pid, vid, start_frame_idx) if start_frame_idx is not None else None
            state = predictor._ALL_INFERENCE_STATES[session_id]["state"]
            # Pre-seed the start frame's cache so SAM3's propagation assertion passes.
            # add_points already seeds the annotated frame; this covers the case where
            # start_frame_idx differs (e.g. frame 0) or propagation is called directly.
            if sam_start is not None:
                if sam_start not in state.get("cached_frame_outputs", {}):
                    state.setdefault("cached_frame_outputs", {})[sam_start] = {}
            # SAM3's parse_action_history_for_propagation checks action_history to decide
            # the propagation type.  add_prompt populates it with "add" entries, which
            # causes it to pick propagation_partial — a mode that requires ALL frames to
            # already be in cached_frame_outputs from a prior full propagation.  Since each
            # batch opens a fresh session we always want propagation_full, so we clear the
            # history here before handing off to handle_stream_request.
            state["action_history"].clear()
            # add_tracker_new_points (called by add_points) registers objects in the tracker
            # but never populates rank0_metadata["obj_first_frame_idx"].  That dict is only
            # filled for detector-found objects inside _process_hotstart.  When a prompted
            # object is unmatched for hotstart_unmatch_thresh (=8) frames, _process_hotstart
            # tries obj_first_frame_idx[obj_id] and crashes with KeyError.
            # Fix: pre-seed all tracked obj_ids with a very negative frame index so that
            # is_within_hotstart = (very_neg > frame_idx - hotstart_delay) is always False,
            # preventing both the KeyError and the hotstart-removal heuristic from firing.
            tracker_meta = state.get("tracker_metadata", {})
            rank0 = tracker_meta.get("rank0_metadata") if tracker_meta else None
            if rank0 is not None:
                first_frame_map = rank0.get("obj_first_frame_idx")
                if first_frame_map is not None:
                    for oid in tracker_meta.get("obj_ids_all_gpu", []):
                        oid_int = int(oid)
                        if oid_int not in first_frame_map:
                            first_frame_map[oid_int] = -999999
                            logger.debug(f"Pre-seeded obj_first_frame_idx[{oid_int}] = -999999")

            req = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
            }
            if sam_start is not None:
                req["start_frame_index"] = sam_start
            if max_frame_num_to_track is not None:
                req["max_frame_num_to_track"] = max_frame_num_to_track
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                yield from predictor.handle_stream_request(req)
