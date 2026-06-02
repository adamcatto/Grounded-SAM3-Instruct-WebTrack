"""
SAM Web Tracker — FastAPI backend server.
Supports SAM3 (primary) with SAM2 fallback.
"""

import asyncio
import functools
import os
import socket
import json
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import base64
import cv2
import io

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from anchor_helpers import (
    ANCHOR_MANUAL_PREFIX_COUNT,
    STREAM_BATCH_SIZE,
    anchor_labeling_has_started,
    compute_anchor_frames,
    normalize_anchor_batch_size,
    video_anchor_batch_size,
)
from project_manager import ProjectManager, iter_objects_topdown
from sam_predictor import SAMPredictor, _get_predictor
from mask_store import VideoMaskStorage
import hierarchy
from video_processor import (
    compute_ds_dims,
    composite_masks_as_png,
    decode_masks_png_base64_to_binary,
    encode_mask_as_png,
    ensure_faststart,
    export_video_with_masks,
    extract_frames,
    extract_frame_indices,
    extract_frame_range,
    generate_thumbnail,
    get_video_info,
    bbox_norm_xywh_score_from_mask,
)


def _video_ds_params(video: dict) -> tuple[Optional[int], Optional[float]]:
    """Return (max_dim, scale_factor) stored in video metadata for lazy downsampling."""
    return video.get("downsample_max_dim"), video.get("downsample_scale_factor")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM3 Web Tracker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
)

pm = ProjectManager()
sam = SAMPredictor()

# Track model loading state
_model_loading = False
_model_load_error: str | None = None


# ─── Propagation state registry ───────────────────────────────────────────────

class PropagationState:
    """Tracks an active (or recently finished) propagation task for one video."""

    def __init__(self):
        self.is_running: bool = False
        self.is_paused: bool = False
        self.paused_at_frame: int = -1
        self.task: asyncio.Task | None = None
        self.subscribers: list[asyncio.Queue] = []
        self.total_frames: int = 0
        self.start_frame: int = 0       # actual_start (earliest annotated frame)
        self.user_start_frame: int = 0  # user's requested start frame (for progress display)

    async def publish(self, event: dict) -> None:
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass


_prop_registry: dict[str, PropagationState] = {}


# ─── Anchor remainder inference (sequential propagate between anchor pairs) ─

class AnchorRemainderState:
    """One active anchor-remainder prediction stream per (project, video)."""

    def __init__(self):
        self.is_running: bool = False
        self.task: asyncio.Task | None = None
        self.subscribers: list[asyncio.Queue] = []
        # Interactive review: background task awaits review_continue_event until POST …/continue.
        self.waiting_review: bool = False
        self.review_continue_event: asyncio.Event | None = None

    async def publish(self, event: dict) -> None:
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=400)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass


_anchor_remainder_registry: dict[str, AnchorRemainderState] = {}


def _get_anchor_remainder_state(pid: str, vid: str) -> AnchorRemainderState:
    key = f"{pid}/{vid}"
    if key not in _anchor_remainder_registry:
        _anchor_remainder_registry[key] = AnchorRemainderState()
    return _anchor_remainder_registry[key]


def _bbox_norm_xywh_score(mask: np.ndarray) -> tuple[list[float], float]:
    m = np.squeeze(mask).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    h, w = m.shape[:2]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    nw = max(1, w)
    nh = max(1, h)
    return [
        float(x0) / nw,
        float(y0) / nh,
        float(x1 - x0 + 1) / nw,
        float(y1 - y0 + 1) / nh,
    ], 1.0


def _extract_frame_masks_bboxes(pid: str, vid: str, item: dict):
    sam_idx = item.get("frame_index", 0)
    real_frame = sam.to_real_idx(pid, vid, sam_idx)
    outputs = item.get("outputs", {})
    frame_masks: dict[int, np.ndarray] = {}
    frame_bboxes: dict[int, list] = {}

    if "out_obj_ids" in outputs:
        obj_ids = outputs.get("out_obj_ids", [])
        masks_list = outputs.get("out_binary_masks", [])
        boxes_list = outputs.get("out_boxes_xywh", [])
        probs_list = outputs.get("out_probs", [])
    else:
        obj_ids, masks_list, boxes_list, probs_list = [], [], [], []
        for out_val in outputs.values():
            if isinstance(out_val, dict):
                obj_ids = out_val.get("out_obj_ids", [])
                masks_list = out_val.get("out_binary_masks", [])
                boxes_list = out_val.get("out_boxes_xywh", [])
                probs_list = out_val.get("out_probs", [])
                break

    for k, oid in enumerate(obj_ids):
        if k < len(masks_list):
            msk = masks_list[k]
            if hasattr(msk, "numpy"):
                msk = msk.numpy()
            frame_masks[int(oid)] = np.squeeze(msk).astype(np.uint8)
        if k < len(boxes_list):
            b = boxes_list[k]
            if hasattr(b, "numpy"):
                b = b.numpy()
            if hasattr(b, "tolist"):
                b = b.tolist()
            score = float(probs_list[k]) if k < len(probs_list) else 1.0
            frame_bboxes[int(oid)] = (b + [score]) if b else []

    return real_frame, frame_masks, frame_bboxes


def _ensure_jpg_for_indices(
    pid: str,
    vid: str,
    tmp_dir: str,
    frame_indices: list[int],
    source_path: str,
    ds_max_dim: Optional[int],
    ds_scale_factor: Optional[float],
) -> None:
    """
    SAM loads every JPG under frames_dir. Anchor remainder builds a sparse folder:
    only the listed frame indices (e.g. all manually labeled anchors plus one predict target).
    """
    out_path = Path(tmp_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ann_dir = pm.annotated_frames_dir(pid, vid)
    preview_dir = pm.frames_dir(pid, vid)
    uniq = sorted(set(frame_indices))
    missing: list[int] = []
    for idx in uniq:
        dst = out_path / f"{idx:06d}.jpg"
        if dst.is_file():
            continue
        copied = False
        for folder in (ann_dir, preview_dir):
            src = folder / f"{idx:06d}.jpg"
            if src.is_file():
                shutil.copy2(src, dst)
                copied = True
                break
        if not copied:
            missing.append(idx)
    if missing:
        extract_frame_indices(
            source_path,
            tmp_dir,
            missing,
            None,
            ds_max_dim,
            ds_scale_factor,
        )
    for idx in uniq:
        if not (out_path / f"{idx:06d}.jpg").is_file():
            raise RuntimeError(
                f"Anchor remainder: failed to prepare JPEG for frame {idx} "
                f"(checked annotated_frames, preview frames/, and video decode)."
            )


def _to_sam_obj_id_from_npz_key(obj_id_str: str) -> int:
    if "_" in obj_id_str:
        parts = obj_id_str.split("_")
        return int(parts[0]) * 1000 + int(parts[1])
    return int(obj_id_str)


def _obj_config_key_from_sam_oid(objects_meta: dict, sam_oid: int) -> str:
    for oid_str in objects_meta.keys():
        if _to_sam_obj_id_from_npz_key(oid_str) == sam_oid:
            return oid_str
    return str(sam_oid)


def _persist_predicted_anchor_frame(pid: str, vid: str, frame_idx: int,
                                    masks_by_oid: dict[int, np.ndarray],
                                    objects_meta: dict) -> None:
    """Persist masks and bboxes for an auto-filled anchor (no synthetic point prompts)."""
    if not masks_by_oid:
        return
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    merged_masks: dict[str, np.ndarray] = {
        str(k): np.asarray(v).astype(np.uint8) for k, v in ms.load_masks_dense(frame_idx).items()
    }

    bbox_json: dict[str, list] = {}
    for oid_int, m in masks_by_oid.items():
        cfg_key = _obj_config_key_from_sam_oid(objects_meta, oid_int)
        merged_masks[cfg_key] = np.squeeze(m).astype(np.uint8)
        bx, score = _bbox_norm_xywh_score(merged_masks[cfg_key] > 0)
        bbox_json[cfg_key] = list(bx) + [score]

    ms.save_frame(frame_idx, merged_masks, bbox_json)
    _invalidate_mask_cache(pid, vid)


async def _run_anchor_remainder_inference_bg(
    pid: str,
    vid: str,
    state_rm: AnchorRemainderState,
    interactive: bool,
) -> None:
    loop = asyncio.get_event_loop()
    try:
        video_f = pm.get_video(pid, vid)
        if video_f is None:
            await state_rm.publish({"event": "error", "data": json.dumps({"error": "video not found"})})
            return

        start_f = int(video_f.get("start_frame") or 0)
        nframes = int(video_f["num_frames"])
        batch_sz = video_anchor_batch_size(video_f)
        anchors = compute_anchor_frames(start_f, nframes, batch_sz)

        annotated = sorted(set(video_f.get("annotated_anchors") or []))
        annotated_set = set(annotated)

        prefix_n = min(ANCHOR_MANUAL_PREFIX_COUNT, len(anchors))
        required_manual = anchors[:prefix_n]
        if not all(a in annotated_set for a in required_manual):
            await state_rm.publish({
                "event": "error",
                "data": json.dumps({
                    "error": f"Commit the first {len(required_manual)} anchor frame(s) manually before predicting the rest.",
                    "required_manual": required_manual,
                }),
            })
            return

        queue = [
            anchors[i] for i in range(len(anchors))
            if i >= prefix_n and anchors[i] not in annotated_set
        ]
        if not queue:
            vid_up = pm.get_video(pid, vid)
            if vid_up is not None:
                req_a = compute_anchor_frames(start_f, int(vid_up["num_frames"]), batch_sz)
                ann_now = sorted(set(vid_up.get("annotated_anchors") or []))
                if bool(req_a) and all(a in ann_now for a in req_a):
                    pm.update_video(pid, vid, {"anchor_labeling_complete": True})
            await state_rm.publish({"event": "done", "data": json.dumps({"status": "nothing_to_infer"})})
            return

        source_path = video_f.get("source_path", "")
        if not source_path or not Path(source_path).is_file():
            await state_rm.publish({"event": "error", "data": json.dumps({"error": "missing source_path"})})
            return

        objects_meta = dict(video_f.get("objects") or {})
        ds_max_dim, ds_scale_factor = _video_ds_params(video_f)

        anchor_idx_set = set(anchors)
        # Freeze manual anchors at job start — inferred anchors must not seed later steps.
        manual_seed_frames = sorted(a for a in annotated if a in anchor_idx_set)
        manual_seed_set = set(manual_seed_frames)
        all_prompts = pm.get_all_point_prompts(pid, vid)

        async def publish(ev: str, payload: dict):
            await state_rm.publish({"event": ev, "data": json.dumps(payload)})

        total_q = len(queue)
        await publish("init", {
            "anchors_total": len(anchors),
            "manual_prefix": prefix_n,
            "anchors_queued": total_q,
            "manual_seed_anchors": manual_seed_frames,
        })

        try:
            for qi, tgt in enumerate(queue):
                idx_in_list = anchors.index(tgt)
                await publish("predict_start", {"frame_idx": tgt, "anchor_index": idx_in_list, "step": qi + 1, "steps": total_q})

                tmp_dir = tempfile.mkdtemp(prefix=f"sam3wt_anchor_remainder_{tgt}_")
                try:
                    frames_needed = sorted(manual_seed_set | {tgt})
                    await loop.run_in_executor(
                        None,
                        functools.partial(
                            _ensure_jpg_for_indices,
                            pid,
                            vid,
                            tmp_dir,
                            frames_needed,
                            source_path,
                            ds_max_dim,
                            ds_scale_factor,
                        ),
                    )
                    await loop.run_in_executor(None, sam.init_session, pid, vid, tmp_dir)
                    await publish("sam_propagate_start", {"frame_idx": tgt, "step": qi + 1, "steps": total_q})

                    def _seed_and_propagate():
                        frame_map = sam._frame_maps.get((pid, vid), [])
                        seed_items: list[tuple[int, int, dict]] = []
                        for obj_id_str, frame_prompts in all_prompts.items():
                            oid_for_sam = _to_sam_obj_id_from_npz_key(obj_id_str)
                            for fidx_str, prompt in frame_prompts.items():
                                fidx = int(fidx_str)
                                if fidx in manual_seed_set and fidx in frame_map:
                                    seed_items.append((fidx, oid_for_sam, prompt))
                        seed_items.sort(key=lambda x: (x[0], x[1]))
                        if not seed_items:
                            raise RuntimeError(
                                "No saved point prompts on manual anchor frames — "
                                "cannot seed anchor remainder inference."
                            )
                        for fidx, oid_for_sam, prompt in seed_items:
                            sam.add_points(
                                pid,
                                vid,
                                frame_idx=fidx,
                                obj_id=oid_for_sam,
                                points=prompt["points"],
                                labels=prompt["labels"],
                            )
                        forward_start_real = min(manual_seed_frames)
                        try:
                            sam_start_idx = frame_map.index(forward_start_real)
                            tgt_sam_idx = frame_map.index(tgt)
                        except ValueError as err:
                            raise RuntimeError(
                                f"Anchor remainder: frame map missing start={forward_start_real} "
                                f"or target={tgt}"
                            ) from err
                        max_track = tgt_sam_idx - sam_start_idx
                        if max_track <= 0:
                            raise RuntimeError(
                                f"Anchor remainder: target frame {tgt} is not after manual anchors "
                                f"in sparse session ordering"
                            )
                        last_fm: dict[int, np.ndarray] = {}
                        for item in sam.propagate_stream(
                            pid,
                            vid,
                            start_frame_idx=forward_start_real,
                            propagation_direction="forward",
                            max_frame_num_to_track=max_track,
                        ):
                            real_frame, fm, _fb = _extract_frame_masks_bboxes(pid, vid, item)
                            if real_frame == tgt:
                                last_fm = dict(fm)
                        return last_fm

                    frame_masks_final = await loop.run_in_executor(None, _seed_and_propagate)
                    if not frame_masks_final:
                        raise RuntimeError(f"No masks produced at anchor frame {tgt}")

                    await loop.run_in_executor(
                        None,
                        functools.partial(
                            _persist_predicted_anchor_frame,
                            pid, vid, tgt, frame_masks_final, objects_meta,
                        ),
                    )

                    vf = pm.get_video(pid, vid)
                    ann2 = sorted(set((vf.get("annotated_anchors") or []) + [tgt]))
                    req_a = compute_anchor_frames(start_f, int(vf["num_frames"]), batch_sz)
                    pm.update_video(pid, vid, {
                        "annotated_anchors": ann2,
                        "anchor_labeling_complete": bool(req_a) and all(a in set(ann2) for a in req_a),
                    })

                    payload_review = {
                        "frame_idx": tgt,
                        "anchor_index": idx_in_list,
                        "anchors_done": qi + 1,
                        "anchors_queued": total_q,
                    }
                    await publish("anchor_predicted", payload_review)
                    if interactive:
                        await publish("review_prompt", payload_review)
                        state_rm.review_continue_event = asyncio.Event()
                        state_rm.waiting_review = True
                        await state_rm.review_continue_event.wait()
                        state_rm.waiting_review = False
                        state_rm.review_continue_event = None
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    await loop.run_in_executor(None, sam.close_session, pid, vid)

            await publish("done", {"status": "ok", "anchors_inferred": total_q})

        except Exception as ex:
            logger.error(f"Anchor remainder inference error: {ex}", exc_info=True)
            await publish("error", {"error": str(ex)})

    finally:
        state_rm.is_running = False
        state_rm.waiting_review = False
        state_rm.review_continue_event = None

# ─── Instance color helpers ───────────────────────────────────────────────────
# For multi-instance objects (e.g., "1", "1_1", "1_2"), we vary the color
# to distinguish each instance visually.

def _lighten_color(hex_color: str, amount: int) -> str:
    """Lighten a hex color by adding `amount` to each RGB channel."""
    hex_color = hex_color.lstrip('#')
    r = min(255, int(hex_color[0:2], 16) + amount)
    g = min(255, int(hex_color[2:4], 16) + amount)
    b = min(255, int(hex_color[4:6], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"

def _shift_hue(hex_color: str, shift: int) -> str:
    """Shift hue by rotating RGB channels and adjusting brightness."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Rotate and add variance
    if shift == 1:
        r, g, b = g, b, r
    elif shift == 2:
        r, g, b = b, r, g
    # Add some brightness variance
    factor = 1.0 + (shift % 3) * 0.15
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def _get_instance_color(obj_id: str, objects: dict) -> str:
    """
    Get color for an object instance. For multi-instance keys like "1_1",
    derive a variation of the base object's color.
    """
    # Parse instance key: "1" -> base_id="1", instance=0; "1_2" -> base_id="1", instance=2
    if '_' in obj_id:
        parts = obj_id.rsplit('_', 1)
        base_id = parts[0]
        try:
            instance_num = int(parts[1])
        except ValueError:
            instance_num = 0
    else:
        base_id = obj_id
        instance_num = 0

    base_color = objects.get(str(base_id), {}).get("color", "#5B8DD9")

    if instance_num == 0:
        return base_color

    # Generate a visually distinct color for each instance
    shifted = _shift_hue(base_color, instance_num)
    lightened = _lighten_color(shifted, instance_num * 25)
    return lightened


# ─── Mask encode cache ────────────────────────────────────────────────────────
# encode_mask_as_png is expensive (numpy + cv2 contour/dilate + PIL PNG encode).
# Cache keyed by (pid, vid, frame_idx, sqlite_revision_or_legacy_mtime_ns).

_mask_encode_cache: dict[tuple, dict] = {}
_MASK_CACHE_MAX = 2000  # max entries (~2 KB overhead per entry, masks are large)


def _get_encoded_masks(pid: str, vid: str, fidx: int, objects: dict) -> dict:
    """Load and encode masks with an in-memory cache keyed by storage revision."""
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    if not ms.has_masks(fidx):
        return {}

    rev_part = ms.png_cache_revision_part(fidx)
    key = ("masks_png", pid, vid, int(fidx), rev_part)
    if key in _mask_encode_cache:
        return _mask_encode_cache[key]

    raw_masks = ms.load_masks_dense(fidx)
    mask_b64: dict[str, str] = {}
    for obj_id, mask in raw_masks.items():
        obj_color = _get_instance_color(str(obj_id), objects)
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

    if len(_mask_encode_cache) >= _MASK_CACHE_MAX:
        # Evict oldest 10% of entries
        evict = _MASK_CACHE_MAX // 10
        for k in list(_mask_encode_cache.keys())[:evict]:
            del _mask_encode_cache[k]

    _mask_encode_cache[key] = mask_b64
    return mask_b64


def _invalidate_mask_cache(pid: str, vid: str) -> None:
    """Remove all cached PNG entries for a given video."""
    prefix = ("masks_png", pid, vid)
    for k in [k for k in _mask_encode_cache if isinstance(k, tuple) and len(k) >= 4 and k[:3] == prefix]:
        del _mask_encode_cache[k]


def _get_prop_state(pid: str, vid: str) -> PropagationState:
    key = f"{pid}/{vid}"
    if key not in _prop_registry:
        _prop_registry[key] = PropagationState()
    return _prop_registry[key]


# ─── Startup: eagerly load SAM model ─────────────────────────────────────────

@app.on_event("startup")
async def startup_load_model():
    """Load the SAM model into GPU memory at server startup."""
    import threading

    def _load():
        global _model_loading, _model_load_error
        _model_loading = True
        try:
            sam.ensure_loaded()
            logger.info(f"SAM model loaded at startup: {sam.model_name()}")
        except Exception as e:
            _model_load_error = str(e)
            logger.error(f"SAM model failed to load at startup: {e}")
        finally:
            _model_loading = False

    threading.Thread(target=_load, daemon=True).start()


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    loaded = sam.is_loaded()
    return {
        "status": "ok",
        "sam_ready": sam.is_ready(),       # checkpoint exists on disk
        "sam_loaded": loaded,              # model in GPU memory
        "sam_loading": _model_loading,     # currently loading
        "sam_model": sam.model_name() if loaded else "loading" if _model_loading else "none",
        "sam_load_error": _model_load_error,
    }


# ─── Projects ────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def list_projects():
    return pm.list_projects()


@app.get("/api/projects/root")
def get_projects_root():
    """Active projects directory and env-configured default (SAM3_TRACKING_PROJECTS_DIR → SAM3_PROJECTS_DIR → ~/.sam3_zero_projects)."""
    return pm.get_projects_root_info()


class SetProjectsRootRequest(BaseModel):
    path: str


@app.post("/api/projects/root")
def set_projects_root(req: SetProjectsRootRequest):
    try:
        pm.set_projects_root(req.path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return pm.get_projects_root_info()


class CreateProjectRequest(BaseModel):
    name: str


@app.post("/api/projects", status_code=201)
def create_project(req: CreateProjectRequest):
    return pm.create_project(req.name)


class MergeProjectsRequest(BaseModel):
    name: str
    left_project_id: str
    right_project_id: str
    output_parent: Optional[str] = None


@app.post("/api/projects/merge", status_code=201)
def merge_projects(req: MergeProjectsRequest):
    if not req.name.strip():
        raise HTTPException(400, "Project name is required")
    if not req.left_project_id.strip() or not req.right_project_id.strip():
        raise HTTPException(400, "Both source projects are required")
    try:
        output_parent = (
            Path(req.output_parent).expanduser() if req.output_parent else None
        )
        return pm.merge_projects(
            name=req.name.strip(),
            left_ref=req.left_project_id.strip(),
            right_ref=req.right_project_id.strip(),
            output_parent=output_parent,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    project = pm.get_project(pid)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None


@app.patch("/api/projects/{pid}")
def update_project(pid: str, req: UpdateProjectRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    return pm.update_project(pid, updates)


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: str):
    pm.delete_project(pid)


# ─── Videos ──────────────────────────────────────────────────────────────────

@app.post("/api/projects/{pid}/videos", status_code=201)
async def add_video(
    pid: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_dim: Optional[int] = Form(None),
    scale_factor: Optional[float] = Form(None),
):
    project = pm.get_project(pid)
    if project is None:
        raise HTTPException(404, "Project not found")

    # Save upload to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Get basic video info
    try:
        info = get_video_info(tmp_path)
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid video file: {e}")

    # Register video in project
    video_meta = pm.add_video(
        pid=pid,
        name=file.filename,
        source_path=tmp_path,
        num_frames=info["num_frames"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
    )
    vid = video_meta["id"]

    # Move upload to permanent location
    perm_path = pm.video_dir(pid, vid) / f"source{suffix}"
    shutil.move(tmp_path, str(perm_path))
    pm.update_video(pid, vid, {"source_path": str(perm_path)})

    # Compute effective dimensions for lazy downsampling (no re-encoding)
    ds_max_dim = max_dim if (max_dim and max_dim > 0) else None
    ds_scale_factor = scale_factor if (scale_factor and scale_factor > 1.0) else None
    eff_w, eff_h = compute_ds_dims(info["width"], info["height"], ds_max_dim, ds_scale_factor)

    updates: dict = {"frames_extracted": True, "all_frames_extracted": False}
    if ds_max_dim is not None:
        updates["downsample_max_dim"] = ds_max_dim
    if ds_scale_factor is not None:
        updates["downsample_scale_factor"] = ds_scale_factor
    if (eff_w, eff_h) != (info["width"], info["height"]):
        updates["width"] = eff_w
        updates["height"] = eff_h

    def do_post_upload():
        try:
            ensure_faststart(str(perm_path))
        except Exception as e:
            logger.warning(f"faststart failed for {pid}/{vid}: {e}")
        pm.update_video(pid, vid, updates)

    background_tasks.add_task(do_post_upload)

    return video_meta


# ─── Import video from server filesystem path ────────────────────────────────

class ImportVideoRequest(BaseModel):
    path: str
    max_dim: Optional[int] = None
    scale_factor: Optional[float] = None
    symlink: bool = False


@app.post("/api/projects/{pid}/videos/import", status_code=201)
async def import_video(
    pid: str,
    req: ImportVideoRequest,
    background_tasks: BackgroundTasks,
):
    project = pm.get_project(pid)
    if project is None:
        raise HTTPException(404, "Project not found")

    src = Path(req.path).resolve()
    if not src.exists():
        raise HTTPException(400, f"File not found: {req.path}")
    if not src.is_file():
        raise HTTPException(400, f"Not a file: {req.path}")

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if src.suffix.lower() not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {src.suffix}. Allowed: {', '.join(allowed_ext)}")

    try:
        info = get_video_info(str(src))
    except Exception as e:
        raise HTTPException(400, f"Invalid video file: {e}")

    # Compute effective dimensions for lazy downsampling (frames are resized on extraction;
    # the source file is NEVER modified regardless of symlink/copy mode).
    ds_max_dim = req.max_dim if (req.max_dim and req.max_dim > 0) else None
    ds_scale_factor = req.scale_factor if (req.scale_factor and req.scale_factor > 1.0) else None
    eff_w, eff_h = compute_ds_dims(info["width"], info["height"], ds_max_dim, ds_scale_factor)

    video_meta = pm.add_video(
        pid=pid,
        name=src.name,
        source_path=str(src),
        num_frames=info["num_frames"],
        fps=info["fps"],
        width=eff_w,
        height=eff_h,
    )
    vid = video_meta["id"]

    perm_path = pm.video_dir(pid, vid) / f"source{src.suffix}"

    if req.symlink:
        # Create a symlink — original file is never touched.
        os.symlink(str(src), str(perm_path))
        logger.info(f"Symlinked {src} → {perm_path}")
    else:
        shutil.copy2(str(src), str(perm_path))

    updates: dict = {
        "source_path": str(perm_path),
        "frames_extracted": True,
        "all_frames_extracted": False,
    }
    if ds_max_dim is not None:
        updates["downsample_max_dim"] = ds_max_dim
    if ds_scale_factor is not None:
        updates["downsample_scale_factor"] = ds_scale_factor

    pm.update_video(pid, vid, updates)

    def do_post_import():
        if req.symlink:
            # Never apply faststart to the symlink target — it would modify the original.
            # The /faststart endpoint provides an explicit escape hatch if needed.
            pass
        else:
            try:
                ensure_faststart(str(perm_path))
            except Exception as e:
                logger.warning(f"faststart failed for imported {pid}/{vid}: {e}")

    background_tasks.add_task(do_post_import)

    return video_meta


# ─── Browse server directory ─────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".h264", ".ts",
                    ".m4v", ".mts", ".m2ts", ".flv", ".wmv", ".mpg", ".mpeg"}

@app.get("/api/browse")
def browse_directory(path: str = Query(...), depth: int = Query(1)):
    """
    List files in a server directory for folder-import selection.
    depth=1  → current directory only (default)
    depth=N  → recurse up to N levels
    depth=0  → unlimited recursion
    """
    root = Path(path)
    if not root.exists():
        raise HTTPException(400, f"Path not found: {path}")
    if not root.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")

    max_depth = depth if depth > 0 else 10_000  # 0 = unlimited

    def _collect(directory: Path, current_depth: int) -> list[dict]:
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name.lower())
        except PermissionError:
            return []
        result = []
        for entry in entries:
            if entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                result.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size": size,
                    "is_video": entry.suffix.lower() in VIDEO_EXTENSIONS,
                })
            elif entry.is_dir() and current_depth < max_depth:
                result.extend(_collect(entry, current_depth + 1))
        return result

    files = _collect(root, 1)
    return {"directory": str(root), "files": files}


@app.get("/api/fs/list_dir")
def fs_list_dir(path: str = Query("")):
    """
    One-level directory listing for workspace / folder pickers.
    Empty path lists the current user's home directory.
    """
    raw = path.strip()
    try:
        if not raw:
            target = Path.home().resolve()
        else:
            target = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise HTTPException(400, f"Invalid path: {e}")
    if not target.exists():
        raise HTTPException(404, f"Path not found: {raw or path}")
    if not target.is_dir():
        raise HTTPException(400, f"Not a directory: {target}")
    parent = None if target.parent == target else str(target.parent)
    entries: list[dict] = []
    try:
        subs = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    for entry in subs:
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        is_project = False
        if is_dir:
            cfg = entry / "config.json"
            if cfg.is_file():
                try:
                    meta = json.loads(cfg.read_text(encoding="utf-8"))
                    is_project = isinstance(meta.get("id"), str) and len(meta.get("id", "")) > 0
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    is_project = False
        try:
            rpath = str(entry.resolve())
        except OSError:
            rpath = str(entry)
        entries.append({
            "name": entry.name,
            "path": rpath,
            "is_dir": is_dir,
            "is_project": is_project,
        })
    return {"path": str(target), "parent": parent, "entries": entries}


# ─── Downsample existing video ───────────────────────────────────────────────

class DownsampleRequest(BaseModel):
    max_dim: Optional[int] = None
    scale_factor: Optional[float] = None


@app.post("/api/projects/{pid}/videos/{vid}/downsample")
async def downsample_video_endpoint(pid: str, vid: str, req: DownsampleRequest):
    """
    Set lazy downsample parameters for a video.
    Frames are resized on extraction — the source file is never modified.
    Clears cached frames/ and annotated_frames/ so they are re-extracted at the new size.
    """
    if not req.max_dim and not req.scale_factor:
        raise HTTPException(400, "Provide either max_dim or scale_factor")
    if req.max_dim is not None and req.max_dim <= 0:
        raise HTTPException(400, "max_dim must be > 0")
    if req.scale_factor is not None and req.scale_factor <= 1.0:
        raise HTTPException(400, "scale_factor must be > 1.0")

    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    # Derive native dims from the source file (ignoring any prior lazy-ds stored dims).
    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(404, "Video source file not found")

    try:
        native = get_video_info(str(Path(source_path).resolve()))
    except Exception as e:
        raise HTTPException(500, f"Cannot read video info: {e}")

    ds_max_dim = req.max_dim if req.max_dim and req.max_dim > 0 else None
    ds_scale_factor = req.scale_factor if req.scale_factor and req.scale_factor > 1.0 else None

    eff_w, eff_h = compute_ds_dims(native["width"], native["height"], ds_max_dim, ds_scale_factor)

    if (eff_w, eff_h) == (native["width"], native["height"]):
        return {
            "status": "skipped",
            "message": f"Video is already within the requested size ({eff_w}×{eff_h})",
            "width": eff_w,
            "height": eff_h,
            "num_frames": native["num_frames"],
        }

    # Clear cached frames — they were extracted at the old resolution.
    for d in [pm.frames_dir(pid, vid), pm.annotated_frames_dir(pid, vid)]:
        if d.exists():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)

    updates: dict = {"width": eff_w, "height": eff_h}
    # Clear any previously stored ds params before storing new ones.
    updates["downsample_max_dim"] = ds_max_dim
    updates["downsample_scale_factor"] = ds_scale_factor

    pm.update_video(pid, vid, updates)
    logger.info(f"Lazy downsample set for {pid}/{vid}: {eff_w}×{eff_h}")

    return {
        "status": "done",
        "width": eff_w,
        "height": eff_h,
        "num_frames": native["num_frames"],
    }


@app.get("/api/projects/{pid}/videos/{vid}")
def get_video(pid: str, vid: str):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return video


@app.post("/api/projects/{pid}/videos/{vid}/faststart")
def apply_faststart(pid: str, vid: str):
    """Manually apply MP4 faststart optimization to an existing video."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(404, "Video source file not found")
    # If it's a symlink, replace with a copy first (can't modify original)
    p = Path(source_path)
    if p.is_symlink():
        real = p.resolve()
        p.unlink()
        shutil.copy2(str(real), str(p))
    result = ensure_faststart(source_path)
    return {"status": "ok", "path": result}


@app.get("/api/projects/{pid}/videos/{vid}/info")
def video_info(pid: str, vid: str):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    frames_dir = pm.frames_dir(pid, vid)
    actual_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
    return {
        "num_frames": video["num_frames"],
        "actual_frames": actual_frames,
        "frames_extracted": video.get("frames_extracted", actual_frames > 0),
        "all_frames_extracted": video.get("all_frames_extracted", False),
        "preview_indices": video.get("preview_indices", []),
        "fps": video["fps"],
        "width": video["width"],
        "height": video["height"],
    }


class UpdateVideoRequest(BaseModel):
    start_frame: Optional[int] = None
    anchor_batch_size: Optional[int] = None


@app.patch("/api/projects/{pid}/videos/{vid}")
def update_video_meta(pid: str, vid: str, req: UpdateVideoRequest):
    """Update mutable video metadata fields (e.g. start_frame, anchor_batch_size)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        return video
    if "anchor_batch_size" in updates:
        if anchor_labeling_has_started(video):
            raise HTTPException(
                409,
                "Anchor labeling has already started — anchor frame interval cannot be changed.",
            )
        updates["anchor_batch_size"] = normalize_anchor_batch_size(updates["anchor_batch_size"])
    return pm.update_video(pid, vid, updates)


@app.delete("/api/projects/{pid}/videos/{vid}", status_code=204)
def remove_video(pid: str, vid: str):
    sam.close_session(pid, vid)
    pm.remove_video(pid, vid)


@app.post("/api/projects/{pid}/videos/{vid}/reset", status_code=200)
def reset_video(pid: str, vid: str):
    """
    Clear all annotations and tracking data for a video while keeping the
    video file itself.  Removes:
      - All objects and point prompts from config.json
      - Saved masks (masks/) and bboxes (bboxes/*.json)
      - Annotated frames (annotated_frames/)
      - Preview frames (frames/)
      - Active SAM session
      - propagation_complete / propagated_frames flags
      - In-memory mask cache
    """
    logger.info(f"=== RESET VIDEO {vid} in project {pid} ===")

    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    # Log what we're about to clear
    logger.info(f"  Current objects: {list(video.get('objects', {}).keys())}")
    logger.info(f"  Current point_prompts: {list(video.get('point_prompts', {}).keys())}")

    # Close any active SAM session
    sam.close_session(pid, vid)

    # Invalidate in-memory mask cache for this video
    _invalidate_mask_cache(pid, vid)

    VideoMaskStorage(pm.video_dir(pid, vid)).wipe_sqlite_file()

    # Wipe masks, bboxes, and frame directories
    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)
    ann_dir = pm.annotated_frames_dir(pid, vid)
    frames_dir = pm.frames_dir(pid, vid)

    for d in (masks_dir, bboxes_dir, ann_dir, frames_dir):
        if d.exists():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)

    # Reset config: clear objects, prompts, propagation state
    pm.clear_propagated_frames(pid, vid)
    pm.update_video(pid, vid, {
        "objects": {},
        "point_prompts": {},
        "instance_groups": {},
        "sam3_session_id": None,
        "propagation_complete": False,
        "frames_extracted": False,
        "all_frames_extracted": False,
        "preview_indices": [],
        "annotated_anchors": [],
        "anchor_labeling_timing": {"frames": {}, "video": {}},
        "anchor_labeling_complete": False,
        "whole_video_inference": {"status": "none", "updated_at": None, "host": None},
    })

    # Verify reset was successful
    video_after = pm.get_video(pid, vid)
    logger.info(f"  After reset - objects: {list(video_after.get('objects', {}).keys())}")
    logger.info(f"  After reset - point_prompts: {list(video_after.get('point_prompts', {}).keys())}")
    logger.info(f"=== RESET COMPLETE for {vid} ===")

    return {"status": "ok"}


# ─── Frames ──────────────────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/frames/{fidx}")
def get_frame(pid: str, vid: str, fidx: int, thumb: bool = False):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    frames_dir = pm.frames_dir(pid, vid)
    frame_path = frames_dir / f"{fidx:06d}.jpg"

    # If the specific frame hasn't been extracted yet, extract it on-the-fly
    if not frame_path.exists():
        source_path = video.get("source_path")
        if source_path and Path(source_path).exists():
            ds_max_dim, ds_scale_factor = _video_ds_params(video)
            try:
                extract_frame_range(
                    source_path, str(frames_dir), fidx, fidx + 1,
                    max_dim=ds_max_dim, scale_factor=ds_scale_factor,
                )
            except Exception:
                pass

    if not frame_path.exists():
        raise HTTPException(404, f"Frame {fidx} not found")
    if thumb:
        data = generate_thumbnail(str(frame_path), height=60)
        if data:
            return StreamingResponse(iter([data]), media_type="image/jpeg",
                                     headers={"Cache-Control": "public, max-age=86400, immutable"})
    return FileResponse(str(frame_path), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400, immutable"})


# ─── Extract single frame for annotation ─────────────────────────────────────

@app.post("/api/projects/{pid}/videos/{vid}/extract_frame/{fidx}")
def extract_annotated_frame(pid: str, vid: str, fidx: int):
    """
    Extract a single frame from the source video into annotated_frames/.
    Used when the user clicks to annotate — we only extract frames on demand.
    Returns the path info so the caller knows it's ready.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(400, "Video source file not found")

    ann_dir = pm.annotated_frames_dir(pid, vid)
    ann_dir.mkdir(parents=True, exist_ok=True)
    frame_path = ann_dir / f"{fidx:06d}.jpg"

    if not frame_path.exists():
        ds_max_dim, ds_scale_factor = _video_ds_params(video)
        try:
            extract_frame_range(
                source_path, str(ann_dir), fidx, fidx + 1,
                max_dim=ds_max_dim, scale_factor=ds_scale_factor,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to extract frame {fidx}: {e}")

    if not frame_path.exists():
        raise HTTPException(500, f"Frame {fidx} could not be extracted")

    return {"status": "ok", "frame_idx": fidx, "path": str(frame_path)}


# ─── Video source (mp4 streaming with Range support) ─────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/source")
async def get_video_source(pid: str, vid: str, request: Request):
    """
    Serve the original video file with HTTP Range request support so the
    browser can start playing immediately and seek without downloading
    the entire file first.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(404, "Video source file not found")

    file_path = Path(source_path).resolve()  # resolve symlinks
    if not file_path.exists():
        raise HTTPException(404, "Video source file not found (resolved)")
    file_size = file_path.stat().st_size
    suffix = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    content_type = media_types.get(suffix, "video/mp4")

    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        range_spec = range_header.strip().lower()
        if not range_spec.startswith("bytes="):
            raise HTTPException(416, "Invalid range header")
        range_val = range_spec[6:]
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)

        if start >= file_size:
            raise HTTPException(416, "Range not satisfiable")

        chunk_size = end - start + 1

        def iter_range():
            CHUNK = 1024 * 1024  # 1 MB chunks
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(CHUNK, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )
    else:
        # No Range header: use Starlette's FileResponse which efficiently
        # serves the whole file and handles etag / last-modified / conditional.
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )


# ─── SAM3 Session ─────────────────────────────────────────────────────────────

@app.post("/api/projects/{pid}/videos/{vid}/session")
def init_session(pid: str, vid: str):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    # Prefer annotated_frames dir (single-frame annotation workflow)
    ann_dir = pm.annotated_frames_dir(pid, vid)
    if ann_dir.exists() and list(ann_dir.glob("*.jpg")):
        frames_dir = str(ann_dir)
    else:
        frames_dir = str(pm.frames_dir(pid, vid))
        if not Path(frames_dir).exists() or not list(Path(frames_dir).glob("*.jpg")):
            raise HTTPException(400, "No frames available. Extract a frame first.")
    try:
        session_id = sam.init_session(pid, vid, frames_dir)
        pm.update_video(pid, vid, {"sam3_session_id": session_id})
        return {"session_id": session_id, "status": "initialized", "model": sam.model_name()}
    except Exception as e:
        raise HTTPException(500, f"Failed to initialize SAM session: {e}")


@app.delete("/api/projects/{pid}/videos/{vid}/session", status_code=204)
def close_session(pid: str, vid: str):
    sam.close_session(pid, vid)
    pm.update_video(pid, vid, {"sam3_session_id": None})


@app.get("/api/projects/{pid}/videos/{vid}/session/state")
def get_session_state(pid: str, vid: str):
    """
    Return a diagnostic snapshot of the current SAM inference state for this video.
    Includes session info, frame map, cached outputs, point prompts, and saved masks.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    session_id = sam.get_session_id(pid, vid)
    frame_map: list[int] = list(sam._frame_maps.get((pid, vid), []))  # real idx by SAM position
    model = sam.model_name()

    # SAM-internal state introspection
    cached_sam_indices: list[int] = []
    action_history_len: int = 0
    obj_ids_tracked: list[int] = []

    if session_id and model == "sam3":
        try:
            predictor = _get_predictor()
            state = predictor._ALL_INFERENCE_STATES.get(session_id, {}).get("state", {})
            cached_sam_indices = sorted(state.get("cached_frame_outputs", {}).keys())
            action_history_len = len(state.get("action_history", []))
            tracker_meta = state.get("tracker_metadata") or {}
            obj_ids_tracked = [int(o) for o in tracker_meta.get("obj_ids_all_gpu", [])]
        except Exception as e:
            logger.warning(f"Could not introspect SAM3 state: {e}")

    elif session_id and model == "sam2":
        try:
            state = sam._sam2_states.get((pid, vid))
            if state is not None:
                cached_sam_indices = sorted(state.get("cached_features", {}).keys())
        except Exception as e:
            logger.warning(f"Could not introspect SAM2 state: {e}")

    # Point prompts from project config
    point_prompts: dict = video.get("point_prompts", {})

    # Saved masks on disk (SQLite store + legacy npz)
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    saved_mask_frames: list[int] = []
    saved_mask_obj_counts: dict[int, int] = {}
    try:
        saved_mask_frames = ms.iter_saved_frame_indices_sorted()
        for fidx in saved_mask_frames:
            try:
                saved_mask_obj_counts[fidx] = ms.mask_object_count(fidx)
            except Exception:
                saved_mask_obj_counts[fidx] = 0
    except Exception:
        pass

    # Annotated frames on disk
    ann_dir = pm.annotated_frames_dir(pid, vid)
    annotated_frame_files: list[int] = []
    if ann_dir.exists():
        for f in sorted(ann_dir.glob("*.jpg")):
            try:
                annotated_frame_files.append(int(f.stem))
            except ValueError:
                pass

    return {
        "session_active": session_id is not None,
        "session_id": session_id,
        "model": model,
        "frame_map": frame_map,
        "cached_sam_indices": cached_sam_indices,
        "action_history_len": action_history_len,
        "obj_ids_tracked": obj_ids_tracked,
        "point_prompts": point_prompts,
        "saved_mask_frames": saved_mask_frames,
        "saved_mask_obj_counts": saved_mask_obj_counts,
        "annotated_frame_files": annotated_frame_files,
        "num_frames": video.get("num_frames", 0),
        "objects": video.get("objects", {}),
    }


class RebuildSessionRequest(BaseModel):
    return_masks_for_frames: list[int] = []
    anchor_mode: bool = False
    anchor_frame: Optional[int] = None


@app.post("/api/projects/{pid}/videos/{vid}/session/rebuild_from_config")
def rebuild_session_from_config(pid: str, vid: str, req: RebuildSessionRequest):
    """
    Re-initialize SAM from annotated_frames and replay all point prompts from config.
    Optionally re-encode and persist masks for requested frames (for undo/redo).
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    prop_state = _get_prop_state(pid, vid)
    if prop_state is not None and prop_state.is_running:
        raise HTTPException(409, "Propagation is running — pause first before rebuilding session.")

    ann_dir_path = pm.annotated_frames_dir(pid, vid)
    ann_dir = str(ann_dir_path)
    if not ann_dir_path.exists() or not list(ann_dir_path.glob("*.jpg")):
        raise HTTPException(400, "No annotated frames extracted yet. Extract a frame first.")

    try:
        if req.anchor_mode and req.anchor_frame is not None:
            af = req.anchor_frame
            start_f = video.get("start_frame", 0)
            num_frames_v = video["num_frames"]
            anchor_frames = compute_anchor_frames(
                start_f, num_frames_v, video_anchor_batch_size(video),
            )
            if af not in anchor_frames:
                raise HTTPException(
                    409,
                    f"Anchor mode: frame {af} is not an anchor frame (anchors: {anchor_frames}).",
                )
            last_out = _anchor_session_reinit_and_replay_frame(pid, vid, af, ann_dir_path)
        else:
            last_out = _full_ann_session_reinit_and_replay(
                pid, vid, ann_dir, collect_last_outputs=True
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"rebuild_from_config failed: {e}", exc_info=True)
        raise HTTPException(500, f"Session rebuild failed: {e}")

    video = pm.get_video(pid, vid) or video
    objects = video["objects"]
    masks_by_frame: dict[str, dict[str, str]] = {}
    for fidx in req.return_masks_for_frames:
        fo = last_out.get(fidx)
        if not fo:
            continue
        mask_b64, raw_masks = _encode_masks_from_sam_frame_output(fo, objects)
        if raw_masks:
            _persist_merged_masks_for_frame(pid, vid, fidx, raw_masks)
        if mask_b64:
            masks_by_frame[str(fidx)] = mask_b64

    return {"status": "ok", "masks_by_frame": masks_by_frame}


@app.post("/api/projects/{pid}/videos/{vid}/predict_frame/{frame_idx}")
def predict_frame(pid: str, vid: str, frame_idx: int):
    """Predict masks for ALL objects on a single frame using the annotated state.

    Rebuilds the SAM session from the annotated frames + saved point prompts, then
    returns predicted masks for *frame_idx* (propagating to it from the nearest
    annotated frame if the frame itself is not annotated).  Sub-object containment
    is applied.  Results are NOT persisted and the frame is NOT marked propagated —
    this is a preview using the annotated inference state without adding to it.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    prop_state = _get_prop_state(pid, vid)
    if prop_state is not None and prop_state.is_running:
        raise HTTPException(409, "Propagation is running — pause first before predicting.")

    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(400, "Video source file not found")

    all_prompts = pm.get_all_point_prompts(pid, vid)
    if not all_prompts:
        raise HTTPException(400, "No annotations yet — label at least one frame first.")

    ann_dir_path = pm.annotated_frames_dir(pid, vid)
    ann_dir_path.mkdir(parents=True, exist_ok=True)
    ann_dir = str(ann_dir_path)

    # Ensure the target frame is available to the session.
    target_jpg = ann_dir_path / f"{frame_idx:06d}.jpg"
    if not target_jpg.exists():
        ds_max_dim, ds_scale = _video_ds_params(video)
        try:
            extract_frame_range(source_path, ann_dir, frame_idx, frame_idx + 1,
                                max_dim=ds_max_dim, scale_factor=ds_scale)
        except Exception as e:
            raise HTTPException(500, f"Failed to extract frame {frame_idx}: {e}")
        if not target_jpg.exists():
            raise HTTPException(500, f"Frame {frame_idx} could not be extracted")

    # Rebuild the annotated session (init + replay all saved prompts).
    try:
        last_out = _full_ann_session_reinit_and_replay(
            pid, vid, ann_dir, collect_last_outputs=True
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"predict_frame rebuild failed: {e}", exc_info=True)
        raise HTTPException(500, f"Session rebuild failed: {e}")

    objects = (pm.get_video(pid, vid) or video)["objects"]

    # Point sub-objects: lower their SAM2 binarization threshold during propagation.
    point_sam_ids = {
        _to_sam_obj_id_from_npz_key(oid)
        for oid, meta in objects.items()
        if meta.get("kind") == "point"
    }
    if point_sam_ids:
        sam.register_point_objects(pid, vid, point_sam_ids)

    def _int_masks_from_fo(fo: dict) -> dict:
        out: dict[int, np.ndarray] = {}
        ids = fo.get("out_obj_ids", [])
        masks = fo.get("out_binary_masks", [])
        for i, oid in enumerate(ids):
            m = masks[i] if i < len(masks) else None
            if m is None:
                continue
            if hasattr(m, "numpy"):
                m = m.numpy()
            out[int(oid)] = np.squeeze(m).astype(np.uint8)
        return out

    # If the target frame is itself annotated, replay already produced its masks.
    frame_masks_int: dict = {}
    fo = last_out.get(frame_idx)
    if fo:
        frame_masks_int = _int_masks_from_fo(fo)
    else:
        # Propagate from the nearest annotated frame to the target.  SAM's frame
        # indices are positions in the sparse annotated session, so direction +
        # max_frame_num_to_track are derived from those positions (mirrors the
        # anchor-remainder predictor).
        frame_map = sam._frame_maps.get((pid, vid), [])
        annotated_in_session = sorted(
            int(f) for f in {int(fk) for fp in all_prompts.values() for fk in fp.keys()}
            if int(f) in frame_map
        )
        if not annotated_in_session:
            raise HTTPException(400, "No annotated frames are loaded in the session.")
        start_from = min(annotated_in_session, key=lambda f: abs(f - frame_idx))
        try:
            start_sam = frame_map.index(start_from)
            tgt_sam = frame_map.index(frame_idx)
        except ValueError as e:
            raise HTTPException(500, f"Frame map missing start/target: {e}")
        direction = "forward" if tgt_sam > start_sam else "backward"
        max_track = abs(tgt_sam - start_sam)

        def _run_predict_pass():
            # SAM3 may yield the target frame more than once (an initial pass with
            # unconfirmed/empty masks, then a refined pass).  Keep the LAST non-empty
            # result instead of breaking on the first occurrence.
            last_fm: dict = {}
            for item in sam.propagate_stream(
                pid, vid,
                start_frame_idx=start_from,
                propagation_direction=direction,
                max_frame_num_to_track=max_track,
            ):
                real_frame, fm, _fb = _extract_frame_masks_bboxes(pid, vid, item)
                if real_frame == frame_idx and fm:
                    last_fm = dict(fm)
            return last_fm

        try:
            frame_masks_int = _run_predict_pass()
        except Exception as e:
            logger.error(f"predict_frame propagation failed: {e}", exc_info=True)
            raise HTTPException(500, f"Prediction failed: {e}")

    # Apply sub-object containment, then encode (do NOT persist / mark propagated).
    masks_str, _bboxes = _contain_frame_masks(objects, frame_masks_int)
    mask_b64: dict[str, str] = {}
    for oid, m in masks_str.items():
        if np.asarray(m).any():
            mask_b64[oid] = encode_mask_as_png(m, _get_instance_color(oid, objects))

    return {"frame_idx": frame_idx, "masks": mask_b64}


# ─── Objects ──────────────────────────────────────────────────────────────────

class AddObjectRequest(BaseModel):
    name: str
    color: Optional[str] = None
    description: str = ""
    min_instances: int = 1
    max_instances: int = 1
    parent_id: Optional[str] = None
    kind: str = "segmentation"
    point_blob_frac: float = 0.06


@app.post("/api/projects/{pid}/videos/{vid}/objects", status_code=201)
def add_object(pid: str, vid: str, req: AddObjectRequest):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    try:
        return pm.add_object(
            pid, vid, req.name, req.color,
            description=req.description,
            min_instances=req.min_instances,
            max_instances=req.max_instances,
            parent_id=req.parent_id,
            kind=req.kind,
            point_blob_frac=req.point_blob_frac,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


class UpdateObjectRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    min_instances: Optional[int] = None
    max_instances: Optional[int] = None
    parent_id: Optional[str] = None
    kind: Optional[str] = None
    point_blob_frac: Optional[float] = None
    visible: Optional[bool] = None


@app.patch("/api/projects/{pid}/videos/{vid}/objects/{oid}")
def update_object(pid: str, vid: str, oid: str, req: UpdateObjectRequest):
    # Note: `visible` is a real boolean field, so filtering on `is not None` keeps it;
    # callers that want to clear parent_id (re-parent to root) must use a dedicated path.
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if "name" in updates:
        pm.rename_object(pid, vid, oid, updates.pop("name"))
    if updates:
        try:
            pm.update_object(pid, vid, oid, **updates)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"status": "ok"}


@app.delete("/api/projects/{pid}/videos/{vid}/objects/{oid}", status_code=204)
def remove_object(pid: str, vid: str, oid: str):
    # Recursively removes the object and its descendants from config; mirror that on
    # the SAM session so every removed id stops being tracked.
    removed = pm.remove_object(pid, vid, oid)
    for rid in removed:
        try:
            sam.remove_object(pid, vid, int(rid))
        except Exception:
            pass


class RestoreObjectRequest(BaseModel):
    object: dict
    point_prompts: dict = {}
    instance_group: Optional[list] = None


@app.post("/api/projects/{pid}/videos/{vid}/objects/restore")
def restore_object_snapshot(pid: str, vid: str, req: RestoreObjectRequest):
    """Restore object metadata + point prompts + instance_groups (undo of delete)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    oid_raw = req.object.get("id")
    oid = str(oid_raw) if oid_raw is not None else ""
    if not oid:
        raise HTTPException(422, "object.id is required")
    try:
        pm.restore_object_entry(
            pid, vid, oid,
            req.object,
            req.point_prompts if req.point_prompts else None,
            req.instance_group,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "ok"}


class RestoreMasksRequest(BaseModel):
    """frame index (string) -> obj_id -> base64 PNG (same as GET masks)."""
    frames: dict[str, dict[str, str]]


@app.post("/api/projects/{pid}/videos/{vid}/masks/restore_frames")
def restore_mask_frames(pid: str, vid: str, req: RestoreMasksRequest):
    if not req.frames:
        return {"status": "ok", "restored": 0}
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    n = 0
    for fidx_str, masks_b64 in req.frames.items():
        try:
            fidx = int(fidx_str)
        except ValueError:
            continue
        dense = decode_masks_png_base64_to_binary(masks_b64)
        if dense:
            ms.save_frame(fidx, dense, None)
        else:
            ms.delete_frame(fidx)
        n += 1
    _invalidate_mask_cache(pid, vid)
    return {"status": "ok", "restored": n}


class ReplaceFramePromptsRequest(BaseModel):
    points: list
    labels: list


@app.put("/api/projects/{pid}/videos/{vid}/objects/{oid}/frames/{frame_idx}/prompts")
def replace_object_frame_prompts(pid: str, vid: str, oid: str, frame_idx: int, req: ReplaceFramePromptsRequest):
    """
    Set or clear point prompts for one object on one frame (config only).
    Use session/rebuild_from_config afterward to refresh SAM + masks.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    prop_state = _get_prop_state(pid, vid)
    if prop_state is not None and prop_state.is_running:
        raise HTTPException(409, "Propagation is running — pause first.")
    if not req.points:
        pm.clear_object_frame_prompt(pid, vid, oid, frame_idx)
        ms = VideoMaskStorage(pm.video_dir(pid, vid))
        if ms.has_masks(frame_idx):
            existing = ms.load_masks_dense(frame_idx)
            remaining = {k: v for k, v in existing.items() if k != oid}
            ms.save_frame(frame_idx, remaining, None)
            _invalidate_mask_cache(pid, vid)
    else:
        pm.save_point_prompts(pid, vid, oid, frame_idx, req.points, req.labels)
    return {"status": "ok"}


# ─── Points / Masks ──────────────────────────────────────────────────────────

class AddPointsRequest(BaseModel):
    frame_idx: int
    points: list            # [[x_norm, y_norm], ...]
    labels: list            # [1, 0, ...]
    text: Optional[str] = None  # override text prompt (defaults to object description)
    anchor_mode: bool = False   # when True: fresh single-frame session, no cross-frame replay


def _replay_prompts(
    pid: str, vid: str, all_prompts: dict, *, collect_last_outputs: bool = False
) -> tuple[list, set, dict[int, dict]]:
    """Replay saved point prompts into the current SAM session (sorted by frame index).

    Returns (items, replayed_obj_ids, last_outputs_by_frame) where last_outputs_by_frame maps
    real frame index -> latest SAM output dict for that frame from the replay loop.

    Object IDs can be integers (e.g., "1") or instance keys (e.g., "1_1", "1_2").
    For SAM's obj_id parameter, we use a unique integer derived from the key.
    """
    frame_map = sam._frame_maps.get((pid, vid), [])
    items: list[tuple[int, str, int, dict]] = []  # (frame_idx, obj_id_str, sam_obj_id, prompt)

    def _to_sam_obj_id(obj_id_str: str) -> int:
        if "_" in obj_id_str:
            parts = obj_id_str.split("_")
            base = int(parts[0])
            instance = int(parts[1])
            return base * 1000 + instance
        else:
            return int(obj_id_str)

    for obj_id_str, fmap in all_prompts.items():
        for fidx_str, prompt in fmap.items():
            rf = int(fidx_str)
            if rf in frame_map:
                sam_obj_id = _to_sam_obj_id(obj_id_str)
                items.append((rf, obj_id_str, sam_obj_id, prompt))
    items.sort(key=lambda x: x[0])

    logger.info(f"=== VISUAL PROMPTS (points) replay: {len(items)} prompts ===")

    replayed_obj_ids: set[str] = set()
    last_outputs_by_frame: dict[int, dict] = {}
    for rf, obj_id_str, sam_oid, prompt in items:
        try:
            points = prompt["points"]
            labels = prompt["labels"]
            logger.info(f"  obj {obj_id_str} (sam_id={sam_oid}), frame {rf}: points={points}, labels={labels}")
            out = sam.add_points(
                pid, vid, frame_idx=rf, obj_id=sam_oid, points=points, labels=labels, text=None
            )
            replayed_obj_ids.add(obj_id_str)
            if collect_last_outputs and out:
                for fk, fo in out.items():
                    if isinstance(fk, int):
                        last_outputs_by_frame[fk] = fo
                    else:
                        try:
                            last_outputs_by_frame[int(fk)] = fo
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            logger.warning(f"Predict replay: obj {obj_id_str} frame {rf}: {e}")

    logger.info(f"  -> Replayed obj IDs with visual prompts: {replayed_obj_ids}")
    return items, replayed_obj_ids, last_outputs_by_frame


def _encode_masks_from_sam_frame_output(
    frame_outputs: dict, objects: dict
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Turn SAM add_points output for one frame into base64 PNGs and raw uint8 masks."""
    mask_b64: dict[str, str] = {}
    raw_masks: dict[str, np.ndarray] = {}
    obj_ids = frame_outputs.get("out_obj_ids", [])
    masks = frame_outputs.get("out_binary_masks", [])
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i] if i < len(masks) else None
        if mask is None:
            continue
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = np.squeeze(mask)
        oid_str = str(obj_id)
        raw_masks[oid_str] = mask
        obj_color = _get_instance_color(oid_str, objects)
        mask_b64[oid_str] = encode_mask_as_png(mask, obj_color)
    return mask_b64, raw_masks


def _persist_merged_masks_for_frame(
    pid: str, vid: str, frame_idx: int, raw_masks: dict[str, np.ndarray]
) -> None:
    if not raw_masks:
        return
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    existing = ms.load_masks_dense(frame_idx)
    merged = {**existing, **{str(k): np.asarray(v) for k, v in raw_masks.items()}}
    ms.save_frame(frame_idx, merged, None)
    _invalidate_mask_cache(pid, vid)


def _contain_frame_masks(objects: dict, frame_masks_int: dict) -> tuple[dict, dict]:
    """Apply sub-object containment to one frame's masks (SAM int-id keyed).

    Walks the object tree top-down: segmentation children are hard-clipped to
    their already-clipped parent; point children are reduced to a fixed blob at
    their tracked centroid sized to the parent bbox; parent-less / missing-parent
    children are dropped.  Returns (str-keyed uint8 masks, str-keyed bboxes
    recomputed from the final masks).  Shared by propagation and single-frame predict.
    """
    str_masks: dict[str, np.ndarray] = {}
    for sam_oid, m in frame_masks_int.items():
        key = _obj_config_key_from_sam_oid(objects, int(sam_oid))
        str_masks[key] = m

    final: dict[str, np.ndarray] = {}
    for oid in iter_objects_topdown(objects):
        if oid not in str_masks:
            continue
        m = hierarchy._as_bool(str_masks[oid])
        meta = objects.get(oid, {})
        parent_id = meta.get("parent_id")
        if parent_id is not None:
            parent_m = final.get(str(parent_id))
            if parent_m is None or not parent_m.any():
                m = np.zeros_like(m, dtype=bool)
            elif meta.get("kind") == "point":
                frac = float(meta.get("point_blob_frac", 0.06) or 0.06)
                c = hierarchy.centroid_norm(m)
                m = (
                    np.zeros_like(m, dtype=bool) if c is None
                    else hierarchy.point_blob_mask(c, parent_m, frac)
                )
            else:
                if parent_m.shape != m.shape:
                    parent_m = hierarchy._resize_bool(parent_m, m.shape)
                m = m & parent_m
        final[oid] = m
    # Defensive: keep masks for ids not present in the object config.
    for oid, m in str_masks.items():
        if oid not in final:
            final[oid] = hierarchy._as_bool(m)

    masks_u8 = {k: v.astype(np.uint8) for k, v in final.items()}
    bboxes: dict[str, list] = {}
    for k, v in final.items():
        if v.any():
            box, score = bbox_norm_xywh_score_from_mask(v)
            bboxes[k] = box + [score]
    return masks_u8, bboxes


def _full_ann_session_reinit_and_replay(
    pid: str, vid: str, ann_dir: str, *, collect_last_outputs: bool
) -> dict[int, dict]:
    """init_session(annotated_frames dir) + replay all point prompts from config."""
    session_id = sam.init_session(pid, vid, ann_dir)
    pm.update_video(pid, vid, {"sam3_session_id": session_id})
    all_prompts = pm.get_all_point_prompts(pid, vid)
    _, _, last_out = _replay_prompts(pid, vid, all_prompts, collect_last_outputs=collect_last_outputs)
    return last_out


def _anchor_session_reinit_and_replay_frame(
    pid: str, vid: str, frame_idx: int, ann_dir_path: Path
) -> dict[int, dict]:
    """Single-frame anchor session: tmp dir with one jpg, replay all prompts on frame_idx."""
    frame_jpg = ann_dir_path / f"{frame_idx:06d}.jpg"
    if not frame_jpg.exists():
        raise HTTPException(400, f"Frame {frame_idx} not yet extracted.")
    tmp_ann = tempfile.mkdtemp(prefix="sam3wt_anchor_ann_")
    try:
        shutil.copy2(str(frame_jpg), str(Path(tmp_ann) / frame_jpg.name))
        session_id = sam.init_session(pid, vid, tmp_ann)
        pm.update_video(pid, vid, {"sam3_session_id": session_id})
    except Exception as e:
        shutil.rmtree(tmp_ann, ignore_errors=True)
        raise HTTPException(500, f"SAM session init failed: {e}")
    finally:
        shutil.rmtree(tmp_ann, ignore_errors=True)

    all_prompts = pm.get_all_point_prompts(pid, vid)
    key = str(frame_idx)
    replay_items: list[tuple[str, dict]] = []
    for obj_id_str, frame_map_prompts in all_prompts.items():
        if key in frame_map_prompts:
            replay_items.append((obj_id_str, frame_map_prompts[key]))
    replay_items.sort(key=lambda x: x[0])

    last_outputs_by_frame: dict[int, dict] = {}
    for obj_id_str, prompt in replay_items:
        sam_oid = _to_sam_obj_id_from_npz_key(obj_id_str)
        try:
            out = sam.add_points(
                pid, vid,
                frame_idx=frame_idx,
                obj_id=sam_oid,
                points=prompt["points"],
                labels=prompt["labels"],
                text=None,
            )
            if out:
                for fk, fo in out.items():
                    if isinstance(fk, int):
                        last_outputs_by_frame[fk] = fo
                    else:
                        try:
                            last_outputs_by_frame[int(fk)] = fo
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            logger.warning(f"anchor replay obj {obj_id_str} frame {frame_idx}: {e}")
    return last_outputs_by_frame


@app.post("/api/projects/{pid}/videos/{vid}/objects/{oid}/points")
def add_points(pid: str, vid: str, oid: str, req: AddPointsRequest):
    """
    Add/update point prompts for an object on a frame.
    Returns per-object mask PNG overlays as base64.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    ann_dir_path = pm.annotated_frames_dir(pid, vid)
    ann_dir = str(ann_dir_path)
    if not ann_dir_path.exists() or not list(ann_dir_path.glob("*.jpg")):
        raise HTTPException(400, "No annotated frames extracted yet. Extract a frame first.")

    # Guard 1: block annotation while propagation is actively running
    prop_state = _get_prop_state(pid, vid)
    if prop_state is not None and prop_state.is_running:
        raise HTTPException(409, "Propagation is running — pause first before adding points.")

    # NOTE: Do not block add_points while anchor remainder inference runs. Remainder holds the same
    # SAM (pid,vid) session during GPU work (waiting_review is False then); blocking here prevented
    # all annotation clicks until review — bad UX and looked like “SAM stopped”. Users may safely
    # annotate during paused review (waiting_review=True); initiating a new init_session during an
    # active remainder step can invalidate that step — acceptable vs silent lock-out.

    # Guard 2: in anchor_mode, only designated anchor frames may be labeled
    if req.anchor_mode:
        start_f = video.get("start_frame", 0)
        num_frames_v = video["num_frames"]
        anchor_frames = compute_anchor_frames(
            start_f, num_frames_v, video_anchor_batch_size(video),
        )
        if req.frame_idx not in anchor_frames:
            raise HTTPException(
                409,
                f"Anchor annotation mode: only anchor frames may be labeled "
                f"(anchor frames: {anchor_frames}). Frame {req.frame_idx} is not an anchor frame."
            )

    if req.anchor_mode:
        # Anchor annotation: use a fresh single-frame session so inference is as
        # fast as the very first annotation.  SAM3 loads frames eagerly during
        # init_state (async_loading_frames=False), so the tmp dir can be deleted
        # immediately after init_session returns.
        #
        # IMPORTANT: only re-init when the frame changes.  Multiple objects are
        # annotated on the same anchor frame — re-initing on each click wipes
        # previously-added objects from the SAM inference state so only the last
        # object's masks are returned.  Reuse the existing session when it is
        # already a single-frame session on this exact frame.
        frame_jpg = ann_dir_path / f"{req.frame_idx:06d}.jpg"
        if not frame_jpg.exists():
            raise HTTPException(400, f"Frame {req.frame_idx} not yet extracted.")
        current_map = sam._frame_maps.get((pid, vid), [])
        need_init = not (len(current_map) == 1 and req.frame_idx in current_map)
        if need_init:
            tmp_ann = tempfile.mkdtemp(prefix="sam3wt_anchor_ann_")
            try:
                shutil.copy2(str(frame_jpg), str(Path(tmp_ann) / frame_jpg.name))
                session_id = sam.init_session(pid, vid, tmp_ann)
                pm.update_video(pid, vid, {"sam3_session_id": session_id})
            except Exception as e:
                raise HTTPException(500, f"SAM session init failed: {e}")
            finally:
                shutil.rmtree(tmp_ann, ignore_errors=True)
            # Replay any already-saved prompts for OTHER objects on this same frame so
            # that the fresh single-frame session has full multi-object context.
            # (The current object's prompt is applied by the add_points call below.)
            same_frame_prompts = pm.get_all_point_prompts(pid, vid)
            for other_oid_str, other_frame_prompts in same_frame_prompts.items():
                if other_oid_str == str(oid):
                    continue  # handled by main add_points call below
                key = str(req.frame_idx)
                if key in other_frame_prompts:
                    other_obj_id_int = _to_sam_obj_id_from_npz_key(other_oid_str)
                    try:
                        sam.add_points(
                            pid, vid,
                            frame_idx=req.frame_idx,
                            obj_id=other_obj_id_int,
                            points=other_frame_prompts[key]["points"],
                            labels=other_frame_prompts[key]["labels"],
                            text=None,
                        )
                    except Exception as e:
                        logger.warning(
                            f"anchor reinit replay obj {other_oid_str} frame {req.frame_idx}: {e}"
                        )
    else:
        # Normal annotation: ensure session contains the requested frame, then
        # replay all saved prompts so SAM has the full cross-frame object context.
        session_id = sam.get_session_id(pid, vid)
        need_reinit = session_id is None

        if not need_reinit:
            frame_map = sam._frame_maps.get((pid, vid), [])
            if req.frame_idx not in frame_map:
                need_reinit = True
                logger.info(
                    f"Frame {req.frame_idx} not in current session "
                    f"(loaded: {frame_map}), re-initializing..."
                )

        if need_reinit:
            try:
                _full_ann_session_reinit_and_replay(
                    pid, vid, ann_dir, collect_last_outputs=False
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"SAM session init failed: {e}")

    # Save prompts to config (always the user's raw points — synthetic negative seeds
    # for segmentation children are regenerated at inference time, never persisted).
    pm.save_point_prompts(pid, vid, oid, req.frame_idx, req.points, req.labels)

    objects = video["objects"]
    obj_meta = objects.get(str(oid), {})
    parent_id = obj_meta.get("parent_id")
    kind = obj_meta.get("kind", "segmentation")
    blob_frac = float(obj_meta.get("point_blob_frac", 0.06) or 0.06)

    # Parent mask for this frame (used to contain children / size point blobs).
    parent_mask = None
    if parent_id is not None:
        try:
            parent_mask = VideoMaskStorage(pm.video_dir(pid, vid)).load_masks_dense(
                req.frame_idx
            ).get(str(parent_id))
        except Exception as e:
            logger.warning(f"add_points: could not load parent {parent_id} mask: {e}")

    # ── point-kind sub-object: no SAM call — emit a fixed blob at the clicked point ──
    if kind == "point" and parent_id is not None:
        # Center on the last positive point; fall back to the first point given.
        center = None
        for pt, lab in zip(req.points, req.labels):
            if lab == 1:
                center = pt
        if center is None and req.points:
            center = req.points[0]
        if center is None:
            return {"frame_idx": req.frame_idx, "masks": {}}
        if parent_mask is None:
            blob = np.zeros((video["height"], video["width"]), dtype=np.uint8)
        else:
            blob = hierarchy.point_blob_mask(center, parent_mask, blob_frac).astype(np.uint8)
        raw_masks = {str(oid): blob}
        _persist_merged_masks_for_frame(pid, vid, req.frame_idx, raw_masks)
        mask_b64 = {
            str(oid): encode_mask_as_png(blob, _get_instance_color(str(oid), objects))
        }
        return {"frame_idx": req.frame_idx, "masks": mask_b64}

    # ── segmentation sub-object: augment with negative seeds outside the parent ──
    points = list(req.points)
    labels = list(req.labels)
    if kind == "segmentation" and parent_mask is not None:
        neg = hierarchy.negative_seed_points(parent_mask)
        if neg:
            points = points + neg
            labels = labels + [0] * len(neg)

    # Standard single-instance point prompt
    try:
        sam_oid = _to_sam_obj_id_from_npz_key(str(oid))
        logger.info(f"add_points: frame={req.frame_idx}, obj={oid} (sam_oid={sam_oid}), points={req.points}, labels={req.labels}")
        outputs = sam.add_points(
            pid, vid,
            frame_idx=req.frame_idx,
            obj_id=sam_oid,
            points=points,
            labels=labels,
            text=None,  # SAM3 tracker mode doesn't support text with points
        )
    except Exception as e:
        import traceback
        logger.error(f"SAM add_points error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"SAM inference error: {e}")

    frame_outputs = outputs.get(req.frame_idx, outputs.get(str(req.frame_idx), {}))
    if not frame_outputs and outputs:
        first_key = next(iter(outputs))
        frame_outputs = outputs[first_key]

    mask_b64, raw_masks = _encode_masks_from_sam_frame_output(frame_outputs, objects)

    # Hard-clip this child to its parent (strict containment), then re-encode it.
    if parent_id is not None and parent_mask is not None and str(oid) in raw_masks:
        clipped = hierarchy._as_bool(raw_masks[str(oid)]) & hierarchy._as_bool(parent_mask)
        raw_masks[str(oid)] = clipped.astype(np.uint8)
        mask_b64[str(oid)] = encode_mask_as_png(
            raw_masks[str(oid)], _get_instance_color(str(oid), objects)
        )
    elif parent_id is not None and str(oid) in raw_masks:
        # Parent has no mask on this frame → drop the child (strict hierarchy).
        raw_masks[str(oid)] = np.zeros_like(hierarchy._as_bool(raw_masks[str(oid)]), dtype=np.uint8)
        mask_b64.pop(str(oid), None)

    _persist_merged_masks_for_frame(pid, vid, req.frame_idx, raw_masks)

    return {"frame_idx": req.frame_idx, "masks": mask_b64}


@app.post("/api/projects/{pid}/videos/{vid}/objects/{oid}/clear")
def clear_object_points(pid: str, vid: str, oid: str):
    """Clear all point prompts for an object."""
    pm.clear_point_prompts(pid, vid, oid)
    try:
        sam.clear_object_prompts(pid, vid, int(oid))
    except Exception as e:
        logger.warning(f"SAM clear error: {e}")
    return {"status": "ok"}


@app.delete("/api/projects/{pid}/videos/{vid}/objects/{oid}/frames/{frame_idx}/points")
def clear_object_frame_points(pid: str, vid: str, oid: str, frame_idx: int):
    """Clear point prompts and saved mask for a single object on a single frame."""
    pm.clear_object_frame_prompt(pid, vid, oid, frame_idx)
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    if ms.has_masks(frame_idx):
        existing = ms.load_masks_dense(frame_idx)
        remaining = {k: v for k, v in existing.items() if k != oid}
        ms.save_frame(frame_idx, remaining, None)
        _invalidate_mask_cache(pid, vid)
    return {"status": "ok"}


class SwapMasksRequest(BaseModel):
    obj_a: str
    obj_b: str
    from_frame: int = -1  # -1 = no lower bound
    to_frame: int = -1    # -1 = no upper bound


@app.post("/api/projects/{pid}/videos/{vid}/masks/swap")
def swap_object_masks(pid: str, vid: str, req: SwapMasksRequest):
    """Swap masks (and bboxes) between two objects across a range of frames."""
    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    indices = ms.iter_frame_indices_in_range(req.from_frame, req.to_frame)
    swapped = 0
    for fidx in indices:
        masks = ms.load_masks_dense(fidx)
        has_a = req.obj_a in masks
        has_b = req.obj_b in masks
        if not has_a and not has_b:
            continue

        a_mask = masks.pop(req.obj_a, None)
        b_mask = masks.pop(req.obj_b, None)
        if b_mask is not None:
            masks[req.obj_a] = b_mask
        if a_mask is not None:
            masks[req.obj_b] = a_mask

        bboxes = ms.load_bboxes(fidx)
        a_bbox = bboxes.pop(req.obj_a, None)
        b_bbox = bboxes.pop(req.obj_b, None)
        if b_bbox is not None:
            bboxes[req.obj_a] = b_bbox
        if a_bbox is not None:
            bboxes[req.obj_b] = a_bbox

        ms.save_frame(fidx, masks, bboxes)
        swapped += 1

    _invalidate_mask_cache(pid, vid)
    return {"status": "ok", "frames_swapped": swapped}


# ─── Saved Masks (post-propagation) ──────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/masks/{fidx}")
def get_saved_mask(pid: str, vid: str, fidx: int):
    """
    Return a composite mask overlay PNG for a frame (after propagation).
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    if not ms.has_masks(fidx):
        return JSONResponse({"masks": {}}, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    mask_b64 = _get_encoded_masks(pid, vid, fidx, video["objects"])
    return JSONResponse(
        {"frame_idx": fidx, "masks": mask_b64},
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def _delete_masks_in_range(
    pid: str, vid: str,
    from_frame: Optional[int] = None,
    to_frame: Optional[int] = None,
) -> int:
    """
    Delete mask/bbox rows and legacy files in [from_frame, to_frame]
    (both inclusive, None = unbounded).
    Returns count of distinct frame indices cleared.
    """
    return VideoMaskStorage(pm.video_dir(pid, vid)).delete_masks_range(from_frame, to_frame)


@app.delete("/api/projects/{pid}/videos/{vid}/masks/{fidx}")
def delete_frame_masks(pid: str, vid: str, fidx: int):
    """Delete saved masks for a single frame (.npz and .json bbox files)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
    bboxes_path = pm.bboxes_dir(pid, vid) / f"{fidx:06d}.json"

    deleted = []
    had_masks = ms.has_masks(fidx) or masks_path.exists()
    had_bbox = bboxes_path.exists()
    ms.delete_frame(fidx)
    if had_masks:
        deleted.append("masks")
    if had_bbox:
        deleted.append("bboxes")

    _invalidate_mask_cache(pid, vid)
    return JSONResponse({"status": "ok", "frame_idx": fidx, "deleted": deleted})


@app.delete("/api/projects/{pid}/videos/{vid}/masks")
def bulk_delete_masks(
    pid: str, vid: str,
    mode: str = "all",
    from_frame: Optional[int] = None,
    to_frame: Optional[int] = None,
):
    """
    Bulk-delete propagation mask files.

    mode:
      all        – delete all mask frames
      from_frame – delete from from_frame to end
      to_frame   – delete from start to to_frame
      range      – delete from from_frame to to_frame (inclusive)
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    if mode == "all":
        deleted = _delete_masks_in_range(pid, vid)
    elif mode == "from_frame":
        if from_frame is None:
            raise HTTPException(422, "from_frame required for mode=from_frame")
        deleted = _delete_masks_in_range(pid, vid, from_frame=from_frame)
    elif mode == "to_frame":
        if to_frame is None:
            raise HTTPException(422, "to_frame required for mode=to_frame")
        deleted = _delete_masks_in_range(pid, vid, to_frame=to_frame)
    elif mode == "range":
        if from_frame is None or to_frame is None:
            raise HTTPException(422, "from_frame and to_frame required for mode=range")
        deleted = _delete_masks_in_range(pid, vid, from_frame=from_frame, to_frame=to_frame)
    else:
        raise HTTPException(422, f"Unknown mode: {mode}")

    _invalidate_mask_cache(pid, vid)
    return JSONResponse({"status": "ok", "mode": mode, "deleted_frames": deleted})


@app.get("/api/projects/{pid}/videos/{vid}/masks/{fidx}/composite")
def get_composite_mask(pid: str, vid: str, fidx: int):
    """Return a single composite RGBA PNG for the frame."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    ms = VideoMaskStorage(pm.video_dir(pid, vid))
    if not ms.has_masks(fidx):
        raise HTTPException(404, "No mask for this frame")

    raw_masks = ms.load_masks_dense(fidx)
    objects = video["objects"]
    colors = {obj_id: obj.get("color", "#5B8DD9") for obj_id, obj in objects.items()}

    b64 = composite_masks_as_png(raw_masks, colors, video["width"], video["height"])
    import base64, io
    png_bytes = base64.b64decode(b64)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.delete("/api/projects/{pid}/videos/{vid}/frames/{frame_idx}/prompts", status_code=200)
def clear_frame_from_inference(pid: str, vid: str, frame_idx: int):
    """
    Remove all point prompts for a specific frame from all objects.
    This effectively removes the frame from the inference state / replay chain.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    pm.clear_frame_prompts(pid, vid, frame_idx)
    return {"status": "cleared", "frame_idx": frame_idx}


# ─── Video Export (SSE) ──────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/export")
async def export_video_sse(pid: str, vid: str):
    """
    Stream export progress as Server-Sent Events while writing an annotated
    MP4 to disk.  Reads only the source video and persisted masks for this video
    (`masks.sqlite` and/or legacy masks/*.npz) — does not touch SAM sessions, annotated_frames, or inference state.

    Progress events: {"frame": N, "total": T, "progress": 0.0-1.0}
    Done event:      {"path": "/abs/path/export.mp4", "total_frames": N}
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(400, "Video source file not found")

    objects = video["objects"]
    colors = {oid: obj.get("color", "#5B8DD9") for oid, obj in objects.items()}
    labels_map = {oid: obj.get("name", f"Object {oid}") for oid, obj in objects.items()}
    video_dir_export = str(pm.video_dir(pid, vid))
    out_path = pm.video_dir(pid, vid) / "export.mp4"

    async def event_gen():
        import concurrent.futures
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=300)

        def _progress(frame_idx: int, total: int):
            asyncio.run_coroutine_threadsafe(
                queue.put(("progress", {
                    "frame": frame_idx,
                    "total": total,
                    "progress": round(min(frame_idx / max(total, 1), 1.0), 4),
                })),
                loop,
            ).result()

        def _run():
            try:
                result = export_video_with_masks(
                    source_path=source_path,
                    out_path=str(out_path),
                    video_dir=video_dir_export,
                    colors=colors,
                    labels=labels_map,
                    num_frames_hint=video.get("num_frames", 0),
                    progress_callback=_progress,
                )
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", {
                        "path": str(out_path),
                        "total_frames": result["total_frames"],
                    })),
                    loop,
                ).result()
            except Exception as exc:
                logger.error(f"Video export failed for {pid}/{vid}: {exc}", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", {"error": str(exc)})),
                    loop,
                ).result()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, _run)

        while True:
            msg_type, payload = await queue.get()
            yield {"event": msg_type, "data": json.dumps(payload)}
            if msg_type in ("done", "error"):
                break

    return EventSourceResponse(event_gen())


# ─── Anchor frame helpers ─────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/anchor_frames")
def get_anchor_frames(pid: str, vid: str):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    start = video.get("start_frame", 0)
    num_frames = video["num_frames"]
    batch_sz = video_anchor_batch_size(video)
    anchors = compute_anchor_frames(start, num_frames, batch_sz)
    return {
        "anchor_frames": anchors,
        "count": len(anchors),
        "anchor_batch_size": batch_sz,
        "manual_anchor_prefix_before_infer": ANCHOR_MANUAL_PREFIX_COUNT,
    }


class AnchorLabelingTiming(BaseModel):
    """Client wall-clock timestamps (ms since epoch); used for anchor labeling analytics."""

    entered_ms: int    # User landed on this anchor frame to label it (after finishing previous anchor)
    finished_ms: int   # User clicked Done / next frame on this anchor


class CommitAnchorRequest(BaseModel):
    anchor_index: int  # 0-based index in the anchor frames list
    labeling_timing: Optional[AnchorLabelingTiming] = None


@app.post("/api/projects/{pid}/videos/{vid}/anchors/{frame_idx}/commit")
def commit_anchor_frame(pid: str, vid: str, frame_idx: int, req: CommitAnchorRequest):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    annotated = list(video.get("annotated_anchors", []))
    if frame_idx not in annotated:
        annotated = sorted(annotated + [frame_idx])
    updates: dict = {"annotated_anchors": annotated}

    if req.labeling_timing is not None:
        lt = req.labeling_timing
        dur = max(0, lt.finished_ms - lt.entered_ms)
        fk = str(int(frame_idx))
        alt = dict(video.get("anchor_labeling_timing") or {})
        frames = dict(alt.get("frames") or {})
        frames[fk] = {
            "entered_frontend_ms": lt.entered_ms,
            "committed_ms": lt.finished_ms,
            "duration_ms": dur,
        }
        vinfo = dict(alt.get("video") or {})
        first = vinfo.get("first_anchor_entered_ms")
        if first is None or lt.entered_ms < first:
            vinfo["first_anchor_entered_ms"] = lt.entered_ms
        vinfo["last_anchor_committed_ms"] = lt.finished_ms
        fe = vinfo.get("first_anchor_entered_ms")
        if fe is not None:
            vinfo["whole_video_labeling_wall_ms"] = lt.finished_ms - fe
        vinfo["sum_anchor_durations_ms"] = sum(int(f.get("duration_ms", 0) or 0) for f in frames.values())

        alt["frames"] = frames
        alt["video"] = vinfo
        updates["anchor_labeling_timing"] = alt

    start = int(video.get("start_frame") or 0)
    num_frames_v = int(video["num_frames"])
    req_anchors = compute_anchor_frames(start, num_frames_v, video_anchor_batch_size(video))
    ann_set = set(annotated)
    updates["anchor_labeling_complete"] = bool(req_anchors) and all(a in ann_set for a in req_anchors)

    pm.update_video(pid, vid, updates)
    return {"status": "ok", "committed_frame": frame_idx, "anchor_index": req.anchor_index}


@app.get("/api/projects/{pid}/videos/{vid}/anchors/predict_remainder_sse")
async def anchor_predict_remainder_sse(
    pid: str,
    vid: str,
    interactive: bool = Query(
        True,
        description="If true, pause after each predicted anchor until POST …/predict_remainder_continue.",
    ),
):
    """Queue-style inference: sequentially propagate masks to unset anchor frames after manual prefix."""

    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    prop_s = _get_prop_state(pid, vid)
    if prop_s.is_running:
        raise HTTPException(
            409,
            "Whole-video propagation is running — pause or finish it before predicting anchor remainders.",
        )

    rm = _get_anchor_remainder_state(pid, vid)
    q = rm.subscribe()

    if not rm.is_running:
        rm.is_running = True
        rm.task = asyncio.create_task(_run_anchor_remainder_inference_bg(pid, vid, rm, interactive))
    else:
        await q.put({
            "event": "catch_up",
            "data": json.dumps({"message": "anchor remainder prediction already in progress"}),
        })

    async def event_gen():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
                    continue
                yield event
                if event.get("event") in ("done", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            rm.unsubscribe(q)

    return EventSourceResponse(event_gen())


@app.post("/api/projects/{pid}/videos/{vid}/anchors/predict_remainder_continue")
async def anchor_predict_remainder_continue(pid: str, vid: str):
    """Acknowledge review of the latest predicted anchor and resume sequential inference."""
    rm = _get_anchor_remainder_state(pid, vid)
    if not rm.waiting_review or rm.review_continue_event is None:
        raise HTTPException(
            status_code=409,
            detail="Anchor remainder inference is not paused for review.",
        )
    rm.review_continue_event.set()
    return {"status": "ok"}


# ─── Propagation background task ─────────────────────────────────────────────

async def _run_propagation_bg(
    pid: str,
    vid: str,
    start_frame: int,
    source_path: str,
    num_frames: int,
    video_dir: Path,
    objects: dict,
    all_prompts: dict,
    state: PropagationState,
    end_frame: int = -1,
    use_all_anchors: bool = False,
    ds_max_dim: Optional[int] = None,
    ds_scale_factor: Optional[float] = None,
    batch_size: int = STREAM_BATCH_SIZE,
) -> None:
    """
    Forward-only batch propagation.

    Processes frames start_frame..num_frames-1 in batch_size chunks.
    Each batch [P1, P2]:
      - Seeds P1 from the previous frame's saved mask (cross-batch continuity /
        resume continuity), when P1 > 0 and masks exist for frame P1-1.
      - Seeds every user-labeled frame within [P1, P2] from their point prompts
        (sorted ascending so SAM sees them in order).
      - When use_all_anchors=True, also extracts and seeds all user-labeled frames
        outside [P1, P2] into the SAM session as global context.  These context
        frames don't influence forward_start — propagation still begins from the
        earliest in-batch seed — but SAM's memory attends to them.
      - Runs forward propagation from the earliest seeded frame to P2.
    Batches with no seeds (no labeled frames, no cross-batch mask) are skipped.
    """
    loop = asyncio.get_event_loop()

    # All frame indices that have user-labeled point prompts (used for all-anchors mode).
    all_labeled_frames: list[int] = sorted({
        int(fidx_str)
        for frame_prompts in all_prompts.values()
        for fidx_str in frame_prompts.keys()
    })

    # SAM object ids of "point" sub-objects (seeded from tiny blob masks; their
    # binarization threshold is lowered on the SAM2 path to resist collapse).
    point_sam_ids: set[int] = {
        _to_sam_obj_id_from_npz_key(oid)
        for oid, meta in objects.items()
        if meta.get("kind") == "point"
    }

    # Compute simple sequential batches
    effective_end = end_frame if (end_frame >= 0 and end_frame < num_frames) else num_frames - 1
    batches: list[tuple[int, int]] = []
    p = start_frame
    while p <= effective_end:
        batches.append((p, min(p + batch_size - 1, effective_end)))
        p += batch_size

    total_batches = len(batches)
    frames_to_process = effective_end - start_frame + 1
    total_propagated = 0

    async def publish(event_name: str, data: dict):
        await state.publish({"event": event_name, "data": json.dumps(data)})

    await publish("init", {
        "actual_start": start_frame,
        "frames_to_process": frames_to_process,
        "total_batches": total_batches,
    })

    def _seed_from_saved_previous(seed_at_frame: int) -> None:
        prev = seed_at_frame - 1
        if prev < 0:
            return
        ms_seed = VideoMaskStorage(video_dir)
        if not ms_seed.has_masks(prev):
            return
        try:
            saved = ms_seed.load_masks_dense(prev)
            for obj_id_str, mask in saved.items():
                obj_id_int = int(obj_id_str.split("_")[0]) if "_" in obj_id_str else int(obj_id_str)
                try:
                    sam.add_mask_prompt(pid, vid, frame_idx=seed_at_frame, obj_id=obj_id_int, mask=mask)
                except Exception as e:
                    logger.warning(f"add_mask_prompt obj {obj_id_str} frame {seed_at_frame}: {e}")
        except Exception as e:
            logger.warning(f"_seed_from_saved_previous prev={prev} seed_at={seed_at_frame}: {e}")

    def _extract_pass_results(item: dict) -> tuple[int, dict[int, np.ndarray], dict[int, list]]:
        """Extract real frame index, masks and bboxes from a propagation item.

        Handles two output formats:
          - SAM3 flat: outputs = {out_obj_ids, out_binary_masks, out_boxes_xywh, out_probs, ...}
          - SAM2 nested (our wrapper): outputs = {frame_idx: {out_obj_ids, ...}}
        """
        sam_idx = item.get("frame_index", 0)
        real_frame = sam.to_real_idx(pid, vid, sam_idx)
        outputs = item.get("outputs", {})
        frame_masks: dict[int, np.ndarray] = {}
        frame_bboxes: dict[int, list] = {}

        # Detect format: SAM3 puts out_obj_ids directly in outputs (flat),
        # SAM2 wrapper nests it under a frame-index key.
        if "out_obj_ids" in outputs:
            # SAM3 flat format
            obj_ids = outputs.get("out_obj_ids", [])
            masks_list = outputs.get("out_binary_masks", [])
            boxes_list = outputs.get("out_boxes_xywh", [])
            probs_list = outputs.get("out_probs", [])
        else:
            # SAM2 nested format — use the first (only) nested entry
            obj_ids, masks_list, boxes_list, probs_list = [], [], [], []
            for out_val in outputs.values():
                if isinstance(out_val, dict):
                    obj_ids = out_val.get("out_obj_ids", [])
                    masks_list = out_val.get("out_binary_masks", [])
                    boxes_list = out_val.get("out_boxes_xywh", [])
                    probs_list = out_val.get("out_probs", [])
                    break

        for k, oid in enumerate(obj_ids):
            if k < len(masks_list):
                m = masks_list[k]
                if hasattr(m, 'numpy'):
                    m = m.numpy()
                frame_masks[int(oid)] = np.squeeze(m).astype(np.uint8)
            if k < len(boxes_list):
                b = boxes_list[k]
                if hasattr(b, 'numpy'):
                    b = b.numpy()
                if hasattr(b, 'tolist'):
                    b = b.tolist()
                score = float(probs_list[k]) if k < len(probs_list) else 0.0
                frame_bboxes[int(oid)] = (b + [score]) if b else []
        return real_frame, frame_masks, frame_bboxes

    def _contain_propagation_frame(frame_masks_int: dict) -> tuple[dict, dict]:
        """Apply sub-object containment to one propagated frame (see _contain_frame_masks)."""
        return _contain_frame_masks(objects, frame_masks_int)

    try:
        for batch_idx, (P1, P2) in enumerate(batches):
            if state.is_paused:
                break

            await publish("batch_start", {
                "batch": batch_idx,
                "batch_start": P1,
                "batch_end": P2 + 1,
                "total_batches": total_batches,
                "status": "extracting",
            })

            tmp_dir = tempfile.mkdtemp(prefix=f"sam3wt_{pid}_{vid}_b{batch_idx}_")
            try:
                _extract_state = {"extracted": 0}

                def _progress_cb(extracted: int, _total: int) -> None:
                    _extract_state["extracted"] = extracted

                future = loop.run_in_executor(
                    None,
                    lambda: extract_frame_range(
                        source_path, tmp_dir, P1, P2 + 1, _progress_cb,
                        max_dim=ds_max_dim, scale_factor=ds_scale_factor,
                    )
                )
                _last_reported = -1
                while not future.done():
                    await asyncio.sleep(0.25)
                    curr = _extract_state["extracted"]
                    if curr != _last_reported:
                        _last_reported = curr
                        await publish("extract_progress", {
                            "extracted": curr,
                            "batch_start": P1,
                            "batch_end": P2 + 1,
                            "frames_to_process": frames_to_process,
                        })
                await future

                # All-anchors mode: extract user-labeled frames outside [P1, P2] so
                # SAM can attend to them as global context during this batch.
                if use_all_anchors:
                    extra_frames = [f for f in all_labeled_frames if f < P1 or f > P2]
                    for af in extra_frames:
                        af_out = Path(tmp_dir) / f"{af:06d}.jpg"
                        if not af_out.exists():
                            await loop.run_in_executor(
                                None,
                                lambda af=af: extract_frame_range(
                                    source_path, tmp_dir, af, af + 1,
                                    max_dim=ds_max_dim, scale_factor=ds_scale_factor,
                                ),
                            )

                await publish("batch_start", {
                    "batch": batch_idx,
                    "batch_start": P1,
                    "batch_end": P2 + 1,
                    "total_batches": total_batches,
                    "status": "initializing_session",
                })

                await loop.run_in_executor(None, sam.init_session, pid, vid, tmp_dir)
                # init_session clears any prior registration, so re-register each batch.
                if point_sam_ids:
                    sam.register_point_objects(pid, vid, point_sam_ids)

                seeded_frames: list[int] = []

                # Cross-batch / resume continuity: seed P1 from frame P1-1's saved mask.
                if P1 > 0:
                    ms_chk = VideoMaskStorage(video_dir)
                    if ms_chk.has_masks(P1 - 1):
                        await loop.run_in_executor(None, _seed_from_saved_previous, P1)
                        seeded_frames.append(P1)

                # Seed every user-labeled frame that is in the current session.
                # Returns (in_batch, context_only): frames in [P1,P2] drive forward_start;
                # context frames outside the batch give SAM global memory without
                # pushing the propagation start point backwards.
                def _seed_labeled_frames() -> tuple[list[int], list[int]]:
                    frame_map = sam._frame_maps.get((pid, vid), [])
                    ms_seed = VideoMaskStorage(video_dir)
                    items: list[tuple[int, int, str, dict]] = []
                    for obj_id_str, frame_prompts in all_prompts.items():
                        obj_id_int = (
                            int(obj_id_str.split("_")[0]) if "_" in obj_id_str
                            else int(obj_id_str)
                        )
                        for fidx_str, prompt in frame_prompts.items():
                            fidx = int(fidx_str)
                            if fidx in frame_map:
                                items.append((fidx, obj_id_int, obj_id_str, prompt))
                    items.sort(key=lambda x: x[0])
                    in_batch: list[int] = []
                    context_only: list[int] = []
                    for fidx, obj_id_int, obj_id_str, prompt in items:
                        try:
                            meta = objects.get(obj_id_str, {})
                            seed_blob = None
                            if meta.get("kind") == "point":
                                # Seed point sub-objects from their persisted tiny blob
                                # mask (the "tiny mask" anchor) rather than a bare point.
                                saved = ms_seed.load_masks_dense(fidx)
                                cand = saved.get(obj_id_str)
                                if cand is not None and np.asarray(cand).any():
                                    seed_blob = np.asarray(cand)
                            if seed_blob is not None:
                                sam.add_mask_prompt(
                                    pid, vid, frame_idx=fidx, obj_id=obj_id_int, mask=seed_blob,
                                )
                            else:
                                sam.add_points(
                                    pid, vid, frame_idx=fidx, obj_id=obj_id_int,
                                    points=prompt["points"], labels=prompt["labels"],
                                )
                            if P1 <= fidx <= P2:
                                if fidx not in in_batch:
                                    in_batch.append(fidx)
                            else:
                                if fidx not in context_only:
                                    context_only.append(fidx)
                        except Exception as e:
                            logger.warning(
                                f"Seed labeled frame: obj {obj_id_int} frame {fidx}: {e}"
                            )
                    return in_batch, context_only

                labeled_in_batch, labeled_context = await loop.run_in_executor(None, _seed_labeled_frames)
                if labeled_context:
                    logger.info(
                        f"Batch {batch_idx} [{P1},{P2}]: seeded {len(labeled_context)} "
                        f"context frame(s) outside batch: {labeled_context}"
                    )
                seeded_frames.extend(labeled_in_batch)

                if not seeded_frames:
                    logger.info(f"Batch {batch_idx} [{P1}, {P2}]: no seeds, skipping")
                    continue

                forward_start = min(seeded_frames)

                def _forward_pass():
                    results = []
                    for item in sam.propagate_stream(
                        pid, vid,
                        start_frame_idx=forward_start,
                        propagation_direction="forward",
                        max_frame_num_to_track=P2 - forward_start,
                    ):
                        results.append(_extract_pass_results(item))
                    return results

                forward_results = await loop.run_in_executor(None, _forward_pass)

                # Last write wins when a frame appears in both cross-batch seed and
                # labeled-frame prompts (forward pass runs once so this is just dedup).
                frame_results: dict[int, tuple] = {}
                for real_frame, frame_masks, frame_bboxes in forward_results:
                    frame_results[real_frame] = (frame_masks, frame_bboxes)

                def _persist_propagation_frame(rf: int, fm: dict, fb: dict):
                    VideoMaskStorage(video_dir).save_frame(
                        rf,
                        {str(k): v for k, v in fm.items()},
                        {str(k): v for k, v in fb.items()} if fb else None,
                    )

                for real_frame, (frame_masks, frame_bboxes) in sorted(frame_results.items()):
                    if frame_masks:
                        # Enforce sub-object containment (clip children to parents,
                        # reshape point children to blobs) before persisting.
                        masks_str, bboxes_str = _contain_propagation_frame(frame_masks)
                        await loop.run_in_executor(
                            None,
                            _persist_propagation_frame,
                            real_frame,
                            masks_str,
                            bboxes_str,
                        )
                    total_propagated += 1
                    progress = (start_frame + total_propagated) / num_frames
                    await publish("progress", {"frame": real_frame, "progress": min(progress, 1.0)})
                    pm.mark_frame_propagated(pid, vid, real_frame)
                    _invalidate_mask_cache(pid, vid)

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        pm.mark_propagation_complete(pid, vid)
        state.is_running = False
        propagated = pm.get_video(pid, vid).get("propagated_frames", [])
        last_frame = max(propagated) if propagated else -1
        await publish("done", {"frame": last_frame})

    except asyncio.CancelledError:
        state.is_running = False
        state.is_paused = True
        raise
    except Exception as e:
        logger.error(f"Propagation error: {e}", exc_info=True)
        state.is_running = False
        try:
            pm.set_video_inference_status(pid, vid, "failed", socket.gethostname())
        except Exception as e2:
            logger.warning(f"Could not persist inference failed status: {e2}")
        await publish("error", {"error": str(e)})
    finally:
        # Each batch calls init_session(pid, vid, …), which closes the previous batch's
        # session for this (pid, vid) only. Nothing closed the final batch's session, and
        # init_session for another video closes only that video's key — leaving every
        # finished video's last SAM session (and GPU state) pinned until restart.
        def _cleanup_sam_propagation_session():
            try:
                sam.close_session(pid, vid)
            except Exception as ce:
                logger.warning(f"close_session after propagate {pid}/{vid}: {ce}")

        try:
            await loop.run_in_executor(None, _cleanup_sam_propagation_session)
        except Exception as ce:
            logger.warning(f"Propagator SAM cleanup executor failed ({pid}/{vid}): {ce}")


# ─── Propagation SSE endpoint ─────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/propagate")
async def propagate_video(pid: str, vid: str, start_frame: int = 0, resume_from: int = -1, end_frame: int = -1, use_all_anchors: bool = False):
    """
    Stream propagation results as Server-Sent Events.

    Propagation runs as a background task so client reconnects join the
    existing run rather than restarting it from scratch.

    resume_from: if >= 0, treat this as a resume from the given frame.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    rm_anchor = _get_anchor_remainder_state(pid, vid)
    if rm_anchor.is_running:
        raise HTTPException(
            409,
            "Anchor remainder prediction is running — wait for it before whole-video propagation.",
        )

    state = _get_prop_state(pid, vid)
    q = state.subscribe()

    logger.info(
        f"GET /propagate: start_frame={start_frame}, resume_from={resume_from}, "
        f"is_running={state.is_running}"
    )

    if not state.is_running:
        all_prompts = pm.get_all_point_prompts(pid, vid)
        if not all_prompts:
            state.unsubscribe(q)
            logger.error(f"GET /propagate: no point prompts for {pid}/{vid}.")
            raise HTTPException(400, "No point prompts found. Annotate anchor frames first.")

        num_frames = video["num_frames"]
        source_path = video.get("source_path", "")
        video_dir_path = pm.video_dir(pid, vid)
        objects = video["objects"]

        # Determine start: for fresh runs use earliest annotated frame,
        # for resumes use the frame after the paused point.
        if resume_from >= 0:
            actual_start = resume_from + 1
        else:
            effective_end_q = end_frame if end_frame >= 0 else num_frames - 1
            frame_idxs = [
                int(fidx)
                for fp in all_prompts.values()
                for fidx in fp.keys()
                if start_frame <= int(fidx) <= effective_end_q
            ]
            actual_start = min(frame_idxs) if frame_idxs else start_frame

        state.total_frames = num_frames
        state.start_frame = actual_start
        state.user_start_frame = start_frame
        state.is_running = True
        state.is_paused = False
        try:
            pm.set_video_inference_status(pid, vid, "running", socket.gethostname())
        except Exception as e:
            logger.warning(f"Persist inference running status: {e}")
        ds_max_dim, ds_scale_factor = _video_ds_params(video)
        batch_sz = video_anchor_batch_size(video)
        state.task = asyncio.create_task(
            _run_propagation_bg(
                pid, vid, actual_start, source_path, num_frames,
                video_dir_path, objects, all_prompts, state,
                end_frame=end_frame,
                use_all_anchors=use_all_anchors,
                ds_max_dim=ds_max_dim,
                ds_scale_factor=ds_scale_factor,
                batch_size=batch_sz,
            )
        )
    else:
        # Propagation already running — send a catch_up event so the newly
        # connected client knows where we are without waiting for the next frame.
        video_fresh = pm.get_video(pid, vid)
        propagated = video_fresh.get("propagated_frames", []) if video_fresh else []
        await q.put({
            "event": "catch_up",
            "data": json.dumps({
                "frames_done": len(propagated),
                "total_frames": state.total_frames,
                "last_frame": max(propagated) if propagated else -1,
                "start_frame": state.user_start_frame,
            }),
        })

    async def event_gen():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
                    continue
                yield event
                if event.get("event") in ("done", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            state.unsubscribe(q)

    return EventSourceResponse(event_gen())


@app.get("/api/projects/{pid}/videos/{vid}/propagate/status")
def get_propagation_status(pid: str, vid: str):
    """Return current propagation progress for a video."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    state = _get_prop_state(pid, vid)
    propagated = video.get("propagated_frames", [])
    return {
        "is_running": state.is_running,
        "is_paused": state.is_paused,
        "paused_at_frame": state.paused_at_frame,
        "frames_done": len(propagated),
        "total_frames": video["num_frames"],
        "propagation_complete": video.get("propagation_complete", False),
        "last_frame": max(propagated) if propagated else -1,
        "start_frame": state.start_frame if state.is_running else 0,
        "whole_video_inference": video.get("whole_video_inference") or {},
    }


@app.post("/api/projects/{pid}/videos/{vid}/propagate/pause")
async def pause_propagation(pid: str, vid: str):
    """Cancel the running propagation task and record the last-reached frame."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    state = _get_prop_state(pid, vid)
    if not state.is_running:
        raise HTTPException(400, "No propagation is currently running")
    if state.task is not None:
        state.task.cancel()
    # Determine the last frame we reached from the saved propagated_frames list
    propagated = video.get("propagated_frames", [])
    paused_frame = max(propagated) if propagated else state.start_frame
    state.paused_at_frame = paused_frame
    state.is_paused = True
    state.is_running = False
    return {"status": "paused", "paused_at_frame": paused_frame}


class ResumeFromFrameRequest(BaseModel):
    resume_frame: int
    clear_from_frame: bool = True  # If true, delete masks from resume_frame onwards


@app.post("/api/projects/{pid}/videos/{vid}/propagate/resume")
async def resume_from_frame(pid: str, vid: str, req: ResumeFromFrameRequest):
    """
    Set up propagation to resume from a specific frame.

    This endpoint:
    1. Optionally clears all masks from resume_frame onwards
    2. Updates propagated_frames to only include frames < resume_frame
    3. Sets paused_at_frame so next propagation starts there

    Call /propagate after this to actually start propagation.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    state = _get_prop_state(pid, vid)
    if state.is_running:
        raise HTTPException(400, "Propagation is currently running. Pause it first.")

    total_frames = video.get("num_frames", 0)
    if req.resume_frame < 0 or (total_frames > 0 and req.resume_frame >= total_frames):
        raise HTTPException(400, f"resume_frame must be between 0 and {total_frames - 1}")

    ms_resume = VideoMaskStorage(pm.video_dir(pid, vid))

    deleted_count = 0
    if req.clear_from_frame:
        deleted_count = ms_resume.delete_masks_range(req.resume_frame, None)

    # Trim progress file and update config
    kept_count = pm.trim_propagated_frames(pid, vid, req.resume_frame)
    pm.update_video(pid, vid, {"propagation_complete": False})

    # Set paused state so next propagation resumes from here
    state.paused_at_frame = req.resume_frame - 1 if req.resume_frame > 0 else -1
    state.is_paused = True

    # Invalidate mask cache
    _invalidate_mask_cache(pid, vid)

    logger.info(
        f"Resume from frame {req.resume_frame}: deleted {deleted_count} files, "
        f"kept {kept_count} propagated frames"
    )

    return {
        "status": "ready_to_resume",
        "resume_frame": req.resume_frame,
        "deleted_files": deleted_count,
        "kept_propagated_frames": kept_count,
    }
