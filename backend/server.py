"""
SAM Web Tracker — FastAPI backend server.
Supports SAM3 (primary) with SAM2 fallback.
"""

import asyncio
import functools
import os
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

from project_manager import ProjectManager
from sam_predictor import SAMPredictor, _get_predictor
from video_processor import (
    encode_mask_as_png,
    ensure_faststart,
    export_video_with_masks,
    extract_frames,
    extract_frame_range,
    get_video_info,
    load_bboxes_json,
    load_masks_npz,
    save_bboxes_json,
    save_masks_npz,
    generate_thumbnail,
    composite_masks_as_png,
)

STREAM_BATCH_SIZE = 1000  # frames per propagation batch / anchor interval

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
# Cache results keyed by (npz_path, mtime_ns) so repeated GET /masks/{fidx}
# requests (e.g. scrubbing back to a visited frame) are instant.

_mask_encode_cache: dict[tuple, dict] = {}
_MASK_CACHE_MAX = 2000  # max entries (~2 KB overhead per entry, masks are large)


def _get_encoded_masks(masks_path: Path, objects: dict) -> dict:
    """Load and encode masks with an in-memory cache keyed by (path, mtime)."""
    if not masks_path.exists():
        return {}
    mtime_ns = masks_path.stat().st_mtime_ns
    key = (str(masks_path), mtime_ns)
    if key in _mask_encode_cache:
        return _mask_encode_cache[key]

    raw_masks = load_masks_npz(str(masks_path))
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
    """Remove all cached entries for a given video (call after propagation/save)."""
    prefix = str(pm.masks_dir(pid, vid))
    for k in [k for k in _mask_encode_cache if k[0].startswith(prefix)]:
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


class CreateProjectRequest(BaseModel):
    name: str


@app.post("/api/projects", status_code=201)
def create_project(req: CreateProjectRequest):
    return pm.create_project(req.name)


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

    # Extract preview frames and apply faststart in background
    frames_dir = str(pm.frames_dir(pid, vid))

    def do_post_upload():
        try:
            # Apply faststart so browser can stream the MP4 immediately
            ensure_faststart(str(perm_path))
        except Exception as e:
            logger.warning(f"faststart failed for {pid}/{vid}: {e}")
        # No longer extracting preview frames — frames are extracted on
        # demand when the user annotates.
        pm.update_video(pid, vid, {
            "frames_extracted": True,
            "all_frames_extracted": False,
        })

    background_tasks.add_task(do_post_upload)

    return video_meta


# ─── Import video from server filesystem path ────────────────────────────────

class ImportVideoRequest(BaseModel):
    path: str


@app.post("/api/projects/{pid}/videos/import", status_code=201)
async def import_video(
    pid: str,
    req: ImportVideoRequest,
    background_tasks: BackgroundTasks,
):
    project = pm.get_project(pid)
    if project is None:
        raise HTTPException(404, "Project not found")

    src = Path(req.path)
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

    video_meta = pm.add_video(
        pid=pid,
        name=src.name,
        source_path=str(src),
        num_frames=info["num_frames"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
    )
    vid = video_meta["id"]

    # Copy the source video into the project directory (we need our own copy
    # so we can apply faststart without modifying the original).
    perm_path = pm.video_dir(pid, vid) / f"source{src.suffix}"
    shutil.copy2(str(src), str(perm_path))
    pm.update_video(pid, vid, {"source_path": str(perm_path)})

    # Apply faststart in background (no preview extraction — on-demand)
    def do_post_import():
        try:
            # Apply faststart so browser can stream the MP4 immediately
            ensure_faststart(str(perm_path))
        except Exception as e:
            logger.warning(f"faststart failed for imported {pid}/{vid}: {e}")
        pm.update_video(pid, vid, {
            "frames_extracted": True,
            "all_frames_extracted": False,
        })

    background_tasks.add_task(do_post_import)

    return video_meta


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


@app.patch("/api/projects/{pid}/videos/{vid}")
def update_video_meta(pid: str, vid: str, req: UpdateVideoRequest):
    """Update mutable video metadata fields (e.g. start_frame)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        return video
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
    pm.update_video(pid, vid, {
        "objects": {},
        "point_prompts": {},
        "instance_groups": {},
        "sam3_session_id": None,
        "propagated_frames": [],
        "propagation_complete": False,
        "frames_extracted": False,
        "all_frames_extracted": False,
        "preview_indices": [],
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
            try:
                extract_frame_range(source_path, str(frames_dir), fidx, fidx + 1)
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
        try:
            extract_frame_range(source_path, str(ann_dir), fidx, fidx + 1)
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

    # Saved masks on disk
    masks_dir = pm.masks_dir(pid, vid)
    saved_mask_frames: list[int] = []
    saved_mask_obj_counts: dict[int, int] = {}
    if masks_dir.exists():
        for npz_file in sorted(masks_dir.glob("*.npz")):
            try:
                fidx = int(npz_file.stem)
                saved_mask_frames.append(fidx)
                # Count how many objects are in each npz
                import numpy as np
                data = np.load(str(npz_file))
                saved_mask_obj_counts[fidx] = len(data.files)
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


# ─── Objects ──────────────────────────────────────────────────────────────────

class AddObjectRequest(BaseModel):
    name: str
    color: Optional[str] = None
    description: str = ""
    min_instances: int = 1
    max_instances: int = 1


@app.post("/api/projects/{pid}/videos/{vid}/objects", status_code=201)
def add_object(pid: str, vid: str, req: AddObjectRequest):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return pm.add_object(
        pid, vid, req.name, req.color,
        description=req.description,
        min_instances=req.min_instances,
        max_instances=req.max_instances,
    )


class UpdateObjectRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    min_instances: Optional[int] = None
    max_instances: Optional[int] = None


@app.patch("/api/projects/{pid}/videos/{vid}/objects/{oid}")
def update_object(pid: str, vid: str, oid: str, req: UpdateObjectRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if "name" in updates:
        pm.rename_object(pid, vid, oid, updates.pop("name"))
    if updates:
        pm.update_object(pid, vid, oid, **updates)
    return {"status": "ok"}


@app.delete("/api/projects/{pid}/videos/{vid}/objects/{oid}", status_code=204)
def remove_object(pid: str, vid: str, oid: str):
    try:
        sam.remove_object(pid, vid, int(oid))
    except Exception:
        pass
    pm.remove_object(pid, vid, oid)


# ─── Points / Masks ──────────────────────────────────────────────────────────

class AddPointsRequest(BaseModel):
    frame_idx: int
    points: list            # [[x_norm, y_norm], ...]
    labels: list            # [1, 0, ...]
    text: Optional[str] = None  # override text prompt (defaults to object description)
    anchor_mode: bool = False   # when True: fresh single-frame session, no cross-frame replay


def _replay_prompts(pid: str, vid: str, all_prompts: dict) -> tuple[list, set]:
    """Replay saved point prompts into the current SAM session (sorted by frame index).

    Returns (items, replayed_obj_ids) where replayed_obj_ids is the set of obj ID strings
    that had at least one point prompt replayed.

    Object IDs can be integers (e.g., "1") or instance keys (e.g., "1_1", "1_2").
    For SAM's obj_id parameter, we use a unique integer derived from the key.
    """
    frame_map = sam._frame_maps.get((pid, vid), [])
    items: list[tuple[int, str, int, dict]] = []  # (frame_idx, obj_id_str, sam_obj_id, prompt)

    def _to_sam_obj_id(obj_id_str: str) -> int:
        """Convert object ID string to SAM integer obj_id.

        "1" -> 1
        "1_1" -> 1001 (base * 1000 + instance)
        "1_2" -> 1002
        """
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
    for rf, obj_id_str, sam_oid, prompt in items:
        try:
            points = prompt["points"]
            labels = prompt["labels"]
            logger.info(f"  obj {obj_id_str} (sam_id={sam_oid}), frame {rf}: points={points}, labels={labels}")
            # SAM3 tracker mode doesn't support text with points
            sam.add_points(pid, vid, frame_idx=rf, obj_id=sam_oid,
                           points=points, labels=labels,
                           text=None)
            replayed_obj_ids.add(obj_id_str)
        except Exception as e:
            logger.warning(f"Predict replay: obj {obj_id_str} frame {rf}: {e}")

    logger.info(f"  -> Replayed obj IDs with visual prompts: {replayed_obj_ids}")
    return items, replayed_obj_ids


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
    prop_state = _prop_registry.get(vid)
    if prop_state is not None and prop_state.is_running:
        raise HTTPException(409, "Propagation is running — pause first before adding points.")

    # Guard 2: in anchor_mode, only designated anchor frames may be labeled
    if req.anchor_mode:
        start_f = video.get("start_frame", 0)
        num_frames_v = video["num_frames"]
        anchor_frames = _compute_anchor_frames(start_f, num_frames_v, STREAM_BATCH_SIZE)
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
                    other_obj_id_int = (
                        int(other_oid_str.split("_")[0]) if "_" in other_oid_str
                        else int(other_oid_str)
                    )
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
                session_id = sam.init_session(pid, vid, ann_dir)
                pm.update_video(pid, vid, {"sam3_session_id": session_id})
            except Exception as e:
                raise HTTPException(500, f"SAM session init failed: {e}")

            # Replay all previously saved point prompts so SAM knows about every
            # object annotated on earlier frames.  Sort by frame index — SAM3
            # requires sequential order or raises "Image features for frame N are
            # not cached" errors.
            all_prompts = pm.get_all_point_prompts(pid, vid)
            frame_map = sam._frame_maps.get((pid, vid), [])
            replay_items: list[tuple[int, int, dict]] = []
            for obj_id_str, frame_map_prompts in all_prompts.items():
                for fidx_str, prompt in frame_map_prompts.items():
                    replay_fidx = int(fidx_str)
                    if replay_fidx in frame_map:
                        replay_items.append((replay_fidx, int(obj_id_str), prompt))
            replay_items.sort(key=lambda x: x[0])
            for replay_fidx, replay_obj_id, prompt in replay_items:
                try:
                    sam.add_points(
                        pid, vid,
                        frame_idx=replay_fidx,
                        obj_id=replay_obj_id,
                        points=prompt["points"],
                        labels=prompt["labels"],
                        text=None,
                    )
                    logger.debug(f"Replayed prompts for obj {replay_obj_id} on frame {replay_fidx}")
                except Exception as e:
                    logger.warning(f"Failed to replay prompts for obj {replay_obj_id} frame {replay_fidx}: {e}")

    # Save prompts to config
    pm.save_point_prompts(pid, vid, oid, req.frame_idx, req.points, req.labels)

    # Standard single-instance point prompt
    try:
        logger.info(f"add_points: frame={req.frame_idx}, obj={oid}, points={req.points}, labels={req.labels}")
        outputs = sam.add_points(
            pid, vid,
            frame_idx=req.frame_idx,
            obj_id=int(oid),
            points=req.points,
            labels=req.labels,
            text=None,  # SAM3 tracker mode doesn't support text with points
        )
    except Exception as e:
        import traceback
        logger.error(f"SAM add_points error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"SAM inference error: {e}")

    # Encode masks as base64 PNGs and collect raw binary masks for persistence
    objects = video["objects"]
    mask_b64: dict[str, str] = {}
    raw_masks: dict[str, np.ndarray] = {}

    frame_outputs = outputs.get(req.frame_idx, outputs.get(str(req.frame_idx), {}))
    if not frame_outputs and outputs:
        # SAM3 might key by int or string
        first_key = next(iter(outputs))
        frame_outputs = outputs[first_key]

    obj_ids = frame_outputs.get("out_obj_ids", [])
    masks = frame_outputs.get("out_binary_masks", [])

    for i, obj_id in enumerate(obj_ids):
        mask = masks[i] if i < len(masks) else None
        if mask is None:
            continue
        # Squeeze to (H, W)
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = np.squeeze(mask)
        raw_masks[str(obj_id)] = mask
        obj_color = _get_instance_color(str(obj_id), objects)
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

    # Persist updated masks to disk so they replace any previously propagated
    # masks for this frame.  Merge with existing npz so objects not yet in the
    # current session still retain their saved masks.
    if raw_masks:
        masks_path = pm.masks_dir(pid, vid) / f"{req.frame_idx:06d}.npz"
        existing = load_masks_npz(str(masks_path)) if masks_path.exists() else {}
        save_masks_npz(str(masks_path), {**existing, **raw_masks})
        _invalidate_mask_cache(pid, vid)

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
    # Remove this object's mask from the frame's NPZ (leave other objects intact)
    masks_path = pm.masks_dir(pid, vid) / f"{frame_idx:06d}.npz"
    if masks_path.exists():
        existing = load_masks_npz(str(masks_path))
        remaining = {k: v for k, v in existing.items() if k != oid}
        if remaining:
            save_masks_npz(str(masks_path), remaining)
        else:
            masks_path.unlink()
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
    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)

    if not masks_dir.exists():
        return {"status": "ok", "frames_swapped": 0}

    swapped = 0
    for npz_path in sorted(masks_dir.glob("*.npz")):
        fidx = int(npz_path.stem)
        if req.from_frame >= 0 and fidx < req.from_frame:
            continue
        if req.to_frame >= 0 and fidx > req.to_frame:
            continue

        masks = load_masks_npz(str(npz_path))
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
        save_masks_npz(str(npz_path), masks)

        # Swap bboxes too
        bbox_path = bboxes_dir / f"{fidx:06d}.json"
        if bbox_path.exists():
            bboxes = json.loads(bbox_path.read_text())
            a_bbox = bboxes.pop(req.obj_a, None)
            b_bbox = bboxes.pop(req.obj_b, None)
            if b_bbox is not None:
                bboxes[req.obj_a] = b_bbox
            if a_bbox is not None:
                bboxes[req.obj_b] = a_bbox
            bbox_path.write_text(json.dumps(bboxes, indent=2))

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

    masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
    if not masks_path.exists():
        return JSONResponse({"masks": {}}, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    mask_b64 = _get_encoded_masks(masks_path, video["objects"])
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
    Delete mask/bbox files in [from_frame, to_frame] (both inclusive, None = unbounded).
    Returns number of frame files deleted.
    """
    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)
    deleted = 0
    for npz in sorted(masks_dir.glob("*.npz")):
        fidx = int(npz.stem)
        if from_frame is not None and fidx < from_frame:
            continue
        if to_frame is not None and fidx > to_frame:
            continue
        npz.unlink()
        bbox = bboxes_dir / f"{fidx:06d}.json"
        if bbox.exists():
            bbox.unlink()
        deleted += 1
    return deleted


@app.delete("/api/projects/{pid}/videos/{vid}/masks/{fidx}")
def delete_frame_masks(pid: str, vid: str, fidx: int):
    """Delete saved masks for a single frame (.npz and .json bbox files)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
    bboxes_path = pm.bboxes_dir(pid, vid) / f"{fidx:06d}.json"

    deleted = []
    if masks_path.exists():
        masks_path.unlink()
        deleted.append("masks")
    if bboxes_path.exists():
        bboxes_path.unlink()
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

    masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
    if not masks_path.exists():
        raise HTTPException(404, "No mask for this frame")

    raw_masks = load_masks_npz(str(masks_path))
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
    MP4 to disk.  Reads only the source video and masks/*.npz files — does
    not touch SAM sessions, annotated_frames, or inference state.

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
    masks_dir = str(pm.masks_dir(pid, vid))
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
                    masks_dir=masks_dir,
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

def _compute_anchor_frames(start_frame: int, num_frames: int, batch_size: int) -> list[int]:
    """Compute anchor frames for bidirectional batch propagation.

    The first anchor is at start_frame so the user labels the very beginning of
    the video and propagation starts immediately from there (forward only, since
    nothing comes before start_frame).  Subsequent anchors sit at the midpoint
    of each following batch window, giving equal forward/backward SAM context.

    Examples (batch_size=1000):
      start=0, num_frames=3500 → [0, 1000, 2000, 3000, 3499]
      start=0, num_frames=3000 → [0, 1000, 2000]
      start=0, num_frames=100  → [0, 99]
    """
    last = num_frames - 1
    anchors = list(range(start_frame, num_frames, batch_size))
    if not anchors:
        anchors = [last]
    elif anchors[-1] != last:
        anchors.append(last)
    return anchors


@app.get("/api/projects/{pid}/videos/{vid}/anchor_frames")
def get_anchor_frames(pid: str, vid: str):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    start = video.get("start_frame", 0)
    num_frames = video["num_frames"]
    anchors = _compute_anchor_frames(start, num_frames, STREAM_BATCH_SIZE)
    return {"anchor_frames": anchors, "count": len(anchors)}


class CommitAnchorRequest(BaseModel):
    anchor_index: int  # 0-based index in the anchor frames list


@app.post("/api/projects/{pid}/videos/{vid}/anchors/{frame_idx}/commit")
def commit_anchor_frame(pid: str, vid: str, frame_idx: int, req: CommitAnchorRequest):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    annotated = list(video.get("annotated_anchors", []))
    if frame_idx not in annotated:
        annotated = sorted(annotated + [frame_idx])
    pm.update_video(pid, vid, {"annotated_anchors": annotated})
    return {"status": "ok", "committed_frame": frame_idx, "anchor_index": req.anchor_index}


# ─── Propagation background task ─────────────────────────────────────────────

async def _run_propagation_bg(
    pid: str,
    vid: str,
    start_frame: int,
    source_path: str,
    num_frames: int,
    masks_dir,
    bboxes_dir,
    objects: dict,
    all_prompts: dict,
    state: PropagationState,
    end_frame: int = -1,
) -> None:
    """
    Forward-only batch propagation.

    Processes frames start_frame..num_frames-1 in STREAM_BATCH_SIZE chunks.
    Each batch [P1, P2]:
      - Seeds P1 from the previous frame's saved mask (cross-batch continuity /
        resume continuity), when P1 > 0 and a mask file for P1-1 exists.
      - Seeds every user-labeled frame within [P1, P2] from their point prompts
        (sorted ascending so SAM sees them in order).
      - Runs forward propagation from the earliest seeded frame to P2.
    Batches with no seeds (no labeled frames, no cross-batch mask) are skipped.
    """
    loop = asyncio.get_event_loop()

    # Compute simple sequential batches
    effective_end = end_frame if (end_frame >= 0 and end_frame < num_frames) else num_frames - 1
    batches: list[tuple[int, int]] = []
    p = start_frame
    while p <= effective_end:
        batches.append((p, min(p + STREAM_BATCH_SIZE - 1, effective_end)))
        p += STREAM_BATCH_SIZE

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

    def _seed_from_npz(frame_idx: int, npz_path: Path):
        """Load masks from npz and add as mask prompts at frame_idx."""
        if not Path(npz_path).exists():
            return
        try:
            saved = load_masks_npz(str(npz_path))
            for obj_id_str, mask in saved.items():
                obj_id_int = int(obj_id_str.split("_")[0]) if "_" in obj_id_str else int(obj_id_str)
                try:
                    sam.add_mask_prompt(pid, vid, frame_idx=frame_idx, obj_id=obj_id_int, mask=mask)
                except Exception as e:
                    logger.warning(f"add_mask_prompt obj {obj_id_str} frame {frame_idx}: {e}")
        except Exception as e:
            logger.warning(f"_seed_from_npz {npz_path}: {e}")

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
                    lambda: extract_frame_range(source_path, tmp_dir, P1, P2 + 1, _progress_cb)
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

                await publish("batch_start", {
                    "batch": batch_idx,
                    "batch_start": P1,
                    "batch_end": P2 + 1,
                    "total_batches": total_batches,
                    "status": "initializing_session",
                })

                await loop.run_in_executor(None, sam.init_session, pid, vid, tmp_dir)

                seeded_frames: list[int] = []

                # Cross-batch / resume continuity: seed P1 from frame P1-1's saved mask.
                if P1 > 0:
                    prev_npz = masks_dir / f"{P1 - 1:06d}.npz"
                    if prev_npz.exists():
                        await loop.run_in_executor(None, _seed_from_npz, P1, prev_npz)
                        seeded_frames.append(P1)

                # Seed every user-labeled frame inside this batch, sorted by frame index.
                def _seed_labeled_frames() -> list[int]:
                    frame_map = sam._frame_maps.get((pid, vid), [])
                    items: list[tuple[int, int, dict]] = []
                    for obj_id_str, frame_prompts in all_prompts.items():
                        obj_id_int = (
                            int(obj_id_str.split("_")[0]) if "_" in obj_id_str
                            else int(obj_id_str)
                        )
                        for fidx_str, prompt in frame_prompts.items():
                            fidx = int(fidx_str)
                            if fidx in frame_map:
                                items.append((fidx, obj_id_int, prompt))
                    items.sort(key=lambda x: x[0])
                    seeded: list[int] = []
                    for fidx, obj_id_int, prompt in items:
                        try:
                            sam.add_points(
                                pid, vid, frame_idx=fidx, obj_id=obj_id_int,
                                points=prompt["points"], labels=prompt["labels"],
                            )
                            if fidx not in seeded:
                                seeded.append(fidx)
                        except Exception as e:
                            logger.warning(
                                f"Seed labeled frame: obj {obj_id_int} frame {fidx}: {e}"
                            )
                    return seeded

                labeled_in_batch = await loop.run_in_executor(None, _seed_labeled_frames)
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

                for real_frame, (frame_masks, frame_bboxes) in sorted(frame_results.items()):
                    if frame_masks:
                        npz_data = {str(k): v for k, v in frame_masks.items()}
                        await loop.run_in_executor(
                            None, save_masks_npz,
                            str(masks_dir / f"{real_frame:06d}.npz"), npz_data,
                        )
                    if frame_bboxes:
                        bbox_data = {str(k): v for k, v in frame_bboxes.items()}
                        await loop.run_in_executor(
                            None, save_bboxes_json,
                            str(bboxes_dir / f"{real_frame:06d}.json"), bbox_data,
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
        await publish("error", {"error": str(e)})


# ─── Propagation SSE endpoint ─────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/propagate")
async def propagate_video(pid: str, vid: str, start_frame: int = 0, resume_from: int = -1, end_frame: int = -1):
    """
    Stream propagation results as Server-Sent Events.

    Propagation runs as a background task so client reconnects join the
    existing run rather than restarting it from scratch.

    resume_from: if >= 0, treat this as a resume from the given frame.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

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
        masks_dir = pm.masks_dir(pid, vid)
        bboxes_dir = pm.bboxes_dir(pid, vid)
        objects = video["objects"]
        all_prompts = pm.get_all_point_prompts(pid, vid)

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
        state.task = asyncio.create_task(
            _run_propagation_bg(
                pid, vid, actual_start, source_path, num_frames,
                masks_dir, bboxes_dir, objects, all_prompts, state,
                end_frame=end_frame,
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

    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)

    deleted_count = 0
    if req.clear_from_frame:
        # Delete all masks/bboxes from resume_frame onwards
        for d in (masks_dir, bboxes_dir):
            if d.exists():
                for f in d.iterdir():
                    try:
                        frame_idx = int(f.stem)
                        if frame_idx >= req.resume_frame:
                            f.unlink()
                            deleted_count += 1
                    except ValueError:
                        pass  # Skip non-numeric filenames

    # Update propagated_frames to only include frames before resume_frame
    propagated = video.get("propagated_frames", [])
    new_propagated = [f for f in propagated if f < req.resume_frame]

    # Update video config
    pm.update_video(pid, vid, {
        "propagated_frames": new_propagated,
        "propagation_complete": False,
    })

    # Set paused state so next propagation resumes from here
    state.paused_at_frame = req.resume_frame - 1 if req.resume_frame > 0 else -1
    state.is_paused = True

    # Invalidate mask cache
    _invalidate_mask_cache(pid, vid)

    logger.info(
        f"Resume from frame {req.resume_frame}: deleted {deleted_count} files, "
        f"kept {len(new_propagated)} propagated frames"
    )

    return {
        "status": "ready_to_resume",
        "resume_frame": req.resume_frame,
        "deleted_files": deleted_count,
        "kept_propagated_frames": len(new_propagated),
    }
