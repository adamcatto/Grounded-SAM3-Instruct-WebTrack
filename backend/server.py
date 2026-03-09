"""
SAM Web Tracker — FastAPI backend server.
Supports SAM3 (primary) with SAM2 fallback.
"""

import asyncio
import functools
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from identity_tracker import IdentityTracker, compute_mask_stats
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

STREAM_BATCH_SIZE = 1000  # frames extracted per streaming mini-batch

# Identity / confusion detection constants
ANOMALY_THRESHOLD = 2.5           # z-score to flag a mask as anomalous
CONFUSION_THRESHOLD = 0.4         # confusion score to open a confusion window
CONFUSION_RESET_THRESHOLD = 0.7   # confusion score → use full annotation fallback
TEXT_SIMILARITY_THRESHOLD = 0.3   # min text similarity for "similar pair" protection
N_POST_WINDOW_FRAMES = 10         # frames after window end for swap detection
MAX_INST_SLOTS = 10               # max SAM obj_id slots reserved per UI object

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
        self.start_frame: int = 0

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
    # Use hue shift + lightening for variety
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


# ─── Annotation Uncertainty helpers ───────────────────────────────────────────

def _save_annotation_uncertainty(
    pid: str, vid: str, frame_idx: int, 
    instance_confidences: list[tuple[str, float]],
) -> dict:
    """
    Save per-frame annotation confidence info to uncertainty.json.
    
    During annotation, we get detection confidence scores from SAM. We store
    these so the Uncertainty tab can show annotation quality even before
    propagation runs.
    
    Args:
        pid: Project ID
        vid: Video ID  
        frame_idx: Frame index being annotated
        instance_confidences: List of (obj_id, confidence) tuples for detected instances
    
    Returns:
        Updated uncertainty data dict
    """
    # Load existing uncertainty (may have propagation data or prior annotations)
    existing = pm.load_uncertainty(pid, vid)
    
    # Initialize structure if needed
    if "per_frame" not in existing:
        existing["per_frame"] = {}
    if "confusion_windows" not in existing:
        existing["confusion_windows"] = []
    if "similarity_matrix" not in existing:
        existing["similarity_matrix"] = {}
    
    # Calculate aggregate confidence for this frame
    if instance_confidences:
        avg_confidence = sum(c for _, c in instance_confidences) / len(instance_confidences)
        min_confidence = min(c for _, c in instance_confidences)
    else:
        avg_confidence = 1.0
        min_confidence = 1.0
    
    # Store per-instance confidence and derive a "confusion-like" score
    # Lower confidence → higher "uncertainty" for consistency with confusion scores
    # confusion_score = 1 - confidence (so lower confidence = higher uncertainty)
    frame_key = str(frame_idx)
    existing["per_frame"][frame_key] = {
        "confusion_score": round(1.0 - min_confidence, 4),  # Use min as worst case
        "annotation_confidence": round(avg_confidence, 4),
        "instance_confidences": {
            obj_id: round(conf, 4) for obj_id, conf in instance_confidences
        },
        "is_annotated": True,  # Flag to distinguish from propagation data
    }
    
    # Save back
    pm.save_uncertainty(pid, vid, existing)
    logger.info(
        f"Saved annotation uncertainty for frame {frame_idx}: "
        f"avg_conf={avg_confidence:.3f}, min_conf={min_confidence:.3f}"
    )
    
    return existing


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
      - Saved masks (masks/*.npz, masks_raw/) and bboxes (bboxes/*.json)
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

    # Wipe masks, masks_raw, bboxes, and frame directories
    masks_dir = pm.masks_dir(pid, vid)
    masks_raw_dir = pm.masks_raw_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)
    ann_dir = pm.annotated_frames_dir(pid, vid)
    frames_dir = pm.frames_dir(pid, vid)

    for d in (masks_dir, masks_raw_dir, bboxes_dir, ann_dir, frames_dir):
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


# ─── Tracking Params ─────────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/tracking_params")
def get_tracking_params(pid: str, vid: str):
    """Get the current tracking parameters for a video."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    
    # Return tracking_params with defaults if not set
    defaults = {
        "min_iou_threshold": 0.15,
        "max_area_ratio": 5.0,
        "max_centroid_jump": 0.25,
        "consecutive_reject_limit": 5,
    }
    params = video.get("tracking_params", {})
    # Merge with defaults
    return {**defaults, **params}


@app.patch("/api/projects/{pid}/videos/{vid}/tracking_params")
def update_tracking_params(pid: str, vid: str, params: dict):
    """
    Update tracking parameters for a video. These are read in real-time
    during propagation, so changes take effect immediately.
    
    Valid params:
      - min_iou_threshold: float (0-1), minimum IoU with previous frame
      - max_area_ratio: float (>1), max allowed area change ratio
      - max_centroid_jump: float (0-1), max centroid distance (normalized)
      - consecutive_reject_limit: int, frames before force-accept
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    
    # Validate params
    allowed_keys = {"min_iou_threshold", "max_area_ratio", "max_centroid_jump", "consecutive_reject_limit"}
    invalid_keys = set(params.keys()) - allowed_keys
    if invalid_keys:
        raise HTTPException(400, f"Invalid tracking params: {invalid_keys}")
    
    # Validate values
    if "min_iou_threshold" in params:
        v = params["min_iou_threshold"]
        if not (0 <= v <= 1):
            raise HTTPException(400, "min_iou_threshold must be between 0 and 1")
    if "max_area_ratio" in params:
        v = params["max_area_ratio"]
        if v < 1:
            raise HTTPException(400, "max_area_ratio must be >= 1")
    if "max_centroid_jump" in params:
        v = params["max_centroid_jump"]
        if not (0 <= v <= 1):
            raise HTTPException(400, "max_centroid_jump must be between 0 and 1")
    if "consecutive_reject_limit" in params:
        v = params["consecutive_reject_limit"]
        if not isinstance(v, int) or v < 1:
            raise HTTPException(400, "consecutive_reject_limit must be a positive integer")
    
    # Merge with existing params
    current = video.get("tracking_params", {})
    updated = {**current, **params}
    pm.update_video(pid, vid, {"tracking_params": updated})
    
    logger.info(f"Updated tracking params for {vid}: {updated}")
    return updated


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


@app.post("/api/projects/{pid}/videos/{vid}/objects/{oid}/instances", status_code=201)
def add_instance(pid: str, vid: str, oid: str):
    """
    Allocate a new SAM obj_id slot for a multi-instance object.
    Returns the new SAM obj_id (or 409 if max_instances already reached).
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    new_slot = pm.allocate_instance_slot(pid, vid, oid)
    if new_slot is None:
        obj = video["objects"].get(oid, {})
        raise HTTPException(409, f"Object already has max_instances={obj.get('max_instances', 1)} slots allocated")
    return {"sam_obj_id": new_slot, "ui_obj_id": oid}


@app.get("/api/projects/{pid}/videos/{vid}/instance_groups")
def get_instance_groups(pid: str, vid: str):
    """Return the {ui_obj_id: [sam_obj_id, ...]} mapping."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return pm.get_instance_groups(pid, vid)


# ─── Points / Masks ──────────────────────────────────────────────────────────

class AddPointsRequest(BaseModel):
    frame_idx: int
    points: list            # [[x_norm, y_norm], ...]
    labels: list            # [1, 0, ...]
    text: Optional[str] = None  # override text prompt (defaults to object description)


def _auto_detect_instances(
    pid: str, vid: str, base_obj_id: str, frame_idx: int,
    user_points: list, user_labels: list,
    text_desc: str, min_inst: int, max_inst: int,
    objects: dict, ann_dir: str
) -> tuple[dict, dict, list, list]:
    """
    Auto-detect multiple instances using text description and assign points.
    
    Strategy:
    1. Use text-based semantic segmentation to find all instances
    2. Find which instance(s) the user's points belong to
    3. Register each instance with its own object ID and point prompts
    
    Returns (raw_masks, mask_b64, new_objects, instance_confidences) where:
    - new_objects is a list of newly created instance object dicts
    - instance_confidences is a list of (obj_id, confidence) tuples
    """
    import numpy as np
    
    obj_id_int = int(base_obj_id)
    raw_masks: dict[str, np.ndarray] = {}
    mask_b64: dict[str, str] = {}
    new_objects: list[dict] = []  # Track newly created instance objects
    instance_confidences: list[tuple[str, float]] = []  # (obj_id, confidence) pairs
    
    # Step 1: Run text-based semantic segmentation to find all instances
    logger.info(f"  Step 1: Text segmentation for '{text_desc}' on frame {frame_idx}")
    try:
        # Re-init session for semantic mode (it resets state)
        session_id = sam.init_session(pid, vid, ann_dir)
        outputs = sam.add_text_prompt(pid, vid, frame_idx=frame_idx, 
                                      obj_id=obj_id_int, text=text_desc)
    except Exception as e:
        logger.warning(f"  Text segmentation failed: {e}")
        return raw_masks, mask_b64, new_objects, instance_confidences
    
    frame_outputs = outputs.get(frame_idx, {})
    detected_obj_ids = frame_outputs.get("out_obj_ids", [])
    detected_masks = frame_outputs.get("out_binary_masks", [])
    
    # Check if we got any masks (handle both list and numpy array cases)
    num_masks = len(detected_masks) if hasattr(detected_masks, '__len__') else 0
    if num_masks == 0:
        logger.info("  No instances detected via text segmentation")
        return raw_masks, mask_b64, new_objects, instance_confidences
    
    logger.info(f"  Text segmentation returned {num_masks} masks")
    
    # Get confidence scores if available
    detected_probs = frame_outputs.get("out_probs", [])
    has_probs = hasattr(detected_probs, '__len__') and len(detected_probs) > 0
    
    logger.info(f"  Text segmentation returned {num_masks} masks, probs available: {has_probs}")
    if has_probs:
        # Log confidence scores
        prob_values = []
        for p in detected_probs:
            if hasattr(p, 'item'):
                prob_values.append(p.item())
            elif hasattr(p, '__float__'):
                prob_values.append(float(p))
            else:
                prob_values.append(p)
        logger.info(f"  Confidence scores: {prob_values[:max_inst]}")
    
    # Process detected masks - take up to max_inst, filtering by confidence
    detected_instances = []
    min_confidence = 0.1  # Minimum confidence threshold
    min_area_pixels = 100  # Minimum mask area
    
    for i in range(min(num_masks, max_inst)):
        m = detected_masks[i]
        if hasattr(m, "numpy"):
            m = m.numpy()
        m = np.squeeze(m)
        
        # Get confidence for this mask
        conf = 1.0
        if has_probs and i < len(detected_probs):
            p = detected_probs[i]
            if hasattr(p, 'item'):
                conf = p.item()
            elif hasattr(p, '__float__'):
                conf = float(p)
            else:
                conf = float(p) if p else 1.0
        
        area = np.sum(m)
        if area < min_area_pixels:
            logger.info(f"  Instance {i}: skipping, area={area} < {min_area_pixels}")
            continue
        
        if conf < min_confidence:
            logger.info(f"  Instance {i}: skipping, confidence={conf:.3f} < {min_confidence}")
            continue
        
        # Calculate centroid
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        h, w = m.shape
        cx = float(xs.mean()) / w
        cy = float(ys.mean()) / h
        
        detected_instances.append({
            'mask': m,
            'centroid': (cx, cy),
            'area': area,
            'confidence': conf,
            'index': i,
        })
        logger.info(f"  Instance {i}: area={area}, confidence={conf:.3f}, centroid=({cx:.3f}, {cy:.3f})")
    
    logger.info(f"  Found {len(detected_instances)} valid instances after filtering")
    
    if not detected_instances:
        return raw_masks, mask_b64, new_objects, instance_confidences
    
    # Step 2: Match user's positive points to detected instances
    # A point belongs to an instance if it falls within the mask or is nearest to it
    def point_in_mask(px: float, py: float, mask: np.ndarray) -> bool:
        h, w = mask.shape
        x_pixel = int(px * w)
        y_pixel = int(py * h)
        if 0 <= x_pixel < w and 0 <= y_pixel < h:
            return mask[y_pixel, x_pixel] > 0
        return False
    
    def distance_to_centroid(px: float, py: float, centroid: tuple) -> float:
        cx, cy = centroid
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    
    # Assign each positive point to an instance
    instance_points: dict[int, list] = {i: [] for i in range(len(detected_instances))}
    instance_labels: dict[int, list] = {i: [] for i in range(len(detected_instances))}
    
    for pt, lbl in zip(user_points, user_labels):
        if lbl != 1:  # Only consider positive points for matching
            continue
        px, py = pt
        
        # First try direct mask containment
        matched_idx = None
        for i, inst in enumerate(detected_instances):
            if point_in_mask(px, py, inst['mask']):
                matched_idx = i
                break
        
        # If no direct hit, assign to nearest centroid
        if matched_idx is None:
            min_dist = float('inf')
            for i, inst in enumerate(detected_instances):
                d = distance_to_centroid(px, py, inst['centroid'])
                if d < min_dist:
                    min_dist = d
                    matched_idx = i
        
        if matched_idx is not None:
            instance_points[matched_idx].append(pt)
            instance_labels[matched_idx].append(lbl)
            logger.info(f"  Point ({px:.3f}, {py:.3f}) matched to instance {matched_idx}")
    
    # Step 3: Re-init session for tracker mode and register each instance
    session_id = sam.init_session(pid, vid, ann_dir)
    
    for i, inst in enumerate(detected_instances):
        # Determine instance key
        if i == 0:
            mask_key = str(obj_id_int)
        else:
            mask_key = f"{obj_id_int}_{i}"
            # Register the instance object
            try:
                new_obj = pm.register_instance_object(pid, vid, base_obj_id, mask_key, i)
                new_objects.append(new_obj)
                logger.info(f"  Registered instance object {mask_key}")
            except Exception as e:
                logger.warning(f"  Failed to register instance {mask_key}: {e}")
        
        # Determine points for this instance
        inst_pts = instance_points.get(i, [])
        inst_lbls = instance_labels.get(i, [])
        
        # If no user points assigned, use centroid
        if not inst_pts:
            cx, cy = inst['centroid']
            inst_pts = [[cx, cy]]
            inst_lbls = [1]
            logger.info(f"  Instance {mask_key}: using centroid ({cx:.3f}, {cy:.3f})")
        else:
            logger.info(f"  Instance {mask_key}: {len(inst_pts)} user points")
        
        # Save point prompts
        pm.save_point_prompts(pid, vid, mask_key, frame_idx, inst_pts, inst_lbls)
        
        # Calculate SAM obj_id for this instance
        if "_" in mask_key:
            parts = mask_key.split("_")
            sam_obj_id = int(parts[0]) * 1000 + int(parts[1])
        else:
            sam_obj_id = int(mask_key)
        
        # Add points to SAM for this instance
        try:
            sam.add_points(pid, vid, frame_idx=frame_idx, obj_id=sam_obj_id,
                          points=inst_pts, labels=inst_lbls, text=None)
        except Exception as e:
            logger.warning(f"  SAM add_points for {mask_key} failed: {e}")
        
        # Store the mask and confidence
        raw_masks[mask_key] = inst['mask']
        instance_confidences.append((mask_key, inst['confidence']))
        obj_color = _get_instance_color(mask_key, objects)
        mask_b64[mask_key] = encode_mask_as_png(inst['mask'], obj_color)
    
    logger.info(f"  Auto-detected {len(raw_masks)} instances for object {base_obj_id}")
    return raw_masks, mask_b64, new_objects, instance_confidences


@app.post("/api/projects/{pid}/videos/{vid}/objects/{oid}/points")
def add_points(pid: str, vid: str, oid: str, req: AddPointsRequest):
    """
    Add/update point prompts for an object on a frame.
    Returns per-object mask PNG overlays as base64.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    # Ensure SAM session exists and includes the requested frame.
    # The annotated_frames dir accumulates frames on demand, so whenever
    # the user annotates a new frame we must re-init the session so SAM
    # picks up the newly extracted JPG.
    ann_dir = str(pm.annotated_frames_dir(pid, vid))
    if not Path(ann_dir).exists() or not list(Path(ann_dir).glob("*.jpg")):
        raise HTTPException(400, "No annotated frames extracted yet. Extract a frame first.")

    session_id = sam.get_session_id(pid, vid)
    need_reinit = session_id is None

    if not need_reinit:
        # Check whether the requested frame is already loaded in the session
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

        # Replay all previously saved point prompts so SAM knows about
        # every object that was annotated on earlier frames.
        # IMPORTANT: collect all items and sort by frame index before replaying.
        # SAM3 requires frames to be processed in sequential order — replaying
        # object-by-object (which may put a later frame before an earlier one)
        # causes "Image features for frame N are not cached" errors.
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
                    text=None,  # SAM3 tracker mode doesn't support text with points
                )
                logger.debug(f"Replayed prompts for obj {replay_obj_id} on frame {replay_fidx}")
            except Exception as e:
                logger.warning(f"Failed to replay prompts for obj {replay_obj_id} frame {replay_fidx}: {e}")

    # Save prompts to config
    pm.save_point_prompts(pid, vid, oid, req.frame_idx, req.points, req.labels)

    # Note: Text descriptions are stored in project config and used for:
    # 1. Text-only segmentation (when no points are provided)
    # 2. Identity tracking during propagation
    # But SAM3's tracker mode (point prompts) doesn't accept text directly.

    # Check if this is a multi-instance object that should auto-detect instances
    obj_info = video["objects"].get(oid, {})
    max_inst = obj_info.get("max_instances", 1)
    min_inst = obj_info.get("min_instances", 1)
    text_desc = obj_info.get("description") or obj_info.get("name", "")
    
    # If max_instances > 1 and we have a text description, use text-based multi-instance detection
    # This finds all instances matching the description and assigns points to them
    if max_inst > 1 and text_desc and len(req.points) > 0:
        logger.info(f"=== MULTI-INSTANCE AUTO-DETECTION: obj={oid}, max={max_inst}, text={text_desc!r} ===")
        
        raw_masks, mask_b64, new_objects, instance_confidences = _auto_detect_instances(
            pid, vid, oid, req.frame_idx, req.points, req.labels,
            text_desc, min_inst, max_inst, video["objects"], ann_dir
        )
        
        # Save annotation uncertainty (per-frame confidence)
        uncertainty_update = None
        if instance_confidences:
            uncertainty_update = _save_annotation_uncertainty(
                pid, vid, req.frame_idx, instance_confidences
            )
        
        # Persist masks
        if raw_masks:
            masks_path = pm.masks_dir(pid, vid) / f"{req.frame_idx:06d}.npz"
            existing = load_masks_npz(str(masks_path)) if masks_path.exists() else {}
            save_masks_npz(str(masks_path), {**existing, **raw_masks})
            _invalidate_mask_cache(pid, vid)
        
        # Return new objects and uncertainty so frontend can merge them into state
        return {
            "frame_idx": req.frame_idx, 
            "masks": mask_b64,
            "new_objects": new_objects,  # List of newly created instance objects
            "uncertainty_update": uncertainty_update,  # Updated uncertainty data
        }

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

    # Build instance confidence list (if SAM provided probabilities)
    instance_confidences: list[tuple[str, float]] = []
    detected_probs = frame_outputs.get("out_probs", [])
    has_probs = hasattr(detected_probs, "__len__") and len(detected_probs) > 0
    if has_probs:
        for i, obj_id in enumerate(obj_ids):
            conf = 1.0
            if i < len(detected_probs):
                p = detected_probs[i]
                if hasattr(p, 'item'):
                    conf = p.item()
                elif hasattr(p, '__float__'):
                    conf = float(p)
                else:
                    try:
                        conf = float(p)
                    except Exception:
                        conf = 1.0
            instance_confidences.append((str(obj_id), conf))

    # Save annotation uncertainty for this single-instance update if available
    uncertainty_update = None
    if instance_confidences:
        try:
            uncertainty_update = _save_annotation_uncertainty(pid, vid, req.frame_idx, instance_confidences)
        except Exception as e:
            logger.warning(f"Failed to save annotation uncertainty: {e}")

    # Persist updated masks to disk so they replace any previously propagated
    # masks for this frame.  Merge with existing npz so objects not yet in the
    # current session still retain their saved masks.
    if raw_masks:
        masks_path = pm.masks_dir(pid, vid) / f"{req.frame_idx:06d}.npz"
        existing = load_masks_npz(str(masks_path)) if masks_path.exists() else {}
        save_masks_npz(str(masks_path), {**existing, **raw_masks})
        _invalidate_mask_cache(pid, vid)

    return {"frame_idx": req.frame_idx, "masks": mask_b64, "uncertainty_update": uncertainty_update}


@app.post("/api/projects/{pid}/videos/{vid}/objects/{oid}/clear")
def clear_object_points(pid: str, vid: str, oid: str):
    """Clear all point prompts for an object."""
    pm.clear_point_prompts(pid, vid, oid)
    try:
        sam.clear_object_prompts(pid, vid, int(oid))
    except Exception as e:
        logger.warning(f"SAM clear error: {e}")
    return {"status": "ok"}


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


# ─── Single-frame prediction ─────────────────────────────────────────────────

class PredictFrameRequest(BaseModel):
    use_prev_frame_mask: bool = True


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


def _register_text_only_objects(pid: str, vid: str, replayed_obj_ids: set[str]) -> dict:
    """
    For objects that have a description but no visual point prompts, run
    text-based semantic segmentation on the earliest available frame.
    
    Returns a dict of {obj_id: mask_outputs} for objects that were processed.
    
    Note: replayed_obj_ids contains string IDs like "1", "1_1", "2", etc.
    We skip objects whose string ID is in replayed_obj_ids (they already have points).
    We also skip instance objects (containing "_") since they derive from parent objects.
    """
    frame_map = sam._frame_maps.get((pid, vid), [])
    if not frame_map:
        logger.debug("_register_text_only_objects: no frames in frame_map")
        return {}
    earliest_frame = frame_map[0]
    video = pm.get_video(pid, vid)
    if not video:
        logger.debug("_register_text_only_objects: no video found")
        return {}
    
    text_only_results = {}
    for obj_id_str, obj in video.get("objects", {}).items():
        # Skip instance objects (they're derived from parent objects via text segmentation)
        if "_" in obj_id_str:
            logger.debug(f"_register_text_only_objects: obj {obj_id_str} is an instance object, skipping")
            continue
        # Skip if already has point prompts
        if obj_id_str in replayed_obj_ids:
            logger.debug(f"_register_text_only_objects: obj {obj_id_str} already has point prompts, skipping")
            continue
        obj_id_int = int(obj_id_str)
        text = obj.get("description") or obj.get("name", "")
        if not text:
            logger.debug(f"_register_text_only_objects: obj {obj_id_str} has no text description")
            continue
        try:
            logger.info(f"_register_text_only_objects: running text-only segmentation for obj {obj_id_str}: {text!r}")
            outputs = sam.add_text_prompt(pid, vid, frame_idx=earliest_frame, 
                                          obj_id=obj_id_int, text=text)
            logger.info(f"_register_text_only_objects: outputs={outputs}")
            if outputs:
                frame_outputs = outputs.get(earliest_frame, {})
                text_only_results[obj_id_str] = frame_outputs
                logger.info(f"_register_text_only_objects: got masks for obj {obj_id_str}: "
                           f"obj_ids={frame_outputs.get('out_obj_ids', [])}")
        except Exception as e:
            logger.warning(f"Text-only prompt for obj {obj_id_str} failed: {e}", exc_info=True)
    
    return text_only_results


@app.post("/api/projects/{pid}/videos/{vid}/predict_frame/{frame_idx}")
def predict_single_frame(pid: str, vid: str, frame_idx: int, req: PredictFrameRequest):
    """
    Run single-frame SAM prediction for the given frame.

    If use_prev_frame_mask=true and frame (frame_idx-1) has a saved .npz mask,
    that mask is temporarily injected as a seed prompt before propagating one
    frame forward.  The session is re-initialised afterwards to remove the
    temporary state, leaving only the user's saved point prompts.

    Returns: {frame_idx, masks: {obj_id: base64_png}, used_prev_frame_mask, prev_frame_idx}
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    source_path = video.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(400, "Video source file not found")

    objects = video.get("objects", {})
    ann_dir = pm.annotated_frames_dir(pid, vid)
    ann_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = pm.masks_dir(pid, vid)

    # 1. Extract frame N to annotated_frames/ if missing
    frame_n_path = ann_dir / f"{frame_idx:06d}.jpg"
    if not frame_n_path.exists():
        try:
            extract_frame_range(source_path, str(ann_dir), frame_idx, frame_idx + 1)
        except Exception as e:
            raise HTTPException(500, f"Failed to extract frame {frame_idx}: {e}")
    if not frame_n_path.exists():
        raise HTTPException(500, f"Frame {frame_idx} could not be extracted")

    # 2. Check for previous-frame saved mask
    used_prev = False
    prev_frame_idx = frame_idx - 1
    prev_masks: dict = {}

    if req.use_prev_frame_mask and frame_idx > 0:
        prev_mask_path = masks_dir / f"{prev_frame_idx:06d}.npz"
        if prev_mask_path.exists():
            try:
                prev_masks = load_masks_npz(str(prev_mask_path))
            except Exception as e:
                logger.warning(f"Could not load prev mask at frame {prev_frame_idx}: {e}")

    # 3. If seeding from prev mask, also ensure frame N-1 is extracted
    if prev_masks:
        frame_nm1_path = ann_dir / f"{prev_frame_idx:06d}.jpg"
        if not frame_nm1_path.exists():
            try:
                extract_frame_range(source_path, str(ann_dir), prev_frame_idx, prev_frame_idx + 1)
            except Exception as e:
                logger.warning(f"Could not extract prev frame {prev_frame_idx}: {e}")
                prev_masks = {}  # skip seeding if frame unavailable

    # 4. Re-init session so it includes all current annotated frames
    ann_dir_str = str(ann_dir)
    try:
        session_id = sam.init_session(pid, vid, ann_dir_str)
        pm.update_video(pid, vid, {"sam3_session_id": session_id})
    except Exception as e:
        raise HTTPException(500, f"SAM session init failed: {e}")

    all_prompts = pm.get_all_point_prompts(pid, vid)
    _, replayed_ids = _replay_prompts(pid, vid, all_prompts)
    
    # For text-only objects (no point prompts), run semantic segmentation
    # and use the returned masks directly
    text_only_masks = _register_text_only_objects(pid, vid, replayed_ids)
    logger.info(f"predict_single_frame: text_only_masks keys={list(text_only_masks.keys())}")

    # 5. Inject prev-frame mask as a temporary seed (not persisted)
    if prev_masks:
        frame_map = sam._frame_maps.get((pid, vid), [])
        if prev_frame_idx in frame_map:
            for obj_id_str, mask in prev_masks.items():
                try:
                    sam.add_mask_prompt(pid, vid, prev_frame_idx, int(obj_id_str), mask)
                    used_prev = True
                except Exception as e:
                    logger.warning(f"Could not seed prev mask obj {obj_id_str}: {e}")

    # 6. Propagate — start from prev frame (if seeded) or earliest available frame
    #    For text-only objects, we already have masks from _register_text_only_objects
    #    but those are for earliest_frame, not frame_idx. We need to run text segmentation
    #    on frame_idx directly.
    frame_map = sam._frame_maps.get((pid, vid), [])
    start_from = prev_frame_idx if used_prev else (frame_map[0] if frame_map else frame_idx)
    if start_from not in frame_map and frame_map:
        start_from = frame_map[0]

    raw_masks: dict = {}
    
    # If we have text-only objects and the requested frame is in the session,
    # run text segmentation directly on that frame
    if text_only_masks and frame_idx in frame_map:
        logger.info(f"=== TEXT-ONLY SEGMENTATION on frame {frame_idx} ===")
        video = pm.get_video(pid, vid)
        for obj_id_str, obj in (video.get("objects", {}) if video else {}).items():
            # Skip instance objects (derived from parent)
            if "_" in obj_id_str:
                continue
            if obj_id_str in replayed_ids:
                logger.info(f"  obj {obj_id_str}: has point prompts, skipping text-only")
                continue  # has point prompts
            obj_id_int = int(obj_id_str)
            text = obj.get("description") or obj.get("name", "")
            if not text:
                logger.info(f"  obj {obj_id_str}: no text description")
                continue
            
            min_inst = obj.get("min_instances", 1)
            max_inst = obj.get("max_instances", 1)
            logger.info(f"  obj {obj_id_str}: text={text!r}, min_inst={min_inst}, max_inst={max_inst}")
            
            try:
                # Re-init session for each text query (semantic mode resets state)
                session_id = sam.init_session(pid, vid, ann_dir_str)
                outputs = sam.add_text_prompt(pid, vid, frame_idx=frame_idx, 
                                              obj_id=obj_id_int, text=text)
                if outputs:
                    frame_outputs = outputs.get(frame_idx, {})
                    out_obj_ids = frame_outputs.get("out_obj_ids", [])
                    masks_list = frame_outputs.get("out_binary_masks", [])
                    logger.info(f"  obj {obj_id_int}: SAM3 returned {len(out_obj_ids)} detections: {out_obj_ids}")
                    
                    # For multi-instance objects, take up to max_instances masks
                    # AND create synthetic point prompts for each instance
                    masks_taken = 0
                    instance_centroids = []  # Store centroids for point prompt generation
                    
                    for i, oid in enumerate(out_obj_ids):
                        if masks_taken >= max_inst:
                            logger.info(f"    Reached max_instances={max_inst}, stopping")
                            break
                        m = masks_list[i] if i < len(masks_list) else None
                        if m is not None:
                            if hasattr(m, "numpy"):
                                m = m.numpy()
                            m = np.squeeze(m)
                            
                            # For multi-instance, use instance slots: obj_id, obj_id_1, obj_id_2, etc.
                            if masks_taken == 0:
                                mask_key = str(obj_id_int)
                            else:
                                mask_key = f"{obj_id_int}_{masks_taken}"
                            raw_masks[mask_key] = m
                            
                            # Calculate centroid for this instance
                            ys, xs = np.where(m > 0)
                            if len(xs) > 0:
                                h, w = m.shape
                                cx = float(xs.mean()) / w
                                cy = float(ys.mean()) / h
                                instance_centroids.append({
                                    'mask_key': mask_key,
                                    'centroid': (cx, cy),
                                    'instance_num': masks_taken,
                                })
                            
                            logger.info(f"    Instance {masks_taken}: mask_key={mask_key}, shape={m.shape}, pixels={np.sum(m)}")
                            masks_taken += 1
                    
                    # Save synthetic point prompts for each instance so they can be tracked
                    # in future frames. Each instance gets its own object slot.
                    for inst_data in instance_centroids:
                        mask_key = inst_data['mask_key']
                        cx, cy = inst_data['centroid']
                        instance_num = inst_data['instance_num']
                        
                        # Register instance objects (e.g., "1_1") in config with varied colors
                        if instance_num > 0:
                            try:
                                pm.register_instance_object(pid, vid, str(obj_id_int), mask_key, instance_num)
                                logger.info(f"    Registered instance object {mask_key} in config")
                            except Exception as reg_e:
                                logger.warning(f"    Failed to register instance object {mask_key}: {reg_e}")
                        
                        # Save point prompt for this instance
                        pm.save_point_prompts(pid, vid, mask_key, frame_idx, [[cx, cy]], [1])
                        logger.info(f"    Saved synthetic point prompt for {mask_key} at ({cx:.3f}, {cy:.3f})")
                    
                    if masks_taken < min_inst:
                        logger.warning(f"  obj {obj_id_int}: only found {masks_taken} instances, min_instances={min_inst}")
            except Exception as e:
                logger.warning(f"Text-only segmentation for obj {obj_id_int} on frame {frame_idx} failed: {e}")
    
    # For objects with point prompts, use propagation
    if replayed_ids:
        # Re-init session and replay prompts for propagation
        session_id = sam.init_session(pid, vid, ann_dir_str)
        _replay_prompts(pid, vid, all_prompts)
        
        try:
            for item in sam.propagate_stream(pid, vid, start_frame_idx=start_from):
                sam_idx = item.get("frame_index")
                real_idx = sam.to_real_idx(pid, vid, sam_idx) if sam_idx is not None else -1
                # Unify output dict regardless of SAM version key style
                outputs = item.get("outputs", {})
                if isinstance(outputs, dict):
                    fo = (outputs.get(sam_idx)
                          or outputs.get(str(sam_idx))
                          or outputs.get(real_idx)
                          or outputs.get(str(real_idx))
                          or outputs)
                else:
                    fo = {}
                if real_idx == frame_idx:
                    obj_ids = fo.get("out_obj_ids", [])
                    masks_list = fo.get("out_binary_masks", [])
                    for i, oid in enumerate(obj_ids):
                        m = masks_list[i] if i < len(masks_list) else None
                        if m is None:
                            continue
                        if hasattr(m, "numpy"):
                            m = m.numpy()
                        raw_masks[str(oid)] = np.squeeze(m)
                    break  # got what we need
        except Exception as e:
            logger.error(f"predict_frame propagation failed: {e}", exc_info=True)
            raise HTTPException(500, f"Prediction failed: {e}")

    # 7. Persist masks for frame N
    if raw_masks:
        masks_path = masks_dir / f"{frame_idx:06d}.npz"
        existing = load_masks_npz(str(masks_path)) if masks_path.exists() else {}
        save_masks_npz(str(masks_path), {**existing, **raw_masks})
        _invalidate_mask_cache(pid, vid)
        pm.mark_frame_propagated(pid, vid, frame_idx)

    # 8. Re-init session to remove the temporary seed (clean state)
    if used_prev:
        try:
            clean_sid = sam.init_session(pid, vid, ann_dir_str)
            pm.update_video(pid, vid, {"sam3_session_id": clean_sid})
            _, clean_replayed = _replay_prompts(pid, vid, all_prompts)
            _register_text_only_objects(pid, vid, clean_replayed)
        except Exception as e:
            logger.warning(f"Session cleanup after predict_frame failed: {e}")

    # 9. Encode masks as base64 PNGs for the frontend
    logger.info(f"=== FINAL MASKS for frame {frame_idx}: {list(raw_masks.keys())} ===")
    for obj_id, mask in raw_masks.items():
        if hasattr(mask, 'shape'):
            logger.info(f"  obj {obj_id}: shape={mask.shape}, pixels={np.sum(mask)}")
    
    mask_b64: dict[str, str] = {}
    for obj_id, mask in raw_masks.items():
        obj_color = _get_instance_color(str(obj_id), objects)
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

    return {
        "frame_idx": frame_idx,
        "masks": mask_b64,
        "used_prev_frame_mask": used_prev,
        "prev_frame_idx": prev_frame_idx if used_prev else None,
    }


# ─── Save frame to inference state ───────────────────────────────────────────

@app.post("/api/projects/{pid}/videos/{vid}/frames/{frame_idx}/save_inference")
def save_frame_to_inference(pid: str, vid: str, frame_idx: int):
    """
    Promote a predicted frame into the session replay chain by saving a
    synthetic centroid point prompt for each object that has a mask.
    When the SAM session is next re-initialised these prompts are replayed,
    making the frame a first-class annotated keyframe.
    Subsequent real point-prompt clicks on the same frame overwrite the
    synthetic prompts naturally via save_point_prompts.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    masks_path = pm.masks_dir(pid, vid) / f"{frame_idx:06d}.npz"
    if not masks_path.exists():
        raise HTTPException(404, f"No saved masks for frame {frame_idx}")

    raw_masks = load_masks_npz(str(masks_path))
    objects = video.get("objects", {})
    saved_count = 0

    for obj_id_str in objects:
        mask = raw_masks.get(obj_id_str)
        if mask is None:
            continue
        mask_arr = np.array(mask, dtype=bool)
        ys, xs = np.where(mask_arr)
        if len(xs) == 0:
            continue
        cx = float(xs.mean()) / mask_arr.shape[1]
        cy = float(ys.mean()) / mask_arr.shape[0]
        pm.save_point_prompts(pid, vid, obj_id_str, frame_idx, [[cx, cy]], [1])
        saved_count += 1

    return {"status": "saved", "frame_idx": frame_idx, "objects_saved": saved_count}


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


# ─── Propagation background task ─────────────────────────────────────────────

async def _run_propagation_bg(
    pid: str,
    vid: str,
    actual_start: int,
    batches: list[tuple[int, int]],
    all_prompts: dict,
    num_frames: int,
    source_path: str,
    masks_dir,
    bboxes_dir,
    objects: dict,
    is_sam3: bool,
    state: PropagationState,
    seed_frame: int = -1,
) -> None:
    """
    Run propagation as a true background task, publishing SSE events to all
    subscribers in `state`.  Survives client disconnects — new SSE connections
    to /propagate will join the running task via the subscriber queue.
    """
    import concurrent.futures

    async def run():
        loop = asyncio.get_event_loop()
        total_propagated = 0
        sam3_annotated_frame: int | None = None
        frames_to_process = num_frames - actual_start

        # ── Emit init event so clients know the true total frames ─────────
        yield {
            "event": "init",
            "data": json.dumps({
                "actual_start": actual_start,
                "frames_to_process": frames_to_process,
                "total_batches": len(batches),
            }),
        }

        # ── Build IdentityTracker ─────────────────────────────────────────
        obj_ids_int = [int(oid) for oid in objects.keys()]
        descriptions = {
            int(oid): obj.get("description") or obj.get("name", "")
            for oid, obj in objects.items()
        }
        try:
            from sam_predictor import _get_predictor as _gp
            _sam_predictor_inst = _gp()
        except Exception:
            _sam_predictor_inst = None
        
        # Get tracking params from video config (allows real-time updates)
        tracking_params = pm.get_tracking_params(pid, vid)
        tracker = IdentityTracker(
            object_ids=obj_ids_int,
            descriptions=descriptions,
            anomaly_threshold=ANOMALY_THRESHOLD,
            confusion_threshold=CONFUSION_THRESHOLD,
            confusion_reset_threshold=CONFUSION_RESET_THRESHOLD,
            text_similarity_threshold=TEXT_SIMILARITY_THRESHOLD,
            sam_predictor=_sam_predictor_inst,
            tracking_params=tracking_params,
        )
        # Register original point prompts as seed annotations for fallback
        for oid_str, frame_prompts in all_prompts.items():
            oid_int = int(oid_str)
            seed_pts = [
                (int(fidx_str), p["points"], p["labels"])
                for fidx_str, p in frame_prompts.items()
            ]
            tracker.set_seed_annotations(oid_int, seed_pts)

        # Masks at seed frames (for swap detection later)
        seed_frame_masks: dict[int, np.ndarray] = {}

        # Post-window masks for swap detection: {window_idx: {obj_id: mask}}
        post_window_masks_by_window: dict[int, dict[int, np.ndarray]] = {}

        masks_raw_dir = pm.masks_raw_dir(pid, vid)

        # If resuming from a paused frame, pre-load that frame's masks as the
        # initial seed so the first batch behaves like a continuation batch.
        prev_batch_last_masks: dict | None = None
        if seed_frame >= 0:
            seed_npz = masks_dir / f"{seed_frame:06d}.npz"
            if seed_npz.exists():
                try:
                    saved = load_masks_npz(str(seed_npz))
                    prev_batch_last_masks = {int(k): v.astype(np.uint8) for k, v in saved.items()}
                    # Pre-seed tracker trajectories so get_adaptive_seeds can use the
                    # seed-frame masks instead of falling back to original annotations.
                    # Without this, last_clean_mask is None (fresh tracker) and
                    # get_adaptive_seeds(0.0) would always pick "annotations" mode.
                    for obj_id_int, mask in prev_batch_last_masks.items():
                        if obj_id_int in tracker.trajectories:
                            traj = tracker.trajectories[obj_id_int]
                            traj.last_clean_mask = mask.copy()
                            traj.last_clean_frame = seed_frame
                    logger.info(f"Resume: seeding from frame {seed_frame} masks ({len(prev_batch_last_masks)} objects)")
                except Exception as e:
                    logger.warning(f"Resume: could not load seed masks from frame {seed_frame}: {e}")

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_len = batch_end - batch_start

            # ── 1. Yield "extracting" event ───────────────────────────────
            yield {
                "event": "batch_start",
                "data": json.dumps({
                    "batch": batch_idx,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "total_batches": len(batches),
                    "status": "extracting",
                }),
            }

            # ── 2. Extract this batch into a fresh temp directory ─────────
            tmp_dir = tempfile.mkdtemp(prefix=f"sam3wt_{pid}_{vid}_b{batch_idx}_")
            try:
                try:
                    # Track extraction progress in a shared mutable container
                    _extract_state = {"extracted": 0}
                    def _progress_cb(extracted: int, _total: int) -> None:
                        _extract_state["extracted"] = extracted

                    future = loop.run_in_executor(
                        None,
                        lambda: extract_frame_range(
                            source_path, tmp_dir, batch_start, batch_end, _progress_cb,
                        ),
                    )

                    # Poll and emit extraction progress while the thread runs
                    _last_reported = -1
                    while not future.done():
                        await asyncio.sleep(0.25)
                        _current = _extract_state["extracted"]
                        if _current != _last_reported:
                            _last_reported = _current
                            yield {
                                "event": "extract_progress",
                                "data": json.dumps({
                                    "extracted": _current,
                                    "batch_start": batch_start,
                                    "batch_end": batch_end,
                                    "frames_to_process": frames_to_process,
                                }),
                            }
                    await future  # propagate any exception from the thread
                except Exception as e:
                    logger.error(f"Frame extraction failed for batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": f"Frame extraction failed: {e}"})}
                    return

                # ── 3. Yield "initializing" event ─────────────────────────
                yield {
                    "event": "batch_start",
                    "data": json.dumps({
                        "batch": batch_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "total_batches": len(batches),
                        "status": "initializing_session",
                    }),
                }

                # ── 4. Init SAM session on just this batch's temp dir ─────
                try:
                    new_sid = await loop.run_in_executor(None, sam.init_session, pid, vid, tmp_dir)
                    pm.update_video(pid, vid, {"sam3_session_id": new_sid})
                except Exception as e:
                    logger.error(f"Session init failed for batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": f"Session init failed: {e}"})}
                    return

                # ── 5. Add prompts ────────────────────────────────────────
                try:
                    if prev_batch_last_masks is None:
                        # First batch that has frame data: replay point prompts
                        # for any annotated frame that falls within this batch.
                        replayed_count = 0
                        for obj_id_str, frame_map in all_prompts.items():
                            for fidx_str, prompt in frame_map.items():
                                real_fidx = int(fidx_str)
                                if batch_start <= real_fidx < batch_end:
                                    logger.info(
                                        f"Batch {batch_idx}: replaying prompts for "
                                        f"obj {obj_id_str} on frame {real_fidx} "
                                        f"(points={len(prompt['points'])})"
                                    )
                                    await loop.run_in_executor(None, functools.partial(
                                        sam.add_points, pid, vid,
                                        frame_idx=real_fidx,
                                        obj_id=int(obj_id_str),
                                        points=prompt["points"],
                                        labels=prompt["labels"],
                                        text=None,  # SAM3 tracker mode doesn't support text
                                    ))
                                    replayed_count += 1
                                    if sam3_annotated_frame is None or real_fidx < sam3_annotated_frame:
                                        sam3_annotated_frame = real_fidx

                        if replayed_count == 0 and all_prompts:
                            # No annotated frames fall within this batch's range.
                            # This happens when the user requests tracking from a
                            # frame beyond all annotations with no prior propagation
                            # (so there's no seed mask at batch_start - 1 either).
                            # Fall back: replay every object's latest prompt at
                            # batch_start so SAM has seed points to work from.
                            logger.warning(
                                f"Batch {batch_idx}: no prompts in range "
                                f"[{batch_start}, {batch_end}); replaying all "
                                f"annotation prompts at frame {batch_start} as fallback."
                            )
                            for obj_id_str, frame_map in all_prompts.items():
                                if not frame_map:
                                    continue
                                # Use the prompt from the highest annotated frame
                                # (closest to where we're starting from).
                                latest_fidx_str = max(frame_map, key=lambda k: int(k))
                                prompt = frame_map[latest_fidx_str]
                                await loop.run_in_executor(None, functools.partial(
                                    sam.add_points, pid, vid,
                                    frame_idx=batch_start,
                                    obj_id=int(obj_id_str),
                                    points=prompt["points"],
                                    labels=prompt["labels"],
                                    text=None,
                                ))
                                replayed_count += 1
                            sam3_annotated_frame = batch_start

                        logger.info(
                            f"Batch {batch_idx}: replayed {replayed_count} prompt(s), "
                            f"annotated_frame={sam3_annotated_frame}"
                        )
                        # Verify the cache was populated
                        if is_sam3 and replayed_count > 0:
                            try:
                                sid = sam.get_session_id(pid, vid)
                                state = _get_predictor()._ALL_INFERENCE_STATES[sid]["state"]
                                cached = state.get("cached_frame_outputs", {})
                                logger.info(
                                    f"Batch {batch_idx}: cached_frame_outputs has "
                                    f"{len(cached)} frame(s): {list(cached.keys())[:10]}"
                                )
                            except Exception as dbg_e:
                                logger.warning(f"Debug cache check failed: {dbg_e}")
                    else:
                        # Subsequent batches: seed from the best available mask for
                        # each object.  Merge two sources (disk wins only when
                        # in-memory is absent):
                        #   1. Saved .npz for the last frame of the previous batch —
                        #      acts as a fallback for objects SAM temporarily lost
                        #      near the end of that batch (so they didn't end up in
                        #      prev_batch_last_masks / last_frame_masks).
                        #   2. In-memory prev_batch_last_masks — more recent, so it
                        #      overrides the disk value when present.
                        seed_masks: dict = {}
                        last_prev_frame = batch_start - 1
                        prev_npz = masks_dir / f"{last_prev_frame:06d}.npz"
                        if prev_npz.exists():
                            try:
                                saved = load_masks_npz(str(prev_npz))
                                seed_masks = {int(k): v.astype(np.uint8) for k, v in saved.items()}
                            except Exception as _e:
                                logger.warning(
                                    f"Batch {batch_idx}: could not load disk masks "
                                    f"for frame {last_prev_frame}: {_e}"
                                )
                        # In-memory carry-forward overrides disk (it's more recent)
                        seed_masks.update(prev_batch_last_masks)

                        # ── Adaptive seeding based on confusion score ──────
                        # Compute max confusion over the just-completed batch
                        last_prev_frame_scores = [
                            tracker.confusion_scores.get(f, 0.0)
                            for f in range(max(0, batch_start - STREAM_BATCH_SIZE), batch_start)
                            if f in tracker.confusion_scores
                        ]
                        batch_confusion = float(np.max(last_prev_frame_scores)) if last_prev_frame_scores else 0.0
                        adaptive_strategy = tracker.get_adaptive_seeds(batch_confusion)
                        logger.info(
                            f"Batch {batch_idx}: confusion={batch_confusion:.3f}, "
                            f"seed strategy modes: "
                            f"{[v[0] for v in adaptive_strategy.values()]}"
                        )

                        seeded_count = 0
                        for obj_id_int, strategy in adaptive_strategy.items():
                            obj_text = pm.get_object_description(pid, vid, str(obj_id_int)) or None
                            mode = strategy[0]

                            if mode == "mask":
                                clean_mask = strategy[1]
                                await loop.run_in_executor(None, functools.partial(
                                    sam.add_mask_prompt, pid, vid,
                                    frame_idx=batch_start,
                                    obj_id=obj_id_int,
                                    mask=clean_mask,
                                    text=obj_text,
                                ))
                                seeded_count += 1

                            elif mode == "both":
                                _, clean_mask, seed_pts = strategy
                                await loop.run_in_executor(None, functools.partial(
                                    sam.add_mask_prompt, pid, vid,
                                    frame_idx=batch_start,
                                    obj_id=obj_id_int,
                                    mask=clean_mask,
                                    text=None,  # SAM3 tracker mode doesn't support text
                                ))
                                for (fidx, pts, lbls) in seed_pts:
                                    await loop.run_in_executor(None, functools.partial(
                                        sam.add_points, pid, vid,
                                        frame_idx=batch_start,
                                        obj_id=obj_id_int,
                                        points=pts,
                                        labels=lbls,
                                        text=None,  # SAM3 tracker mode doesn't support text
                                    ))
                                seeded_count += 1

                            elif mode == "annotations":
                                _, seed_pts = strategy
                                for (fidx, pts, lbls) in seed_pts:
                                    await loop.run_in_executor(None, functools.partial(
                                        sam.add_points, pid, vid,
                                        frame_idx=batch_start,
                                        obj_id=obj_id_int,
                                        points=pts,
                                        labels=lbls,
                                        text=None,  # SAM3 tracker mode doesn't support text
                                    ))
                                seeded_count += 1

                        # Fallback: for objects not covered by adaptive strategy,
                        # use their last available mask from seed_masks
                        for obj_id_int, mask_np in seed_masks.items():
                            if obj_id_int not in adaptive_strategy:
                                obj_text = pm.get_object_description(pid, vid, str(obj_id_int)) or None
                                await loop.run_in_executor(None, functools.partial(
                                    sam.add_mask_prompt, pid, vid,
                                    frame_idx=batch_start,
                                    obj_id=obj_id_int,
                                    mask=mask_np,
                                    text=obj_text,
                                ))
                                seeded_count += 1

                        logger.info(
                            f"Batch {batch_idx}: seeded {seeded_count} object(s) "
                            f"on frame {batch_start} "
                            f"(confusion={batch_confusion:.3f}, "
                            f"in-memory: {len(prev_batch_last_masks)}, "
                            f"disk-backed: {prev_npz.exists()})"
                        )
                except Exception as e:
                    logger.error(f"Prompt setup failed for batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": f"Prompt setup failed: {e}"})}
                    return

                # ── 6. Run propagation ────────────────────────────────────
                queue: asyncio.Queue = asyncio.Queue(maxsize=50)
                # Initialise from previous batch so objects that SAM temporarily
                # loses near the end of a batch still have a carry-forward mask.
                last_frame_masks: dict = dict(prev_batch_last_masks) if prev_batch_last_masks else {}
                batch_frame_count = 0

                # Always propagate forward.  For the first batch, start from
                # the annotated frame (not frame 0) — SAM needs the prompt
                # cache at the start frame.  Frames before the annotated
                # frame in this batch won't get masks, which is fine since
                # nothing was annotated there.
                # Subsequent batches start from batch_start (seeded via masks).
                if batch_idx == 0 and sam3_annotated_frame is not None:
                    _start = sam3_annotated_frame
                else:
                    _start = batch_start
                _direction = "forward"

                def _run_propagation(
                    b_start=batch_start, b_len=batch_len,
                    s3_start=_start, direction=_direction,
                ):
                    try:
                        prop_iter = sam.propagate_stream(
                            pid, vid,
                            start_frame_idx=s3_start,
                            max_frame_num_to_track=b_len if not is_sam3 else None,
                            propagation_direction=direction,
                        )
                        for item in prop_iter:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("frame", item)), loop
                            ).result()
                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("error", exc)), loop
                        ).result()
                    finally:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("done", None)), loop
                        ).result()

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                loop.run_in_executor(executor, _run_propagation)

                try:
                    while True:
                        msg_type, payload = await queue.get()
                        if msg_type == "error":
                            raise payload
                        if msg_type == "done":
                            break

                        # ── Translate SAM's internal sequential index → real frame index
                        item = payload
                        sam_idx = item.get("frame_index", batch_frame_count)
                        real_frame_idx = sam.to_real_idx(pid, vid, sam_idx)

                        frame_outputs = item.get("outputs", {})
                        if isinstance(frame_outputs, dict):
                            # SAM3 keys outputs by its internal index; try both forms
                            fo = (frame_outputs.get(sam_idx)
                                  or frame_outputs.get(str(sam_idx))
                                  or frame_outputs.get(real_frame_idx)
                                  or frame_outputs.get(str(real_frame_idx))
                                  or frame_outputs)
                        else:
                            fo = {}

                        obj_ids = fo.get("out_obj_ids", [])
                        masks_raw = fo.get("out_binary_masks", [])
                        boxes_raw = fo.get("out_boxes_xywh", [])
                        probs_raw = fo.get("out_probs", [])

                        # Build raw masks dict from SAM output
                        sam_output_masks: dict[int, np.ndarray] = {}
                        bboxes_to_save: dict = {}

                        for i, oid in enumerate(obj_ids):
                            if i < len(masks_raw):
                                mask = masks_raw[i]
                                if hasattr(mask, "numpy"):
                                    mask = mask.numpy()
                                mask = np.squeeze(mask).astype(np.uint8)
                                sam_output_masks[int(oid)] = mask
                            if i < len(boxes_raw):
                                box = boxes_raw[i]
                                if hasattr(box, "tolist"):
                                    box = box.tolist()
                                score = float(probs_raw[i]) if i < len(probs_raw) else 1.0
                                bboxes_to_save[str(oid)] = list(box) + [score]

                        # ── Temporal Consistency Validation ────────────────
                        # Filter masks that are temporally inconsistent (big jumps
                        # in position/size/IoU) and fall back to previous frame
                        
                        # Refresh tracking params from video config (allows real-time updates)
                        fresh_video = pm.get_video(pid, vid)
                        if fresh_video:
                            fresh_params = fresh_video.get("tracking_params", {})
                            tracker.update_tracking_params(fresh_params)
                        
                        is_first_frame = (real_frame_idx == batch_start)
                        _fshape = (
                            (fresh_video["height"], fresh_video["width"])
                            if fresh_video else None
                        )
                        display_masks, tracking_masks, rejections = tracker.validate_and_filter_masks(
                            real_frame_idx,
                            sam_output_masks,
                            force_accept=is_first_frame,  # Accept first frame of batch
                            frame_shape=_fshape,
                        )
                        if rejections:
                            logger.debug(f"Frame {real_frame_idx} rejections: {rejections}")

                        # display_masks → saved to disk (SAM output; shows what the model saw)
                        # tracking_masks → carry-forward state (clean masks during confusion)
                        masks_to_save = {str(k): v for k, v in display_masks.items()}

                        # Use tracking_masks (not display_masks) for next-frame seeding so
                        # overlap-confused predictions don't poison future tracking state.
                        for _oid, _m in tracking_masks.items():
                            if _m.any():
                                last_frame_masks[_oid] = _m

                        # Identity tracker sees display masks (what SAM produced)
                        current_frame_masks = display_masks

                        if masks_to_save:
                            masks_path = masks_dir / f"{real_frame_idx:06d}.npz"
                            save_masks_npz(str(masks_path), masks_to_save)
                            # Save to masks_raw/ only if it doesn't exist yet
                            # (preserve the very first propagation pass)
                            raw_path = masks_raw_dir / f"{real_frame_idx:06d}.npz"
                            if not raw_path.exists():
                                save_masks_npz(str(raw_path), masks_to_save)
                        if bboxes_to_save:
                            save_bboxes_json(str(bboxes_dir / f"{real_frame_idx:06d}.json"), bboxes_to_save)

                        # ── Update IdentityTracker ────────────────────────
                        tracker.update_frame(
                            real_frame_idx,
                            current_frame_masks,
                            {int(k): v for k, v in bboxes_to_save.items()},
                        )
                        confusion_score = tracker.confusion_scores.get(real_frame_idx, 0.0)

                        # Collect seed-frame masks (earliest annotated frames)
                        if sam3_annotated_frame is not None and real_frame_idx == sam3_annotated_frame:
                            seed_frame_masks = {int(k): v for k, v in current_frame_masks.items()}

                        # Collect post-window masks for swap detection
                        confusion_windows_so_far = tracker.get_confusion_windows()
                        for w_idx, window in enumerate(confusion_windows_so_far):
                            w_end = window["end"]
                            # Capture masks from frames just after window ends
                            if w_end < real_frame_idx <= w_end + N_POST_WINDOW_FRAMES:
                                if w_idx not in post_window_masks_by_window:
                                    post_window_masks_by_window[w_idx] = {}
                                for oid_int, m in current_frame_masks.items():
                                    if oid_int not in post_window_masks_by_window[w_idx]:
                                        post_window_masks_by_window[w_idx][oid_int] = m

                        pm.mark_frame_propagated(pid, vid, real_frame_idx)
                        batch_frame_count += 1
                        total_propagated += 1
                        progress = total_propagated / num_frames if num_frames > 0 else 1.0

                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "frame": real_frame_idx,
                                "progress": round(progress, 4),
                                "done": False,
                                "batch": batch_idx,
                                "batch_start": batch_start,
                                "batch_end": batch_end,
                                "obj_ids": [str(o) for o in obj_ids],
                                "uncertainty_score": round(confusion_score, 4),
                            }),
                        }

                except Exception as e:
                    logger.error(f"Propagation error in batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return

                # ── 7. Carry masks forward for the next batch ─────────────
                # Invalidate encode cache so scrubbing after propagation gets fresh masks
                _invalidate_mask_cache(pid, vid)
                prev_batch_last_masks = last_frame_masks if last_frame_masks else prev_batch_last_masks
                logger.info(
                    f"Batch {batch_idx} done: {batch_frame_count} frames, "
                    f"{len(last_frame_masks)} object mask(s) carried forward"
                )

            finally:
                # Always delete the temp directory for this batch
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── All batches done ──────────────────────────────────────────────
        pm.mark_propagation_complete(pid, vid)

        # Save uncertainty data
        try:
            confusion_windows = tracker.get_confusion_windows()
            uncertainty_data = tracker.build_uncertainty_json(confusion_windows)
            pm.save_uncertainty(pid, vid, uncertainty_data)
            logger.info(
                f"Uncertainty saved: {len(confusion_windows)} confusion window(s), "
                f"{len(uncertainty_data.get('per_frame', {}))} frames tracked"
            )
        except Exception as ue:
            logger.warning(f"Failed to save uncertainty data: {ue}")
            confusion_windows = []

        # Build and save correction records (identity swap detection)
        try:
            correction_records = tracker.build_correction_records(
                confusion_windows,
                post_window_masks_by_window,
                seed_frame_masks,
            )
            for record in correction_records:
                pm.save_correction(pid, vid, record)
            if correction_records:
                logger.info(f"Created {len(correction_records)} pending correction record(s)")
        except Exception as ce:
            logger.warning(f"Failed to build correction records: {ce}")

        # ── Retroactive ghost mask corrections (occlusion windows) ────────────
        try:
            correction_events = []
            for pair, window in tracker.occlusion_windows.items():
                # Attempt final resolution if not yet done
                if window.identity_map is None:
                    tracker.resolve_post_overlap_identity(pair)
                if window.identity_map is not None and not window.corrected:
                    corrections = tracker.get_retroactive_corrections(pair)
                    for frame_idx_c, corrected_masks in corrections:
                        c_path = masks_dir / f"{frame_idx_c:06d}.npz"
                        save_masks_npz(str(c_path), {str(k): v for k, v in corrected_masks.items()})
                    if corrections:
                        _invalidate_mask_cache(pid, vid)
                        a, b = pair
                        correction_events.append({
                            "pair": [str(a), str(b)],
                            "onset_frame": window.onset_frame,
                            "end_frame": window.end_frame,
                            "corrected_frames": [f for f, _ in corrections],
                            "swap_detected": bool(window.identity_map),
                        })
                        logger.info(
                            f"Retroactive correction: pair {pair}, "
                            f"{len(corrections)} frames rewritten, "
                            f"swap={bool(window.identity_map)}"
                        )
            for ev in correction_events:
                yield {"event": "correction", "data": json.dumps(ev)}
        except Exception as corr_err:
            logger.warning(f"Retroactive correction failed: {corr_err}")

        yield {
            "event": "done",
            "data": json.dumps({
                "frame": max(0, total_propagated - 1),
                "progress": 1.0,
                "done": True,
                "total_frames": total_propagated,
                "confusion_windows": len(confusion_windows) if 'confusion_windows' in dir() else 0,
            }),
        }

    try:
        async for event in run():
            await state.publish(event)
    except Exception as e:
        logger.error(f"Propagation background task crashed: {e}", exc_info=True)
        await state.publish({"event": "error", "data": json.dumps({"error": str(e)})})
    finally:
        state.is_running = False


# ─── Propagation SSE endpoint ─────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/propagate")
async def propagate_video(pid: str, vid: str, start_frame: int = 0, resume_from: int = -1):
    """
    Stream propagation results as Server-Sent Events.

    Propagation runs as a background task so client reconnects join the
    existing run rather than restarting it from scratch.

    resume_from: if >= 0, treat this as a resume from the given frame.
      The first batch will seed from the saved mask at resume_from rather
      than replaying point prompts (like subsequent batches normally do).
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
        session_id = sam.get_session_id(pid, vid)
        if session_id is None:
            state.unsubscribe(q)
            logger.error(
                f"GET /propagate: no active SAM session for {pid}/{vid}. "
                "Call POST /session first."
            )
            raise HTTPException(400, "No active SAM session. Initialize session and add prompts first.")

        num_frames = video["num_frames"]
        source_path = video.get("source_path", "")
        masks_dir = pm.masks_dir(pid, vid)
        bboxes_dir = pm.bboxes_dir(pid, vid)
        objects = video["objects"]
        is_sam3 = sam.model_name() == "sam3"

        all_prompts = pm.get_all_point_prompts(pid, vid)

        if resume_from >= 0:
            # Resume mode: start from the frame after the paused frame,
            # seeding from the saved mask at resume_from (like a mid-stream batch).
            actual_start = resume_from + 1
            seed_frame = resume_from
        else:
            # Fresh start: find the earliest annotated frame and begin there.
            first_ann_frame: int | None = None
            for obj_id_str, frame_map in all_prompts.items():
                for fidx_str in frame_map:
                    fidx = int(fidx_str)
                    if first_ann_frame is None or fidx < first_ann_frame:
                        first_ann_frame = fidx
            actual_start = min(start_frame, first_ann_frame) if first_ann_frame is not None else start_frame
            seed_frame = -1

        batches: list[tuple[int, int]] = []
        for s in range(actual_start, num_frames, STREAM_BATCH_SIZE):
            batches.append((s, min(s + STREAM_BATCH_SIZE, num_frames)))

        state.total_frames = num_frames
        state.start_frame = actual_start
        state.is_running = True
        state.is_paused = False
        state.task = asyncio.create_task(
            _run_propagation_bg(
                pid, vid, actual_start, batches, all_prompts,
                num_frames, source_path, masks_dir, bboxes_dir,
                objects, is_sam3, state,
                seed_frame=seed_frame,
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
                "start_frame": state.start_frame,
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
    masks_raw_dir = pm.masks_raw_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)
    
    deleted_count = 0
    if req.clear_from_frame:
        # Delete all masks/bboxes from resume_frame onwards
        for d in (masks_dir, masks_raw_dir, bboxes_dir):
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


# ─── Swap object masks for a frame ────────────────────────────────────────────

class SwapMasksRequest(BaseModel):
    obj_id_a: str
    obj_id_b: str
    swap_mode: str = "this_frame"  # "this_frame" or "all_future"


@app.post("/api/projects/{pid}/videos/{vid}/masks/{fidx}/swap")
def swap_masks(pid: str, vid: str, fidx: int, req: SwapMasksRequest):
    """
    Swap the saved masks of two objects for this frame only.
    For all_future mode, use the SSE endpoint /masks/{fidx}/swap_stream instead.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    if req.swap_mode not in ("this_frame", "all_future"):
        raise HTTPException(400, f"Invalid swap_mode: {req.swap_mode}")

    masks_dir_path = pm.masks_dir(pid, vid)

    # For all_future, only do this_frame here; bulk swap is handled by swap_stream SSE
    frames_to_swap = [fidx]

    swapped_count = 0
    first_frame_masks = None

    for frame_idx in frames_to_swap:
        masks_path = masks_dir_path / f"{frame_idx:06d}.npz"
        if not masks_path.exists():
            continue

        raw = load_masks_npz(str(masks_path))
        key_a = req.obj_id_a
        key_b = req.obj_id_b

        if key_a not in raw and key_b not in raw:
            continue

        mask_a = raw.get(key_a)
        mask_b = raw.get(key_b)
        updated = dict(raw)

        if mask_a is not None:
            updated[key_b] = mask_a
        elif key_b in updated:
            del updated[key_b]
        if mask_b is not None:
            updated[key_a] = mask_b
        elif key_a in updated:
            del updated[key_a]

        save_masks_npz(str(masks_path), updated)
        swapped_count += 1

        if frame_idx == fidx:
            first_frame_masks = updated

    _invalidate_mask_cache(pid, vid)

    objects = video.get("objects", {})
    mask_b64: dict[str, str] = {}
    if first_frame_masks:
        for obj_id, mask in first_frame_masks.items():
            color = _get_instance_color(str(obj_id), objects)
            mask_b64[str(obj_id)] = encode_mask_as_png(mask, color)

    logger.info(f"Swapped {req.obj_id_a} <-> {req.obj_id_b} for frame {fidx}")

    return {
        "frame_idx": fidx,
        "masks": mask_b64,
        "swapped_frames": swapped_count,
        "swap_mode": req.swap_mode,
    }


@app.get("/api/projects/{pid}/videos/{vid}/masks/{fidx}/swap_stream")
async def swap_masks_stream(pid: str, vid: str, fidx: int, obj_id_a: str, obj_id_b: str):
    """
    SSE endpoint: swap obj_id_a <-> obj_id_b for frame fidx and all subsequent frames.
    Streams progress events so the client can show a progress bar.
    Events:
      progress  {"done": N, "total": M, "swapped": K}
      done      {"swapped_frames": K, "frame_idx": fidx, "masks": {...base64...}}
      error     {"error": "..."}
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    async def event_gen():
        import concurrent.futures
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            try:
                masks_dir_path = pm.masks_dir(pid, vid)
                total_frames = int(video.get("num_frames", 0))
                frames_to_process = list(range(fidx, total_frames))
                total = len(frames_to_process)
                swapped_count = 0
                first_frame_masks = None
                CHUNK = 200

                for chunk_start in range(0, len(frames_to_process), CHUNK):
                    chunk = frames_to_process[chunk_start:chunk_start + CHUNK]
                    for frame_idx in chunk:
                        masks_path = masks_dir_path / f"{frame_idx:06d}.npz"
                        if not masks_path.exists():
                            continue
                        raw = load_masks_npz(str(masks_path))
                        if obj_id_a not in raw and obj_id_b not in raw:
                            continue
                        mask_a = raw.get(obj_id_a)
                        mask_b = raw.get(obj_id_b)
                        updated = dict(raw)
                        if mask_a is not None:
                            updated[obj_id_b] = mask_a
                        elif obj_id_b in updated:
                            del updated[obj_id_b]
                        if mask_b is not None:
                            updated[obj_id_a] = mask_b
                        elif obj_id_a in updated:
                            del updated[obj_id_a]
                        save_masks_npz(str(masks_path), updated)
                        swapped_count += 1
                        if frame_idx == fidx:
                            first_frame_masks = updated

                    done_so_far = min(chunk_start + CHUNK, total)
                    logger.info(
                        f"[swap_stream] {obj_id_a}<->{obj_id_b}: "
                        f"{done_so_far}/{total} frames scanned, {swapped_count} swapped"
                    )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("progress", {"done": done_so_far, "total": total, "swapped": swapped_count})),
                        loop,
                    ).result()

                _invalidate_mask_cache(pid, vid)

                objects_cfg = video.get("objects", {})
                mask_b64: dict[str, str] = {}
                if first_frame_masks:
                    for obj_id, mask in first_frame_masks.items():
                        color = _get_instance_color(str(obj_id), objects_cfg)
                        mask_b64[str(obj_id)] = encode_mask_as_png(mask, color)

                logger.info(
                    f"[swap_stream] complete: {obj_id_a}<->{obj_id_b}, "
                    f"{swapped_count} frames swapped (started at {fidx})"
                )
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", {
                        "swapped_frames": swapped_count,
                        "frame_idx": fidx,
                        "masks": mask_b64,
                    })),
                    loop,
                ).result()
            except Exception as exc:
                logger.error(f"[swap_stream] failed: {exc}", exc_info=True)
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


# ─── Saved Masks (raw / pre-correction) ───────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/masks/{fidx}/raw")
def get_raw_mask(pid: str, vid: str, fidx: int):
    """
    Return pre-correction masks from masks_raw/ directory.
    Used by TrackCorrectionPanel to show the before/after view.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    masks_raw_dir = pm.masks_raw_dir(pid, vid)
    masks_path = masks_raw_dir / f"{fidx:06d}.npz"
    if not masks_path.exists():
        # Fallback to current masks if no raw version exists
        masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
        if not masks_path.exists():
            return JSONResponse({"masks": {}})

    mask_b64 = _get_encoded_masks(masks_path, video["objects"])
    return JSONResponse({"frame_idx": fidx, "masks": mask_b64})


# ─── Uncertainty ──────────────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/uncertainty")
def get_uncertainty(pid: str, vid: str):
    """Return the uncertainty data (confusion scores, windows, similarity matrix)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return pm.load_uncertainty(pid, vid)


@app.get("/api/projects/{pid}/videos/{vid}/uncertainty/frames/{fidx}")
def get_uncertainty_frame_detail(pid: str, vid: str, fidx: int):
    """
    Get detailed uncertainty info for a specific frame, including:
    - The frame image (base64 JPEG)
    - Mask overlay (base64 RGBA PNG)
    - Per-object uncertainty scores for that frame
    
    This is lazily loaded when the user clicks a frame in the uncertainty heatmap.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    
    # Get uncertainty data
    uncertainty = pm.load_uncertainty(pid, vid)
    per_frame = uncertainty.get("per_frame", {})
    frame_uncertainty = per_frame.get(str(fidx), {})
    
    # Load frame image
    ann_dir = pm.annotated_frames_dir(pid, vid)
    frame_path = ann_dir / f"{fidx:06d}.jpg"
    frame_b64 = None
    
    if frame_path.exists():
        with open(frame_path, "rb") as f:
            frame_b64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        # Try extracting the frame
        source_path = video.get("source_path", "")
        if source_path and Path(source_path).exists():
            try:
                extract_frame_range(source_path, str(ann_dir), fidx, fidx + 1)
                if frame_path.exists():
                    with open(frame_path, "rb") as f:
                        frame_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    
    # Load masks
    masks_path = pm.masks_dir(pid, vid) / f"{fidx:06d}.npz"
    masks_b64 = {}
    if masks_path.exists():
        masks_b64 = _get_encoded_masks(masks_path, video["objects"])
    
    # Get per-object scores for this frame
    per_object = frame_uncertainty.get("per_object", {})
    
    return {
        "frame_idx": fidx,
        "frame_image": frame_b64,
        "masks": masks_b64,
        "confusion_score": frame_uncertainty.get("confusion_score", 0),
        "per_object": per_object,
    }


# ─── Identity Corrections ─────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/videos/{vid}/corrections")
def list_corrections(pid: str, vid: str):
    """List all identity correction records for a video."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return pm.load_corrections(pid, vid)


class ApplyCorrectionRequest(BaseModel):
    method: str = "swap"   # "swap" or "repropagate"


@app.post("/api/projects/{pid}/videos/{vid}/corrections/{correction_id}/apply")
async def apply_correction(pid: str, vid: str, correction_id: str, req: ApplyCorrectionRequest):
    """
    Apply an identity correction.
    method=swap: swap mask/bbox keys for obj_id_a ↔ obj_id_b in [swap_onset, window_end].
    method=repropagate: re-propagate the window with corrected seeds (background task).
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    record = pm.load_correction(pid, vid, correction_id)
    if record is None:
        raise HTTPException(404, f"Correction {correction_id} not found")
    if record["status"] != "pending":
        raise HTTPException(400, f"Correction already {record['status']}")

    obj_id_a = record["obj_id_a"]
    obj_id_b = record["obj_id_b"]
    swap_onset = record["swap_onset"]
    window_end = record["window_end"]

    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)

    if req.method == "swap":
        affected_frames = []
        for fidx in range(swap_onset, window_end + 1):
            # Swap masks
            masks_path = masks_dir / f"{fidx:06d}.npz"
            if masks_path.exists():
                raw = load_masks_npz(str(masks_path))
                key_a = obj_id_a
                key_b = obj_id_b
                mask_a = raw.get(key_a)
                mask_b = raw.get(key_b)
                updated = dict(raw)
                if mask_a is not None:
                    updated[key_b] = mask_a
                elif key_b in updated:
                    del updated[key_b]
                if mask_b is not None:
                    updated[key_a] = mask_b
                elif key_a in updated:
                    del updated[key_a]
                save_masks_npz(str(masks_path), updated)
                affected_frames.append(fidx)

            # Swap bboxes
            bboxes_path = bboxes_dir / f"{fidx:06d}.json"
            if bboxes_path.exists():
                bboxes = load_bboxes_json(str(bboxes_path))
                bbox_a = bboxes.get(obj_id_a)
                bbox_b = bboxes.get(obj_id_b)
                if bbox_a is not None:
                    bboxes[obj_id_b] = bbox_a
                if bbox_b is not None:
                    bboxes[obj_id_a] = bbox_b
                save_bboxes_json(str(bboxes_path), bboxes)

        _invalidate_mask_cache(pid, vid)
        pm.update_correction(pid, vid, correction_id, {"status": "applied", "method": "swap"})
        logger.info(f"Correction {correction_id} applied (swap): {len(affected_frames)} frames")
        return {"status": "applied", "method": "swap", "affected_frames": len(affected_frames)}

    elif req.method == "repropagate":
        # Re-propagation is a background operation — not implemented in this call
        # (would require a full mini-batch re-propagation infrastructure)
        raise HTTPException(501, "Re-propagation correction not yet implemented in this version")

    else:
        raise HTTPException(400, f"Unknown correction method: {req.method}")


@app.post("/api/projects/{pid}/videos/{vid}/corrections/{correction_id}/reject")
def reject_correction(pid: str, vid: str, correction_id: str):
    """Mark a correction as rejected (no masks changed)."""
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    record = pm.load_correction(pid, vid, correction_id)
    if record is None:
        raise HTTPException(404, f"Correction {correction_id} not found")
    pm.update_correction(pid, vid, correction_id, {"status": "rejected"})
    return {"status": "rejected"}
