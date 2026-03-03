"""
SAM Web Tracker — FastAPI backend server.
Supports SAM3 (primary) with SAM2 fallback.
"""

import asyncio
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

BASE_PROJECTS = Path("/opt/.sam3_projects")

# Track model loading state
_model_loading = False
_model_load_error: str | None = None


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
def health():
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
    perm_path = BASE_PROJECTS / pid / "videos" / vid / f"source{suffix}"
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
    perm_path = BASE_PROJECTS / pid / "videos" / vid / f"source{src.suffix}"
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
      - Saved masks (masks/*.npz) and bboxes (bboxes/*.json)
      - Annotated frames (annotated_frames/)
      - Preview frames (frames/)
      - Active SAM session
      - propagation_complete / propagated_frames flags
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    # Close any active SAM session
    sam.close_session(pid, vid)

    # Wipe masks and bboxes directories
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
        "sam3_session_id": None,
        "propagated_frames": [],
        "propagation_complete": False,
        "frames_extracted": False,
        "all_frames_extracted": False,
        "preview_indices": [],
    })

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
            return StreamingResponse(iter([data]), media_type="image/jpeg")
    return FileResponse(str(frame_path), media_type="image/jpeg")


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


@app.post("/api/projects/{pid}/videos/{vid}/objects", status_code=201)
def add_object(pid: str, vid: str, req: AddObjectRequest):
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")
    return pm.add_object(pid, vid, req.name, req.color)


class RenameObjectRequest(BaseModel):
    name: str


@app.patch("/api/projects/{pid}/videos/{vid}/objects/{oid}")
def rename_object(pid: str, vid: str, oid: str, req: RenameObjectRequest):
    pm.rename_object(pid, vid, oid, req.name)
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
    points: list   # [[x_norm, y_norm], ...]
    labels: list   # [1, 0, ...]


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
                )
                logger.debug(f"Replayed prompts for obj {replay_obj_id} on frame {replay_fidx}")
            except Exception as e:
                logger.warning(f"Failed to replay prompts for obj {replay_obj_id} frame {replay_fidx}: {e}")

    # Save prompts to config
    pm.save_point_prompts(pid, vid, oid, req.frame_idx, req.points, req.labels)

    # Call SAM
    try:
        logger.info(f"add_points: frame={req.frame_idx}, obj={oid}, points={req.points}, labels={req.labels}")
        outputs = sam.add_points(
            pid, vid,
            frame_idx=req.frame_idx,
            obj_id=int(oid),
            points=req.points,
            labels=req.labels,
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
        obj_color = objects.get(str(obj_id), {}).get("color", "#5B8DD9")
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

    # Persist updated masks to disk so they replace any previously propagated
    # masks for this frame.  Merge with existing npz so objects not yet in the
    # current session still retain their saved masks.
    if raw_masks:
        masks_path = pm.masks_dir(pid, vid) / f"{req.frame_idx:06d}.npz"
        existing = load_masks_npz(str(masks_path)) if masks_path.exists() else {}
        save_masks_npz(str(masks_path), {**existing, **raw_masks})

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
        return JSONResponse({"masks": {}})

    raw_masks = load_masks_npz(str(masks_path))
    objects = video["objects"]

    # Encode each object mask
    mask_b64: dict[str, str] = {}
    for obj_id, mask in raw_masks.items():
        obj_color = objects.get(str(obj_id), {}).get("color", "#5B8DD9")
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

    return {"frame_idx": fidx, "masks": mask_b64}


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
    out_path = BASE_PROJECTS / pid / "videos" / vid / "export.mp4"

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


# ─── Propagation (SSE) — streaming mini-batch frame processing ───────────────

@app.get("/api/projects/{pid}/videos/{vid}/propagate")
async def propagate_video(pid: str, vid: str, start_frame: int = 0):
    """
    Stream propagation results as Server-Sent Events.

    Frames are decoded from the source video in mini-batches of STREAM_BATCH_SIZE
    into a *temporary directory* that is deleted after each batch.  This means
    SAM only loads ~150 frames per session init (seconds) instead of the entire
    video (potentially tens of minutes).

    Cross-batch identity tracking:
      • At the end of batch N the last frame's per-object masks are kept.
      • Batch N+1 re-initialises a fresh SAM session on the new temp dir, then
        seeds it with those masks via add_mask_prompt so object IDs carry over.
    """
    video = pm.get_video(pid, vid)
    if video is None:
        raise HTTPException(404, "Video not found")

    session_id = sam.get_session_id(pid, vid)
    if session_id is None:
        raise HTTPException(400, "No active SAM session. Initialize session and add prompts first.")

    num_frames = video["num_frames"]
    source_path = video.get("source_path", "")
    masks_dir = pm.masks_dir(pid, vid)
    bboxes_dir = pm.bboxes_dir(pid, vid)
    objects = video["objects"]
    is_sam3 = sam.model_name() == "sam3"

    # Find the earliest annotated frame so the first batch always contains it.
    all_prompts = pm.get_all_point_prompts(pid, vid)
    first_ann_frame: int | None = None
    for obj_id_str, frame_map in all_prompts.items():
        for fidx_str in frame_map:
            fidx = int(fidx_str)
            if first_ann_frame is None or fidx < first_ann_frame:
                first_ann_frame = fidx
    # Clamp: start no later than the earliest annotated frame
    actual_start = min(start_frame, first_ann_frame) if first_ann_frame is not None else start_frame

    # Build mini-batch ranges starting from actual_start
    batches: list[tuple[int, int]] = []
    for s in range(actual_start, num_frames, STREAM_BATCH_SIZE):
        batches.append((s, min(s + STREAM_BATCH_SIZE, num_frames)))

    import concurrent.futures

    async def event_generator():
        loop = asyncio.get_event_loop()
        total_propagated = 0
        prev_batch_last_masks: dict | None = None  # obj_id(int) -> np.ndarray
        sam3_annotated_frame: int | None = None

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
                    await loop.run_in_executor(
                        None,
                        extract_frame_range,
                        source_path, tmp_dir, batch_start, batch_end,
                    )
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
                    new_sid = sam.init_session(pid, vid, tmp_dir)
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
                                    sam.add_points(
                                        pid, vid,
                                        frame_idx=real_fidx,
                                        obj_id=int(obj_id_str),
                                        points=prompt["points"],
                                        labels=prompt["labels"],
                                    )
                                    replayed_count += 1
                                    if sam3_annotated_frame is None or real_fidx < sam3_annotated_frame:
                                        sam3_annotated_frame = real_fidx
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
                        for obj_id_int, mask_np in seed_masks.items():
                            sam.add_mask_prompt(
                                pid, vid,
                                frame_idx=batch_start,
                                obj_id=obj_id_int,
                                mask=mask_np,
                            )
                        logger.info(
                            f"Batch {batch_idx}: seeded {len(seed_masks)} object(s) "
                            f"on frame {batch_start} "
                            f"(in-memory: {len(prev_batch_last_masks)}, "
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

                        masks_to_save: dict = {}
                        bboxes_to_save: dict = {}
                        current_frame_masks: dict = {}

                        for i, oid in enumerate(obj_ids):
                            if i < len(masks_raw):
                                mask = masks_raw[i]
                                if hasattr(mask, "numpy"):
                                    mask = mask.numpy()
                                mask = np.squeeze(mask).astype(np.uint8)
                                masks_to_save[str(oid)] = mask
                                current_frame_masks[int(oid)] = mask
                            if i < len(boxes_raw):
                                box = boxes_raw[i]
                                if hasattr(box, "tolist"):
                                    box = box.tolist()
                                score = float(probs_raw[i]) if i < len(probs_raw) else 1.0
                                bboxes_to_save[str(oid)] = list(box) + [score]

                        # Per-object update: never replace the whole dict, so an
                        # object absent from a single frame doesn't drop out of the
                        # carry-forward state used to seed the next batch.
                        for _oid, _m in current_frame_masks.items():
                            if _m.any():
                                last_frame_masks[_oid] = _m

                        if masks_to_save:
                            save_masks_npz(str(masks_dir / f"{real_frame_idx:06d}.npz"), masks_to_save)
                        if bboxes_to_save:
                            save_bboxes_json(str(bboxes_dir / f"{real_frame_idx:06d}.json"), bboxes_to_save)

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
                            }),
                        }

                except Exception as e:
                    logger.error(f"Propagation error in batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return

                # ── 7. Carry masks forward for the next batch ─────────────
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
        yield {
            "event": "done",
            "data": json.dumps({
                "frame": max(0, total_propagated - 1),
                "progress": 1.0,
                "done": True,
                "total_frames": total_propagated,
            }),
        }

    return EventSourceResponse(event_generator())
