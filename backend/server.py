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
from sam_predictor import SAMPredictor
from video_processor import (
    encode_mask_as_png,
    ensure_faststart,
    extract_frames,
    extract_preview_frames,
    extract_frame_range,
    get_video_info,
    load_bboxes_json,
    load_masks_npz,
    save_bboxes_json,
    save_masks_npz,
    generate_thumbnail,
    composite_masks_as_png,
)

STREAM_BATCH_SIZE = 150  # frames extracted per streaming mini-batch

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
        try:
            result = extract_preview_frames(str(perm_path), frames_dir, max_preview=80)
            pm.update_video(pid, vid, {
                "num_frames": result["num_frames"],
                "preview_indices": result["preview_indices"],
                "frames_extracted": True,           # preview frames ready
                "all_frames_extracted": False,       # full extraction not done
            })
            logger.info(
                f"Extracted {result['preview_count']} preview frames "
                f"(of {result['num_frames']} total) for {pid}/{vid}"
            )
        except Exception as e:
            logger.error(f"Preview frame extraction failed for {pid}/{vid}: {e}")

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

    # Apply faststart and extract preview frames in background
    frames_dir = str(pm.frames_dir(pid, vid))

    def do_post_import():
        try:
            # Apply faststart so browser can stream the MP4 immediately
            ensure_faststart(str(perm_path))
        except Exception as e:
            logger.warning(f"faststart failed for imported {pid}/{vid}: {e}")
        try:
            result = extract_preview_frames(str(perm_path), frames_dir, max_preview=80)
            pm.update_video(pid, vid, {
                "num_frames": result["num_frames"],
                "preview_indices": result["preview_indices"],
                "frames_extracted": True,
                "all_frames_extracted": False,
            })
            logger.info(
                f"Imported & extracted {result['preview_count']} preview frames "
                f"(of {result['num_frames']} total) for {pid}/{vid}"
            )
        except Exception as e:
            logger.error(f"Preview frame extraction failed for imported {pid}/{vid}: {e}")

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
    frames_dir = str(pm.frames_dir(pid, vid))
    if not Path(frames_dir).exists() or not list(Path(frames_dir).glob("*.jpg")):
        raise HTTPException(400, "Frames not yet extracted. Wait for frame extraction to complete.")
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

    # Ensure SAM session exists (auto-init if needed)
    session_id = sam.get_session_id(pid, vid)
    if session_id is None:
        frames_dir = str(pm.frames_dir(pid, vid))
        if not Path(frames_dir).exists() or not list(Path(frames_dir).glob("*.jpg")):
            raise HTTPException(400, "Frames not yet extracted")
        try:
            session_id = sam.init_session(pid, vid, frames_dir)
            pm.update_video(pid, vid, {"sam3_session_id": session_id})
        except Exception as e:
            raise HTTPException(500, f"SAM session init failed: {e}")

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

    # Encode masks as base64 PNGs
    objects = video["objects"]
    mask_b64: dict[str, str] = {}

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
        obj_color = objects.get(str(obj_id), {}).get("color", "#5B8DD9")
        mask_b64[str(obj_id)] = encode_mask_as_png(mask, obj_color)

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
                        for obj_id_str, frame_map in all_prompts.items():
                            for fidx_str, prompt in frame_map.items():
                                real_fidx = int(fidx_str)
                                if batch_start <= real_fidx < batch_end:
                                    sam.add_points(
                                        pid, vid,
                                        frame_idx=real_fidx,
                                        obj_id=int(obj_id_str),
                                        points=prompt["points"],
                                        labels=prompt["labels"],
                                    )
                                    if sam3_annotated_frame is None or real_fidx < sam3_annotated_frame:
                                        sam3_annotated_frame = real_fidx
                    else:
                        # Subsequent batches: seed from previous batch's last masks
                        for obj_id_int, mask_np in prev_batch_last_masks.items():
                            sam.add_mask_prompt(
                                pid, vid,
                                frame_idx=batch_start,
                                obj_id=obj_id_int,
                                mask=mask_np,
                            )
                        logger.info(
                            f"Batch {batch_idx}: seeded {len(prev_batch_last_masks)} "
                            f"object mask(s) on frame {batch_start}"
                        )
                except Exception as e:
                    logger.error(f"Prompt setup failed for batch {batch_idx}: {e}", exc_info=True)
                    yield {"event": "error", "data": json.dumps({"error": f"Prompt setup failed: {e}"})}
                    return

                # ── 6. Run propagation ────────────────────────────────────
                queue: asyncio.Queue = asyncio.Queue(maxsize=50)
                last_frame_masks: dict = {}
                batch_frame_count = 0

                # SAM3: bidirectional from annotated frame in first batch,
                # forward-only in subsequent batches.
                # SAM2: always forward from batch_start.
                if is_sam3:
                    _start = sam3_annotated_frame if (batch_idx == 0 and sam3_annotated_frame is not None) else batch_start
                    _direction = "both" if batch_idx == 0 else "forward"
                else:
                    _start = batch_start
                    _direction = "forward"

                def _run_propagation(
                    b_start=batch_start, b_len=batch_len,
                    s3_start=_start, direction=_direction,
                ):
                    try:
                        if is_sam3:
                            prop_iter = sam.propagate_stream(
                                pid, vid,
                                start_frame_idx=s3_start,
                                propagation_direction=direction,
                            )
                        else:
                            prop_iter = sam.propagate_stream(
                                pid, vid,
                                start_frame_idx=b_start,
                                max_frame_num_to_track=b_len,
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

                        if current_frame_masks:
                            last_frame_masks = current_frame_masks

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
