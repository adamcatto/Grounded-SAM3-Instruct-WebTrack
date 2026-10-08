# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git conventions

- Never add Claude as a co-author on git commits (no `Co-Authored-By: Claude ...` trailer).

## Commands

**Start everything:**
```bash
bash start.sh                  # Launches backend + frontend together
bash start_backend.sh          # Backend only (port 8000); also starts the agent vLLM on :8001 if not running (AGENT_LLM_AUTOSTART=0 to skip)
bash start_frontend.sh         # Frontend only (port 5173)
bash stop.sh                   # Kill both servers
```

**One-time setup:**
```bash
bash setup.sh                  # Install deps, clone SAM3, install npm packages
HF_TOKEN=xxx conda run -n sam3 python scripts/download_model.py  # Download model
```

**Frontend dev commands** (from `frontend/`):
```bash
npm run dev      # Vite dev server
npm run build    # tsc + vite build
npm run preview  # Preview production build
```

**Backend runs** inside the `sam3` conda env via uvicorn:
```
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
(The start script handles conda activation and LD_LIBRARY_PATH.)

**Environment variables:**
- `VITE_BACKEND_URL`: Override the backend URL for direct API/video/SSE connections (default: empty, use the Vite `/api` proxy)
- `FRONTEND_HOST` / `BACKEND_HOST`: bind interfaces (default `127.0.0.1`; the app has no auth)
- `CORS_ALLOW_ORIGINS`: comma-separated allowed origins for direct API calls (no `*`)

There are no automated tests.

## Architecture

### Data Flow

The app has three phases:

1. **Setup**: User creates a project, uploads a video (or imports from server path via `POST /videos/import`) → backend applies MP4 faststart optimization in the background for smooth browser streaming.

2. **Annotation**: User selects a frame, clicks objects → frame is extracted on-demand from source video → frontend sends normalized [0,1] coordinates to `/api/projects/{pid}/videos/{vid}/objects/{oid}/points` → backend calls SAM inference → returns base64-encoded RGBA PNG masks → rendered on canvas overlay.

3. **Propagation**: User clicks "Track Objects" → frontend opens an SSE connection to `/api/projects/{pid}/videos/{vid}/propagate` → backend extracts frames in 1000-frame mini-batches (`STREAM_BATCH_SIZE`) into temp directories, runs SAM propagation per batch, saves masks as `.npz` and bboxes as `.json`, streams progress events → frontend shows live progress → user reviews results by scrubbing timeline.

### Backend (`backend/`)

| File | Responsibility |
|---|---|
| `server.py` | All FastAPI routes, SSE propagation endpoint, HTTP range video streaming |
| `sam_predictor.py` | SAM3/SAM2 wrapper: session lifecycle, point prompt inference, propagation generator |
| `project_manager.py` | Project/video/object CRUD, `config.json` persistence, directory paths |
| `video_processor.py` | Frame extraction (cv2), mask PNG encoding, npz/json save/load |

**SAM session lifecycle**: One session per (project, video) pair. Opened on `POST /session`, stays open during annotation and propagation, closed on `DELETE /session`. Sessions hold the inference state and a frame index map.

**Session re-initialization on new frame annotation**: When the user annotates a frame not yet in the current session (e.g. a new frame extracted on-demand), `POST /objects/{oid}/points` detects this and re-inits the session against the full `annotated_frames/` directory. All saved point prompts are then replayed **sorted by frame index** (SAM3 requires sequential order — out-of-order replay triggers "Image features for frame N are not cached" errors).

**Frame directories**: Three distinct directories exist for JPG frames:
- `frames/` — populated on-demand when the user views a frame; used as the video frame strip in the UI
- `annotated_frames/` — single frames extracted on-demand when the user annotates; SAM sessions for annotation point at this directory
- Temp dirs (`/tmp/sam3wt_*`) — created per propagation mini-batch, deleted after each batch completes

**Frame index translation**: Preview frames have gaps (e.g., `000000.jpg`, `001902.jpg`), but SAM expects sequential indices (0, 1, 2…). `sam_predictor._build_frame_map` scans the frames directory and `_to_sam_idx` translates real → SAM indices on every call.

**Cross-batch identity tracking**: Propagation runs in 1000-frame mini-batches (`STREAM_BATCH_SIZE` in `server.py`). Each batch uses a temp directory that is deleted after processing. The last frame's masks from batch N are fed as prompts to seed batch N+1, preserving object IDs across batches.

**Model fallback**: SAM3 is preferred (`pretrained_models/sam3.pt`). If missing, falls back to SAM2 (`pretrained_models/sam2.1_hiera_large.pt`). Both use the same internal API surface in `sam_predictor.py`.

**SAM3 internal state workarounds** (in `sam_predictor.propagate_stream`): SAM3 has several bugs that require direct mutation of its internal state dict (`_ALL_INFERENCE_STATES[sid]["state"]`):
- `cached_frame_outputs[frame_idx]` must be pre-seeded with `{}` before `add_prompt` or propagation — SAM3 asserts it exists
- `action_history` must be cleared before propagation — otherwise SAM3 picks `propagation_partial` mode (which requires a prior full propagation) instead of `propagation_full`
- `rank0_metadata["obj_first_frame_idx"][obj_id]` must be pre-seeded with `-999999` for all tracked objects — without it SAM3's hotstart heuristic crashes with `KeyError` when an object is unmatched for 8+ frames

**SAM3 mask handoff between batches**: SAM3 has no `add_new_mask` API, so cross-batch seeding uses a point prompt at the mask's center-of-mass instead (see `add_mask_prompt`).

### Frontend (`frontend/src/`)

| Path | Responsibility |
|---|---|
| `store/useStore.ts` | Zustand global state: current frame, objects, masks, propagation status |
| `api/client.ts` | Typed Axios wrappers for all backend endpoints + SSE helper |
| `components/FrameViewer/` | Frame display + `AnnotationCanvas` click-to-annotate overlay |
| `components/VideoPlayer/` | HTML5 video element wrapper for smooth playback mode |
| `components/LeftPanel/` | Object list, propagation controls, SSE event handling |
| `components/Timeline/` | Playback controls, thumbnail strip, per-object progress rows |

**Frame display strategy**: `FrameViewer` switches between JPEG frames (for annotation precision when paused) and an HTML5 `<video>` element (for smooth playback). The video source endpoint supports HTTP 206 range requests for seeking. Video `src`, SSE propagation and agent streams all go through the Vite `/api` proxy (which passes Range requests and unbuffered SSE); set `VITE_BACKEND_URL` only to bypass it, and add that origin to `CORS_ALLOW_ORIGINS`.

**Coordinate system**: Frontend sends normalized [0,1] point coordinates. Backend receives and passes them directly to SAM. Saved masks are full-resolution `(H, W)` binary uint8 arrays.

**Mask rendering**: `maskUtils.ts` caches decoded base64 mask PNGs as `HTMLImageElement` objects. `AnnotationCanvas` composites live (post-click) masks from `currentFrameMasks` and saved (post-propagation) masks from `savedMaskCache`.

**Object IDs**: Assigned by `ProjectManager.add_object` as sequential integers starting at 1, stored as string keys in `config.json` (e.g. `"1"`, `"2"`). SAM receives them as `int`. The same ID string is used in `masks/*.npz` dict keys and bbox JSON keys.

### Storage Layout

```
/opt/.sam3_projects/
└── <project-uuid>/
    ├── config.json          # All metadata: project, videos, objects, point prompts
    └── videos/<video-uuid>/
        ├── frames/          # 000000.jpg ... (extracted on-demand when frames are viewed)
        ├── annotated_frames/ # Frames extracted on-demand when user annotates
        ├── masks/           # 000000.npz → {obj_<id>: uint8 H×W array}
        ├── bboxes/          # 000000.json → {obj_id: [x,y,w,h,score]}
        └── source.<ext>     # Original video file (faststart-optimized for browser streaming)
```

Vite proxies `/api/*` → `http://localhost:8000`, so all API calls in the frontend use `/api/` paths.
