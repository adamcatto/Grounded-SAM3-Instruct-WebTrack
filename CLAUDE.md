# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Start everything:**
```bash
bash start.sh                  # Launches backend + frontend together
bash start_backend.sh          # Backend only (port 8000)
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
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
(The start script handles conda activation and LD_LIBRARY_PATH.)

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

**Frame index translation**: Preview frames have gaps (e.g., `000000.jpg`, `001902.jpg`), but SAM expects sequential indices (0, 1, 2…). `sam_predictor._build_frame_map` scans the frames directory and `_to_sam_idx` translates real → SAM indices on every call.

**Cross-batch identity tracking**: Propagation runs in 1000-frame mini-batches (`STREAM_BATCH_SIZE` in `server.py`). Each batch uses a temp directory that is deleted after processing. The last frame's masks from batch N are fed as prompts to seed batch N+1, preserving object IDs across batches.

**Model fallback**: SAM3 is preferred (`pretrained_models/sam3.pt`). If missing, falls back to SAM2 (`pretrained_models/sam2.1_hiera_large.pt`). Both use the same internal API surface in `sam_predictor.py`.

### Frontend (`frontend/src/`)

| Path | Responsibility |
|---|---|
| `store/useStore.ts` | Zustand global state: current frame, objects, masks, propagation status |
| `api/client.ts` | Typed Axios wrappers for all backend endpoints + SSE helper |
| `components/FrameViewer/` | Frame display + `AnnotationCanvas` click-to-annotate overlay |
| `components/LeftPanel/` | Object list, propagation controls, SSE event handling |
| `components/Timeline/` | Playback controls, thumbnail strip, per-object progress rows |

**Frame display strategy**: `FrameViewer` switches between JPEG frames (for annotation precision when paused) and an HTML5 `<video>` element (for smooth playback). The video source endpoint supports HTTP 206 range requests for seeking. In dev mode, the video `src` bypasses Vite's proxy and hits the backend directly (CORS is `allow_origins=["*"]`) to avoid proxy buffering that breaks Range requests. SSE propagation events also bypass the proxy.

**Coordinate system**: Frontend sends normalized [0,1] point coordinates. Backend receives and passes them directly to SAM. Saved masks are full-resolution `(H, W)` binary uint8 arrays.

**Mask rendering**: `maskUtils.ts` caches decoded base64 mask PNGs as `HTMLImageElement` objects. `AnnotationCanvas` composites live (post-click) masks from `currentFrameMasks` and saved (post-propagation) masks from `savedMaskCache`.

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
