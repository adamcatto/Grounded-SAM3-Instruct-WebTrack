# SAM3 Web Tracker

A full-stack web application for interactive video object segmentation and tracking, built on Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3). Point-click to annotate objects in a single frame, then propagate masks across the entire video. Includes a downstream behavioral analysis pipeline for quantifying animal behavior from tracked masks.

![UI Screenshot](assets/ai/sam_webapp/Screenshot%202026-02-12%20at%2012.52.08%20PM.png)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Workflow](#usage-workflow)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Storage Layout](#storage-layout)
- [Downstream Analysis](#downstream-analysis)
  - [Locomotion Pipeline](#locomotion-pipeline)
  - [Behavioral Clustering](#behavioral-clustering)
  - [Multi-Project Pipeline](#multi-project-pipeline)
- [HPC Batch Processing](#hpc-batch-processing)
- [Scripts & Utilities](#scripts--utilities)
- [Configuration](#configuration)
- [Development](#development)

---

## Overview

SAM3 Web Tracker provides a browser-based interface for segmenting and tracking multiple objects across long videos. It is designed for research workflows where precise object masks are needed frame-by-frame -- for example, tracking individual animals in behavioral neuroscience experiments.

The system consists of three layers:

1. **Web application** (React frontend + FastAPI backend) for interactive annotation and propagation
2. **Mask storage** (SQLite + legacy NPZ) for efficient frame-level mask persistence
3. **Downstream analysis** (Python pipeline) for locomotion quantification, behavioral feature extraction, clustering, and multi-experiment statistical comparison

---

## Features

**Annotation & Tracking**
- Point-click annotation: select any frame, click to place positive/negative prompts on objects
- Real-time SAM inference: see mask predictions instantly after each click
- Batch propagation: track objects across thousands of frames via SSE-streamed mini-batches
- Multi-object support: track multiple objects simultaneously with distinct colors and IDs
- Anchor frame system: annotate key frames, propagate between them for long videos
- Pause/resume propagation mid-video

**Video Management**
- Upload or import videos from server paths
- Automatic MP4 faststart optimization for smooth browser streaming
- HTTP 206 range-request support for seeking
- On-demand frame extraction (no upfront decompression of full video)
- Multi-project organization with CRUD operations
- Project merging

**Visualization & Review**
- Smooth playback with HTML5 video element
- Frame-by-frame scrubbing with mask overlays
- Per-object progress tracking on the timeline
- Composite mask export as overlay video
- SAM session state debug panel

**Downstream Analysis**
- Locomotion quantification (centroid path-length in sliding windows)
- 33-dimensional behavioral feature extraction from mask pairs
- Leiden community detection clustering with UMAP embedding
- Multi-experiment comparison with batch correction (ComBat)
- Statistical testing (Fisher's exact, Mann-Whitney U, Wilcoxon signed-rank)
- Publication-ready plots (boxplots, heatmaps, ethograms, UMAPs)

---

## Architecture

```
                    Browser (React + Zustand)
                         |
                    Vite proxy /api/* --> localhost:8000
                         |
                    FastAPI (server.py)
                    /           \
          SAMPredictor       ProjectManager
          (sam3/sam2)         (config.json)
               |                    |
          pretrained_models/    ~/.sam3_zero_projects/
          sam3.pt                 <project-uuid>/
          sam2.1_hiera_large.pt     config.json
                                    videos/<vid>/
                                      source.mp4
                                      masks.sqlite
                                      frames/
                                      annotated_frames/
```

### Data Flow

**Phase 1 -- Setup:** Create project, upload/import video. Backend applies faststart optimization.

**Phase 2 -- Annotation:** User clicks objects on a frame. Frontend sends normalized `[0, 1]` coordinates to the backend, which runs SAM inference and returns base64-encoded RGBA PNG masks rendered on a canvas overlay.

**Phase 3 -- Propagation:** User triggers "Track Objects." Frontend opens an SSE connection. Backend extracts frames in 1000-frame mini-batches into temp directories, runs SAM propagation per batch, saves masks to SQLite (+ legacy NPZ/JSON), and streams progress events back. The last frame's masks from batch N seed batch N+1 to maintain object identity across batches.

---

## Quick Start

### Prerequisites

- Python 3.10+ with conda
- Node.js 18+ (or via nvm)
- CUDA-capable GPU (recommended; CPU works but is slow)
- HuggingFace account with access to [SAM3](https://huggingface.co/facebook/sam3) (gated model)

### One-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/adamcatto/Grounded-SAM3-Instruct-WebTrack.git
cd Grounded-SAM3-Instruct-WebTrack

# 2. Run the setup script (installs backend deps, SAM3, frontend deps)
bash setup.sh

# 3. Download model weights (requires HuggingFace token)
HF_TOKEN=hf_your_token conda run -n sam3 python scripts/download_model.py
```

### Start the Application

```bash
# Terminal 1: Backend (port 8000)
bash start_backend.sh

# Terminal 2: Frontend (port 5173)
bash start_frontend.sh
```

Open `http://localhost:5173` in your browser.

### Stop

```bash
bash stop.sh
```

---

## Usage Workflow

1. **Create a project** from the left sidebar
2. **Import or upload a video** (MP4 recommended)
3. **Add objects** (e.g., "Mouse A", "Mouse B") with the + button
4. **Navigate to a frame** and click on an object to place point prompts
   - Left-click = positive prompt (this is the object)
   - Right-click = negative prompt (this is NOT the object)
5. **Review the predicted mask** -- add more points to refine
6. **Repeat** for each object on one or more anchor frames
7. **Click "Track Objects"** to propagate masks across all frames
8. **Scrub the timeline** to review results; re-annotate frames where tracking drifted
9. **Export** composite overlay video or use masks for downstream analysis

---

## Backend API

The FastAPI backend (`backend/server.py`) exposes 60+ endpoints. Key groups:

| Group | Endpoints | Description |
|-------|-----------|-------------|
| **Projects** | `POST/GET/PATCH/DELETE /api/projects` | CRUD, merge |
| **Videos** | `POST /api/projects/{pid}/videos` | Upload/import, info, source streaming (HTTP 206) |
| **Sessions** | `POST/DELETE /api/.../session` | Open/close SAM inference session per (project, video) |
| **Annotation** | `POST /api/.../objects/{oid}/points` | Add point prompts, receive mask predictions |
| **Propagation** | `GET /api/.../propagate` | SSE endpoint streaming batch progress |
| **Masks** | `GET /api/.../masks/{frame}` | Load/delete/swap masks and bounding boxes |
| **Export** | `GET /api/.../export` | Render video with mask overlays |

### SAM Model Fallback

The backend loads SAM3 (`pretrained_models/sam3.pt`) as the primary model. If unavailable, it falls back to SAM2.1 Large (`pretrained_models/sam2.1_hiera_large.pt`). Both share the same inference API in `sam_predictor.py`.

### Key Backend Modules

| File | Responsibility |
|------|----------------|
| `server.py` | All FastAPI routes, SSE propagation, range-request video streaming |
| `sam_predictor.py` | SAM3/SAM2 wrapper: session lifecycle, point prompt inference, propagation |
| `project_manager.py` | Project/video/object CRUD, `config.json` persistence |
| `video_processor.py` | Frame extraction (OpenCV), mask encoding, MP4 faststart |
| `mask_store.py` | SQLite mask storage with RLE compression, legacy NPZ fallback |
| `mask_seg_codec.py` | Binary COCO RLE compression codec |
| `classifier.py` | Optional DINOv2-small per-pixel object classifier |

---

## Frontend

React 18 + TypeScript + Vite, styled with Tailwind CSS.

### Key Components

| Component | Role |
|-----------|------|
| `FrameViewer/` | JPEG frame display (paused) or HTML5 `<video>` (playback) with canvas mask overlay |
| `AnnotationCanvas` | Click-to-annotate with normalized coordinate conversion and live mask rendering |
| `LeftPanel/` | Object list, propagation controls, SSE event handler |
| `Timeline/` | Playback controls, frame thumbnail strip, per-object progress rows |
| `VideoPlayer/` | HTML5 video element wrapper for smooth playback |
| `store/useStore.ts` | Zustand global state: projects, videos, objects, masks, propagation status |
| `api/client.ts` | Typed Axios wrappers for all backend endpoints + SSE helper |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_BACKEND_URL` | `http://localhost:8000` | Backend URL for direct video/SSE connections (bypasses Vite proxy) |

---

## Storage Layout

```
~/.sam3_zero_projects/                      # or $SAM3_TRACKING_PROJECTS_DIR
  <project-uuid>/
    config.json                             # Project metadata, objects, point prompts
    videos/
      <video-uuid>_<video-name>/
        source.mp4                          # Original video (faststart-optimized)
        masks.sqlite                        # Consolidated mask storage (primary)
        masks/                              # Legacy: per-frame NPZ files
          000000.npz                        #   {obj_id: uint8 H*W array}
        bboxes/                             # Legacy: per-frame JSON files
          000000.json                       #   {obj_id: [x, y, w, h, score]}
        frames/                             # Extracted on-demand when user views
        annotated_frames/                   # Extracted on-demand for SAM annotation
```

Mask reads go through `VideoMaskStorage` which checks SQLite first, falling back to legacy NPZ. Writes go to both for backward compatibility.

---

## Downstream Analysis

The `downstream_analysis/` package processes tracked masks into quantitative behavioral metrics. It is independent of the web application and runs from the command line.

### Locomotion Pipeline

Computes centroid path-length for each tracked object in sliding windows.

```bash
conda run --no-capture-output -n sam3 python -m downstream_analysis \
  --project-dir /path/to/project
```

**Output:** Per-video histograms, aggregate CDFs, bimodal GMM fits (stationary vs. mobile fractions).

### Behavioral Clustering

Extracts 33-dimensional behavioral sequence features from pairs of tracked objects, then clusters behavioral motifs.

#### Feature Hierarchy

```
Frame features (21-dim per frame):
  Per-object (7 each, x2):  centroid_x/y, area, major/minor axis, orientation, eccentricity
  Interaction (7):           centroid distance, contour distance, overlap, body angles, combined area

Sequence features (33-dim per sliding window):
  Per-object (12 each, x2): energy, speed (mean/std/max), angular velocity (mean/std),
                             frac_time_moving, area (mean/std/range/slope), eccentricity
  Interaction (9):           approach/retreat rate, parallel movement, chase metric/lag,
                             proximity fraction, relative heading, min distance (mean/std), overlap
```

#### Single-Project Pipeline

```bash
# Step 1: Extract features (parallelizable across videos via bsub)
conda run -n sam3 python -m downstream_analysis.clustering.generate_bsub_jobs \
  --project-dir /path/to/project --submit-all

# Step 2: Cluster + compare (after all feature jobs complete)
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/project --skip-mask-verification
```

**Clustering method:** k-NN graph (k=30) with Leiden community detection (configurable resolution). Falls back to spectral clustering if `leidenalg` is not installed.

**Embedding:** UMAP (2D), falls back to t-SNE.

**Output:** Cluster assignments, UMAP plots, feature heatmaps, per-video ethograms, housing condition comparisons, Excel report.

### Multi-Project Pipeline

Merges interaction data across experiments, applies batch correction, and runs cross-condition statistical comparisons.

#### Experiment Design

The pipeline supports 4 experiment types with an identity registry mapping each video to specific mice:

| Exp | Name | Description | Pairs |
|-----|------|-------------|-------|
| 1 | Habituation | Two familiar GH littermates | 10 |
| 2 | Test Day | SH or GH resident + novel GH intruder | 21 |
| 3 | SH Intruder | GH resident + SH intruder | 10 |
| 4 | Locomotion | Single mouse alone (SH or GH) | 21 |

**Condition groups** for statistical comparison:
- GH + Littermate (Exp 1)
- SH Resident + GH Intruder (Exp 2)
- GH Resident + GH Intruder (Exp 2)
- GH Resident + SH Intruder (Exp 3)
- SH / GH Alone (Exp 4, locomotion)

#### Running the Multi-Project Pipeline

```bash
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.multi_project_cli \
  --project-dirs \
    /path/to/185b3df6-Home-Cage-Interactions-0126-Hab \
    /path/to/3c9bddf2-Home-Cage-Interactions-0126-test-day \
    /path/to/a3d11289-Home-Cage-Interactions-0126-SH-Intruder \
  --locomotion-dir /path/to/37254501-Loc-Home-Cage-Interactions-0126 \
  --output-dir /path/to/multi_project_analysis \
  --skip-mask-verification \
  --batch-correction combat
```

#### Pipeline Stages

1. **Feature extraction** per project (reuses cached `.npz` files)
2. **Identity enrichment** via `experiment_registry.csv` -- maps each video to mouse IDs, roles, housing
3. **Batch correction** (parametric ComBat) at the `(experiment, camera_view)` level to remove technical variation
4. **Clustering** (Leiden + UMAP) on the combined corrected feature matrix
5. **Condition comparisons** -- Fisher's exact per-cluster, Mann-Whitney U per-feature, Cohen's d
6. **Locomotion comparison** (optional) -- paired Wilcoxon signed-rank for same-mouse alone vs. with-intruder
7. **Plots** -- condition boxplots, cluster composition bars, UMAP colored by experiment/condition/batch, before/after correction, locomotion paired boxplots, ethograms

#### Multi-Project Modules

| Module | Purpose |
|--------|---------|
| `experiment_registry.csv` | Identity mapping: 103 rows, (experiment, session, box, shave_pattern) to mouse ID/role/housing |
| `experiment_registry.py` | Registry loader with `resolve_identities()` and `validate_registry()` |
| `batch_correction.py` | Parametric ComBat (empirical Bayes) + z-score fallback, pure numpy/scipy |
| `multi_project.py` | `MultiProjectPipeline` orchestrator |
| `multi_project_comparison.py` | Cross-condition statistical tests |
| `locomotion_comparison.py` | 12-dim per-object features, per-camera paired design |
| `condition_boxplots.py` | Per-feature and cluster composition boxplots |
| `plots_multi.py` | UMAP variants (by experiment, condition, batch, before/after correction) |
| `multi_project_cli.py` | CLI entry point |

---

## HPC Batch Processing

Feature extraction is the bottleneck (~55 min/video). The pipeline supports HPC parallelization via LSF (bsub).

### Generate and Submit Jobs

```bash
# Generate .lsf files for all videos in a project
conda run -n sam3 python -m downstream_analysis.clustering.generate_bsub_jobs \
  --project-dir /path/to/project

# Submit all jobs
conda run -n sam3 python -m downstream_analysis.clustering.generate_bsub_jobs \
  --project-dir /path/to/project --submit-all
```

### Parallel Tracking (Propagation)

For running SAM propagation on many videos without the web UI:

```bash
# Launch parallel tracking across all incomplete videos in a project
python scripts/parallel_tracking_launcher.py \
  --project-dir /path/to/project

# Or submit as a batch job
bsub < scripts/bsub_parallel_project_tracking.bsub
```

---

## Scripts & Utilities

| Script | Purpose |
|--------|---------|
| `scripts/download_model.py` | Download SAM3/SAM2 checkpoints from HuggingFace (requires `HF_TOKEN`) |
| `scripts/parallel_tracking_launcher.py` | Orchestrate batch propagation across projects |
| `scripts/parallel_tracking_worker.py` | Single-video propagation worker process |
| `scripts/run_pending_inference.py` | Monitor and auto-resume incomplete propagations |
| `scripts/merge_projects.py` | Merge two SAM3 projects (combine configs, videos) |
| `scripts/migrate_masks_sqlite.py` | Migrate legacy NPZ/JSON masks to SQLite storage |

---

## Configuration

### Backend

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SAM3_TRACKING_PROJECTS_DIR` | `~/.sam3_zero_projects` | Root directory for project data |
| `HF_TOKEN` | -- | HuggingFace token for model download |

### Clustering

Parameters are set in `downstream_analysis/clustering/config.py` or via CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window-size` | 90 | Sliding window size in frames (3s at 30fps) |
| `--stride` | 30 | Window stride in frames (1s) |
| `--n-neighbors` | 30 | k-NN graph neighbors |
| `--resolution` | 1.0 | Leiden community detection resolution |
| `--batch-correction` | `combat` | Batch correction method (`combat`, `zscore_per_batch`, `none`) |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_BACKEND_URL` | `http://localhost:8000` | Backend URL (bypasses Vite proxy for Range requests and SSE) |

---

## Development

### Project Structure

```
Grounded-SAM3-Instruct-WebTrack/
  backend/                    # Python FastAPI server
    server.py                 # Routes, SSE, video streaming
    sam_predictor.py          # SAM3/SAM2 inference wrapper
    project_manager.py        # Project CRUD, config.json
    video_processor.py        # Frame extraction, mask encoding
    mask_store.py             # SQLite mask storage
  frontend/                   # React + TypeScript
    src/
      store/useStore.ts       # Zustand global state
      api/client.ts           # Typed API client
      components/             # UI components
  downstream_analysis/        # Post-tracking analysis
    clustering/               # Behavioral feature clustering
    locomotion.py             # Centroid path-length
    tracking_io.py            # Load masks via VideoMaskStorage
    pipeline.py               # Locomotion analysis pipeline
  scripts/                    # Utilities and batch processing
  pretrained_models/          # SAM3/SAM2 checkpoints
  .sam3_src/                  # SAM3 source (cloned during setup)
  setup.sh                    # One-time environment setup
  start_backend.sh            # Launch backend
  start_frontend.sh           # Launch frontend
  stop.sh                     # Kill servers
```

### Running Without GPU

The backend will run on CPU but inference will be significantly slower. Propagation of a 10-minute video that takes ~2 minutes on a modern GPU may take 30+ minutes on CPU.

### Adding New Analysis Modules

1. Add your module to `downstream_analysis/clustering/`
2. Import from `tracking_io.py` for mask loading and `clustering_pipeline.py` for the clustering result container
3. Use `VideoMaskStorage` (not direct NPZ globbing) for mask reads
4. Register in `__init__.py` if it should be importable from the package

---

## License

This project builds on [SAM 3](https://github.com/facebookresearch/sam3) by Meta AI Research, which is released under the Apache 2.0 license.
