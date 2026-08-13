# SAM3 Web Tracker

A full-stack web application for interactive video object segmentation and tracking, built on Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3). Point-click to annotate objects in a single frame, then propagate masks across the entire video. Includes a downstream behavioral analysis pipeline for quantifying animal behavior from tracked masks.

![UI Screenshot](assets/ai/sam_webapp/Screenshot%202026-02-12%20at%2012.52.08%20PM.png)

---

## Table of Contents

<details open>
<summary><strong>Getting started</strong></summary>

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [One-Time Setup](#one-time-setup)
  - [Start the Application](#start-the-application)
  - [Stop](#stop)
- [Usage Workflow](#usage-workflow)

</details>

<details>
<summary><strong>Web application</strong></summary>

- [Architecture](#architecture)
  - [Data Flow](#data-flow)
- [Backend API](#backend-api)
  - [SAM Model Fallback](#sam-model-fallback)
  - [Key Backend Modules](#key-backend-modules)
- [Frontend](#frontend)
  - [Key Components](#key-components)
  - [Environment Variables](#environment-variables)
- [Storage Layout](#storage-layout)

</details>

<details>
<summary><strong>Downstream analysis</strong></summary>

- [Downstream Analysis](#downstream-analysis)
- [Locomotion Pipeline](#locomotion-pipeline)
- [Behavioral Clustering](#behavioral-clustering)
  - [Feature Hierarchy](#feature-hierarchy)
  - [Single-Project Pipeline](#single-project-pipeline)
- [Multi-Project Pipeline](#multi-project-pipeline)
  - [Experiment Design](#experiment-design)
  - [Running the Multi-Project Pipeline](#running-the-multi-project-pipeline)
  - [Pipeline Stages](#pipeline-stages)
  - [Multi-Project Output Layout](#multi-project-output-layout)
  - [Multi-Project Modules](#multi-project-modules)

</details>

<details>
<summary><strong>HPC &amp; automation</strong></summary>

- [HPC Batch Processing](#hpc-batch-processing)
  - [Generate and Submit Jobs](#generate-and-submit-jobs)
  - [Parallel Tracking (Propagation)](#parallel-tracking-propagation)
- [Scripts &amp; Utilities](#scripts--utilities)

</details>

<details open>
<summary><strong>Configuration</strong></summary>

- [Configuration](#configuration)
- [Recommended first-time setup](#recommended-first-time-setup)
- [Environment variables](#environment-variables)
- [Project storage and <code>config.json</code>](#project-storage-and-configjson)
- [Model checkpoints](#model-checkpoints)
- [HPC configuration (<code>configs/env.yaml</code>)](#hpc-configuration-configsenvyaml)
- [Clustering pipeline parameters](#clustering-pipeline-parameters)
- [Multi-project analysis configuration](#multi-project-analysis-configuration)
  - [Project directory → experiment name mapping](#1-project-directory--experiment-name-mapping)
  - [Identity registry (<code>experiment_registry.csv</code>)](#2-identity-registry-experiment_registrycsv)
  - [Batch definition and correction](#3-batch-definition-and-correction)
- [Feature-extraction bsub jobs](#feature-extraction-bsub-jobs)
- [Configuration quick reference](#configuration-quick-reference)

</details>

<details>
<summary><strong>Development</strong></summary>

- [Development](#development)
  - [Project Structure](#project-structure)
  - [Running Without GPU](#running-without-gpu)
  - [Adding New Analysis Modules](#adding-new-analysis-modules)
- [License](#license)

</details>

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


| Group           | Endpoints                             | Description                                           |
| --------------- | ------------------------------------- | ----------------------------------------------------- |
| **Projects**    | `POST/GET/PATCH/DELETE /api/projects` | CRUD, merge                                           |
| **Videos**      | `POST /api/projects/{pid}/videos`     | Upload/import, info, source streaming (HTTP 206)      |
| **Sessions**    | `POST/DELETE /api/.../session`        | Open/close SAM inference session per (project, video) |
| **Annotation**  | `POST /api/.../objects/{oid}/points`  | Add point prompts, receive mask predictions           |
| **Propagation** | `GET /api/.../propagate`              | SSE endpoint streaming batch progress                 |
| **Masks**       | `GET /api/.../masks/{frame}`          | Load/delete/swap masks and bounding boxes             |
| **Export**      | `GET /api/.../export`                 | Render video with mask overlays                       |


### SAM Model Fallback

The backend loads SAM3 (`pretrained_models/sam3.pt`) as the primary model. If unavailable, it falls back to SAM2.1 Large (`pretrained_models/sam2.1_hiera_large.pt`). Both share the same inference API in `sam_predictor.py`.

### Key Backend Modules


| File                 | Responsibility                                                            |
| -------------------- | ------------------------------------------------------------------------- |
| `server.py`          | All FastAPI routes, SSE propagation, range-request video streaming        |
| `sam_predictor.py`   | SAM3/SAM2 wrapper: session lifecycle, point prompt inference, propagation |
| `project_manager.py` | Project/video/object CRUD, `config.json` persistence                      |
| `video_processor.py` | Frame extraction (OpenCV), mask encoding, MP4 faststart                   |
| `mask_store.py`      | SQLite mask storage with RLE compression, legacy NPZ fallback             |
| `mask_seg_codec.py`  | Binary COCO RLE compression codec                                         |
| `classifier.py`      | Optional DINOv2-small per-pixel object classifier                         |


---

## Frontend

React 18 + TypeScript + Vite, styled with Tailwind CSS.

### Key Components


| Component           | Role                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| `FrameViewer/`      | JPEG frame display (paused) or HTML5 `<video>` (playback) with canvas mask overlay |
| `AnnotationCanvas`  | Click-to-annotate with normalized coordinate conversion and live mask rendering    |
| `LeftPanel/`        | Object list, propagation controls, SSE event handler                               |
| `Timeline/`         | Playback controls, frame thumbnail strip, per-object progress rows                 |
| `VideoPlayer/`      | HTML5 video element wrapper for smooth playback                                    |
| `store/useStore.ts` | Zustand global state: projects, videos, objects, masks, propagation status         |
| `api/client.ts`     | Typed Axios wrappers for all backend endpoints + SSE helper                        |


### Environment Variables


| Variable           | Default                 | Purpose                                                            |
| ------------------ | ----------------------- | ------------------------------------------------------------------ |
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


| Exp | Name        | Description                           | Pairs |
| --- | ----------- | ------------------------------------- | ----- |
| 1   | Habituation | Two familiar GH littermates           | 10    |
| 2   | Test Day    | SH or GH resident + novel GH intruder | 21    |
| 3   | SH Intruder | GH resident + SH intruder             | 10    |
| 4   | Locomotion  | Single mouse alone (SH or GH)         | 21    |


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
8. **Batch diagnostics** -- per-feature batch effects, PERMANOVA, PCA before/after correction (`batch_diagnostics/`)

#### Multi-Project Output Layout

```
<output-dir>/
  plots/                    # UMAPs, boxplots, ethograms, etc.
  batch_diagnostics/        # Batch QA stats + figures (η², PERMANOVA, PCA)
  results/                  # JSON/CSV assignments, comparisons, batch info
    cluster_assignments.csv
    condition_comparison.json
    batch_correction_info.json
    combined_dataset.npz
```

#### Multi-Project Modules


| Module                        | Purpose                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `experiment_registry.csv`     | Identity mapping: 103 rows, (experiment, session, box, shave_pattern) to mouse ID/role/housing |
| `experiment_registry.py`      | Registry loader with `resolve_identities()` and `validate_registry()`                          |
| `batch_correction.py`         | Parametric ComBat (empirical Bayes) + z-score fallback, pure numpy/scipy                       |
| `multi_project.py`            | `MultiProjectPipeline` orchestrator                                                            |
| `multi_project_comparison.py` | Cross-condition statistical tests                                                              |
| `locomotion_comparison.py`    | 12-dim per-object features, per-camera paired design                                           |
| `condition_boxplots.py`       | Per-feature and cluster composition boxplots                                                   |
| `plots_multi.py`              | UMAP variants (by experiment, condition, batch, before/after correction)                       |
| `batch_diagnostics.py`        | Batch effect stats, PERMANOVA, PCA/heatmap diagnostics                                         |
| `multi_project_cli.py`        | CLI entry point                                                                                |


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

For a **single project on one machine** (sequential propagation, or a dry-run before launching
workers), use `scripts/run_pending_inference.py`. It prints a RUN/SKIP plan for each video, then
calls the same `/api/.../propagate` SSE endpoint the web UI uses. Start the backend first
(`uvicorn server:app` or `./start.sh`). See [Scripts & Utilities](#scripts--utilities) for
examples including `--list-only`.

---

## Scripts & Utilities


| Script                                  | Purpose                                                               |
| --------------------------------------- | --------------------------------------------------------------------- |
| `scripts/download_model.py`             | Download SAM3/SAM2 checkpoints from HuggingFace (requires `HF_TOKEN`) |
| `scripts/parallel_tracking_launcher.py` | Orchestrate batch propagation across projects                         |
| `scripts/parallel_tracking_worker.py`   | Single-video propagation worker process                               |
| `scripts/run_pending_inference.py`      | CLI propagation for eligible videos (dry-run with `--list-only`, or run sequentially) |
| `scripts/merge_projects.py`             | Merge two SAM3 projects (combine configs, videos)                     |
| `scripts/migrate_masks_sqlite.py`       | Migrate legacy NPZ/JSON masks to SQLite storage                       |

### `run_pending_inference.py`

Runs whole-video propagation for videos that are ready: anchor labeling complete,
`propagation_complete` false, not actively running on another worker, and with point prompts
present. Useful after labeling in the UI when you want to finish tracking from a terminal, retry
failed jobs, or inspect eligibility before starting HPC workers.

**Prerequisites:** SAM backend running and reachable (default `http://127.0.0.1:8000`). Project
resolved via `--project-dir` (directory containing `config.json`) or `--pid` under
`SAM3_PROJECTS_DIR` / `~/.sam3_zero_projects`.

**Output:** eligibility summary on **stderr**; live SSE progress on **stdout** (suppress with
`--quiet-stream`).

```bash
# Dry-run: show which videos would RUN vs SKIP (no HTTP calls)
python scripts/run_pending_inference.py \
  --project-dir ~/.sam3_zero_projects/ab12-demo \
  --backend http://127.0.0.1:8000 \
  --list-only

# Propagate all eligible videos in the project (sequential)
python scripts/run_pending_inference.py \
  --project-dir /path/to/project \
  --backend http://127.0.0.1:8000

# Reset stuck "running" status left by a killed job, then propagate
python scripts/run_pending_inference.py \
  --project-dir /path/to/project \
  --clear-stuck \
  --backend http://127.0.0.1:8000

# Single video, match web UI "use all anchors", less progress spam
python scripts/run_pending_inference.py \
  --pid ab12-demo \
  --videos vid001 \
  --use-all-anchors \
  --quiet-stream
```

Common flags: `--videos` (subset), `--stale-timeout` (seconds before a `running` job is treated
as dead), `--sse-timeout` (HTTP read timeout for long propagations). `parallel_tracking_worker.py`
reuses the same eligibility logic and SSE client for multi-GPU / LSF batch runs.


---

## Configuration

Configuration is layered: **environment variables** (where data and models live), **`configs/env.yaml`** (HPC batch tracking), **`config.json` per project** (videos/objects/masks), **CLI flags** (clustering parameters), and **Python defaults** in `downstream_analysis/clustering/config.py`. In practice you usually set the env vars once per machine, copy `env_template.yaml` for HPC work, and pass CLI flags when re-running analysis with different clustering settings.

### Recommended first-time setup

Use this checklist when configuring a new machine (laptop, lab workstation, or HPC login node):

```bash
# 1. Clone and enter the repo
git clone https://github.com/adamcatto/Grounded-SAM3-Instruct-WebTrack.git
cd Grounded-SAM3-Instruct-WebTrack

# 2. Create / activate the sam3 conda environment (Python 3.10+)
conda create -n sam3 python=3.10 -y
conda activate sam3

# 3. Run setup (edit conda path inside setup.sh if yours differs from the default)
bash setup.sh

# 4. Point project storage at a writable location (see below)
export SAM3_TRACKING_PROJECTS_DIR=/path/to/your/projects

# 5. Download model weights (requires HuggingFace access to facebook/sam3)
export HF_TOKEN=hf_your_token_here
conda run -n sam3 python scripts/download_model.py

# 6. For HPC batch propagation / feature extraction, create a local env file:
cp configs/env_template.yaml configs/env.yaml
# Edit configs/env.yaml — queue, account, conda path, log directories

# 7. Verify the web app
bash start_backend.sh    # terminal 1
bash start_frontend.sh   # terminal 2
# Open http://localhost:5173
```

Add the `export SAM3_TRACKING_PROJECTS_DIR=...` line to your `~/.bashrc` (or job preamble) so the backend, downstream CLIs, and HPC workers all resolve the same project folders.

On shared HPC filesystems, prefer a project directory under your allocation (e.g. `/sc/arion/projects/YourLab/Behavior/projects/`) rather than `~/.sam3_zero_projects`, so jobs on compute nodes see the same data as your interactive sessions.

---

### Environment variables


| Variable                     | Default                 | Used by                      | Description                                                                                                                                               |
| ---------------------------- | ----------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SAM3_TRACKING_PROJECTS_DIR` | `~/.sam3_zero_projects` | Backend, downstream analysis | Root folder for all SAM3 project directories. **Set this first** on any machine that runs tracking or analysis.                                           |
| `SAM3_PROJECTS_DIR`          | (same as above)         | Backend, downstream analysis | Legacy alias; if both are set, `SAM3_TRACKING_PROJECTS_DIR` wins.                                                                                         |
| `HF_TOKEN`                   | —                       | `scripts/download_model.py`  | HuggingFace token for downloading gated SAM3 weights.                                                                                                     |
| `SAM3_ENV_YAML`              | `configs/env.yaml`      | Parallel tracking scripts    | Path to YAML config for HPC propagation workers (see below).                                                                                              |
| `VITE_BACKEND_URL`           | `http://localhost:8000` | Frontend (build-time)        | Backend URL for video Range requests and SSE propagation when not using the Vite dev proxy. Set when the UI runs on a different host than the GPU server. |


**Example — shared lab storage on Minerva:**

```bash
export SAM3_TRACKING_PROJECTS_DIR=/sc/arion/projects/KennyComputational/Behavior/projects
export HF_TOKEN=hf_...
```

**Example — frontend on laptop, backend on GPU node:**

```bash
# On the GPU machine
bash start_backend.sh   # listens on 0.0.0.0:8000

# On your laptop (before npm run dev / start_frontend.sh)
export VITE_BACKEND_URL=http://gpu-node.your.cluster:8000
bash start_frontend.sh
```

The backend reads `SAM3_TRACKING_PROJECTS_DIR` at startup. You can also change the active projects root at runtime from the UI (project drawer → set projects folder); that override applies to the running server process only and does not change downstream CLI behavior — CLIs always use the env var.

---

### Project storage and `config.json`

Each tracked experiment is a directory under the projects root:

```
$SAM3_TRACKING_PROJECTS_DIR/
  3c9bddf2-Home-Cage-Interactions-0126-test-day/   # {short-uuid}-{slug}
    config.json                                      # project metadata
    videos/
      <video-uuid>_<video-name>/
        source.mp4
        masks.sqlite
        ...
```

**Naming:** Directories are `{8-char-uuid}-{human-readable-slug}`. Downstream tools resolve projects by short uuid prefix (the part before the first `-` that matches `config.json → id`).

**`config.json` structure (simplified):** The web backend owns this file. Downstream analysis reads it for video lists, object names, frame ranges, and propagation status — it does not edit project metadata.

```json
{
  "id": "3c9bddf2",
  "name": "Home-Cage-Interactions-0126-test-day",
  "videos": {
    "<video-uuid>": {
      "id": "<video-uuid>",
      "name": "test1a_3A",
      "width": 1920,
      "height": 1080,
      "num_frames": 54000,
      "start_frame": 0,
      "objects": {
        "1": { "name": "HeadShave", "color": "#5B8DD9" },
        "2": { "name": "NoShave", "color": "#E8A445" }
      },
      "propagation_complete": true
    }
  }
}
```

**Practical notes:**

- Object display names (`HeadShave`, `BackShave`, `NoShave`) are matched against `experiment_registry.csv` during multi-project analysis to assign mouse identities.
- Video folder names (`test1a_3A`) encode session + camera view; the registry parser expects this convention.
- Analysis outputs are written beside the project: `<project>/analysis_of_tracking_data/clustering/` (single-project) or a separate `--output-dir` (multi-project).

---

### Model checkpoints


| File                                      | Purpose                                                         |
| ----------------------------------------- | --------------------------------------------------------------- |
| `pretrained_models/sam3.pt`               | Primary SAM3 weights (download via `scripts/download_model.py`) |
| `pretrained_models/sam2.1_hiera_large.pt` | Automatic fallback if SAM3 is missing                           |


Both paths are relative to the repo root. The backend loads whichever checkpoint exists (`backend/sam_predictor.py`). No environment variable is required if files are in the default location.

---

### HPC configuration (`configs/env.yaml`)

Batch propagation and LSF job submission use a **per-clone, gitignored** YAML file. Copy the template and customize for your cluster:

```bash
cp configs/env_template.yaml configs/env.yaml
```

Alternatively, point at a shared config elsewhere:

```bash
export SAM3_ENV_YAML=/sc/arion/projects/YourLab/sam3-env.yaml
```

**`parallel_tracking`** — read by `scripts/parallel_tracking_worker.py`:


| Key                         | Typical value                    | Purpose                                                     |
| --------------------------- | -------------------------------- | ----------------------------------------------------------- |
| `backend_url`               | `http://127.0.0.1:8000`          | FastAPI server used for SSE propagation                     |
| `local_gpu_workers`         | `0` or `4`                       | Spawn one worker process per GPU on the same node           |
| `backend_port_base`         | `8810`                           | Ports `8810`, `8811`, … when using multi-GPU local backends |
| `auto_start_local_backends` | `true` / `false`                 | Worker starts uvicorn per GPU automatically                 |
| `claims_file`               | `tracked_videos_in_progress.txt` | Coordination file beside `config.json`                      |
| `idle_loops_before_exit`    | `288`                            | Worker exits after N idle polls (for batch arrays)          |


**`lsf_parallel_tracking`** — read by `scripts/parallel_tracking_launcher.py`:


| Key                 | Example (Minerva)                               | Maps to                        |
| ------------------- | ----------------------------------------------- | ------------------------------ |
| `queue_project`     | `acc_KennyComputational`                        | `bsub -P`                      |
| `queue`             | `gpu`                                           | `bsub -q`                      |
| `num_process_slots` | `32`                                            | `bsub -n`                      |
| `resources`         | `rusage[mem=16000]`, `span[hosts=1]`, `h100nvl` | `bsub -R` (one per entry)      |
| `gpu_allocation`    | `num=1`                                         | `bsub -gpu`                    |
| `run_limit`         | `144:00`                                        | `bsub -W`                      |
| `stdout` / `stderr` | `/sc/arion/.../logs/%J.out`                     | Log paths (`%J` = job id)      |
| `shell_preamble`    | `conda activate ...`                            | Commands run before the worker |
| `worker_python`     | `/path/to/envs/sam3/bin/python`                 | Python in batch jobs           |


**Submit parallel tracking:**

```bash
# Edit configs/env.yaml first, then:
python scripts/parallel_tracking_launcher.py /path/to/project

# Or run locally without LSF:
python scripts/parallel_tracking_launcher.py --local /path/to/project
```

CLI flags on the worker/launcher override YAML values when provided.

---

### Clustering pipeline parameters

Defaults live in `downstream_analysis/clustering/config.py` (`ClusteringConfig` dataclass). Override via CLI on both single- and multi-project runs:


| Parameter         | CLI flag                   | Default                   | When to change                                                                                  |
| ----------------- | -------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| Window size       | `--window-size`            | 90 frames (~3 s @ 30 fps) | Longer windows smooth behavior; shorter windows capture brief motifs                            |
| Stride            | `--stride`                 | 30 frames (~1 s)          | Smaller stride = more overlapping windows, denser sampling                                      |
| k-NN neighbors    | `--n-neighbors`            | 30                        | Increase for smoother cluster graphs on large datasets                                          |
| Leiden resolution | `--resolution`             | 1.0                       | **Lower → fewer clusters; higher → more clusters** (cheap to re-run; frame features are cached) |
| Batch correction  | `--batch-correction`       | `combat`                  | Multi-project only: `combat`, `zscore_per_batch`, or `none`                                     |
| Skip mask check   | `--skip-mask-verification` | off                       | Use after full propagation; saves startup time on large projects                                |


**Internal defaults** (edit `config.py` if needed, not exposed on CLI):


| Field                        | Default  | Purpose                                          |
| ---------------------------- | -------- | ------------------------------------------------ |
| `close_proximity_threshold`  | 0.05     | Fraction of video diagonal for "close" proximity |
| `stationary_speed_threshold` | 0.001    | Speed below which an animal counts as stationary |
| `normalize_method`           | `zscore` | Feature normalization before clustering          |
| `umap_min_dist`              | 0.1      | UMAP embedding compactness                       |


**Single-project example — re-cluster with fewer clusters:**

```bash
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/project \
  --skip-mask-verification \
  --resolution 0.5
```

Cached per-video frame features in `<project>/analysis_of_tracking_data/clustering/features/` are reused; only clustering and plots re-run.

---

### Multi-project analysis configuration

The multi-project pipeline merges several interaction projects and optionally one locomotion project. Three things usually need attention beyond CLI flags:

#### 1. Project directory → experiment name mapping

`experiment_registry.py` maps the **8-character uuid prefix** of each project folder to an experiment name:


| UUID prefix | Experiment    | Role in pipeline                                 |
| ----------- | ------------- | ------------------------------------------------ |
| `185b3df6`  | `hab`         | Habituation (littermates)                        |
| `3c9bddf2`  | `test_day`    | Test day (SH/GH resident + GH intruder)          |
| `a3d11289`  | `sh_intruder` | SH intruder condition                            |
| `37254501`  | `locomotion`  | Single-mouse alone (optional `--locomotion-dir`) |


For new study cohorts, add entries to `_PROJECT_SLUG_MAP` in `experiment_registry.py` or ensure your project folder uuid prefix is listed.

#### 2. Identity registry (`experiment_registry.csv`)

Maps `(experiment, session, box, shave_pattern)` → mouse ID, role, housing. Bundled CSV covers the Home-Cage-Interactions-0126 study. For new experiments:

1. Copy `downstream_analysis/clustering/experiment_registry.csv` to a study-specific path.
2. Add rows for each `(session, box, mouse)` combination.
3. Pass `--registry-csv /path/to/your_registry.csv` to `multi_project_cli`.

Validate coverage before a long run:

```python
from pathlib import Path
from downstream_analysis.clustering.experiment_registry import validate_registry

errors = validate_registry([
    Path("/path/to/Hab"),
    Path("/path/to/test-day"),
])
print(errors)  # empty list = all 2-object videos resolved
```

#### 3. Batch definition and correction

Batch ID is `{experiment_name}_{camera_view}` (e.g. `test_day_3A`). ComBat correction runs at this level by default to remove camera/session technical variation while preserving biological condition structure.

After correction, inspect `<output-dir>/batch_diagnostics/`:

- `batch_diagnostics.json` — global η² and PERMANOVA before/after
- `pca_before_after_by_batch.png` — visual check that batch clusters mix post-correction
- `per_feature_batch_effects.csv` — features that still carry batch signal

Use `--batch-correction zscore_per_batch` if some batches are very small (<10 windows); use `none` to disable correction for debugging.

---

### Feature-extraction bsub jobs

`downstream_analysis/clustering/generate_bsub_jobs.py` has **site-specific constants** at the top of the file (account, queue, walltime, conda path in the generated LSF scripts). Edit these for your cluster, or override at generation time:

```bash
conda run -n sam3 python -m downstream_analysis.clustering.generate_bsub_jobs \
  --project-dir /path/to/project \
  --account acc_YourProject \
  --queue premium \
  --walltime 08:00 \
  --mem 16000 \
  --submit-all
```

Generated scripts land in `<project>/analysis_of_tracking_data/clustering/bsub_jobs/` with logs in `bsub_logs/`. See `downstream_analysis/clustering/RUN_PIPELINE.md` for the full step-by-step.

---

### Configuration quick reference


| Goal                             | What to configure                                                   |
| -------------------------------- | ------------------------------------------------------------------- |
| Store projects on shared disk    | `SAM3_TRACKING_PROJECTS_DIR`                                        |
| Download SAM3 weights            | `HF_TOKEN` + `scripts/download_model.py`                            |
| UI on laptop, GPU backend remote | `VITE_BACKEND_URL`                                                  |
| HPC parallel propagation         | `configs/env.yaml` (from `env_template.yaml`)                       |
| LSF queue/account/memory         | `configs/env.yaml → lsf_parallel_tracking`                          |
| Multi-GPU on one node            | `parallel_tracking.local_gpu_workers` + `auto_start_local_backends` |
| Fewer/more behavior clusters     | `--resolution` (re-run clustering only)                             |
| New mouse cohort / study         | Custom `experiment_registry.csv` + uuid map                         |
| Batch effect QA                  | Inspect `<output-dir>/batch_diagnostics/` after multi-project run   |


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