# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may
change the API and on-disk project format.

## [0.1.1] - 2026-10-01

### Changed

- Restored remote frontend access for trusted lab/VPN networks by listening on
  `0.0.0.0` by default, while retaining `FRONTEND_HOST=127.0.0.1` for local or
  SSH-tunneled access.
- Added machine-local project-root configuration through a gitignored `.env`
  file, with a portable `.env.example` template and shell-variable overrides.

### Security

- Added a startup warning and documentation that remotely exposing the
  frontend grants unauthenticated access to the application.

## [0.1.0] - 2026-09-28

First tagged release.

### Features

- Browser UI (React + FastAPI) for point-prompt annotation and multi-object
  video tracking with SAM 3 (SAM 2 fallback), SSE-streamed propagation in
  mini-batches with cross-batch identity handoff, anchor frames, pause/resume.
- Agent chat pane: text-prompted segmentation and agentic annotation with a
  local (vLLM / Ollama) or cloud vision LLM; per-project system prompts; in-app
  trace viewer and optional RL-sample capture.
- Pose/point tracking with CoTracker3 and multi-view video registration.
- Video management: upload or import from server paths, lazy downsampling,
  MP4 faststart, HTTP Range streaming, SQLite mask storage, project merging.
- Downstream analysis: locomotion, 33-D behavioral features, Leiden
  clustering + UMAP, multi-project comparison with ComBat batch correction,
  PDF/PPTX reporting.
- HPC batch jobs on Slurm or LSF via `scripts/hpc_submit.py` (tracking job
  arrays, per-video feature extraction, behavior quantification).

### Security

- Servers bind to `127.0.0.1` by default (`FRONTEND_HOST` / `BACKEND_HOST`);
  the app has no authentication.
- CORS restricted to an allow-list (`CORS_ALLOW_ORIGINS`) instead of `*`.
- Mask blobs are decoded with a restricted unpickler, so opening an untrusted
  project folder cannot execute code.
- Dependencies updated past known advisories (FastAPI/Starlette,
  python-multipart, Pillow, Vite 7, axios); `npm audit` clean.

[0.1.0]: https://github.com/adamcatto/Grounded-SAM3-Instruct-WebTrack/releases/tag/v0.1.0
[0.1.1]: https://github.com/adamcatto/Grounded-SAM3-Instruct-WebTrack/compare/v0.1.0...v0.1.1
