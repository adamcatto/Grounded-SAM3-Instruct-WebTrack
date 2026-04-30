# Consolidated mask storage (`masks.sqlite`)

This document describes how per-video segmentation masks and bounding boxes are stored in SQLite alongside legacy NPZ/JSON files, why it exists, and how to migrate older projects.

## Motivation

Previously each frame used `masks/{frame:06d}.npz` plus `bboxes/{frame:06d}.json`. Long videos meant tens of thousands of small files (inode churn, slow globs, noisy downstream scans). The consolidated store keeps **one `masks.sqlite` per video directory**, with **keyed lookups by `frame_idx`** and **incremental upserts** during propagation.

## Layout

For a video directory `videos/<uuid>_…/`:

| Path | Role |
|------|------|
| `masks.sqlite` | Primary store: one row per frame (`seg_blob`, `bbox_json`). |
| `masks/*.npz` | Legacy dual-write / fallback read until removed after migration. |
| `bboxes/*.json` | Legacy dual-write / fallback read until removed after migration. |

## Schema

SQLite pragmas at open (see `backend/mask_store.py`): WAL, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`.

Tables:

- **`mask_store_meta`** — keys include `schema_version`, `seg_codec`, **`cache_revision`** (integer as text; incremented on any write touching masks/bboxes for cache invalidation semantics).
- **`frame_segmentation`** — `frame_idx` (PK), `seg_blob` (BLOB), `bbox_json` (TEXT), `updated_at`.

## Segmentation blob (`seg_blob`)

Implemented in `backend/mask_seg_codec.py`: versioned binary with magic `SAM3WT01`, zlib-compressed pickle payload `{h, w, rles}` where each object id maps to a **COCO RLE** via `pycocotools`. Decode yields dense `uint8` masks; encoding is **lossless** relative to the legacy NPZ mask arrays.

## API surface (`VideoMaskStorage`)

Defined in `backend/mask_store.py` for a given `video_dir`:

- **Read:** `load_masks_dense(frame_idx)`, `load_bboxes(frame_idx)`, `has_masks(frame_idx)` — **SQLite row first**, then legacy NPZ/JSON if no row.
- **Write:** `save_frame(frame_idx, masks, bboxes=None)` — upserts SQLite, bumps `cache_revision`, writes legacy NPZ + JSON for compatibility.
- **Delete:** `delete_frame`, `delete_masks_range` — clears SQLite rows and matching legacy files; range delete counts **distinct frame indices** touched.
- **Enumerate:** `iter_saved_frame_indices_sorted()` — union of SQLite `frame_idx` and legacy `.npz` stems (matches swap/analysis needs across migrated and unmigrated trees).
- **Reset:** `wipe_sqlite_file()` removes the DB (and WAL/shm sidecars) when a video is fully reset.

Bbox JSON matches legacy shape: `{ obj_id_str: [x, y, w, h, score] }` normalized xywh + score. Missing bbox entries on save are filled from prior row or `bbox_norm_xywh_score_from_mask` in `backend/video_processor.py`.

## Server integration

`backend/server.py` routes propagation, point merges, swaps, deletes, composite/export, diagnostics, and SAM seeding through **`VideoMaskStorage`**. In-memory PNG encode cache keys are **`("masks_png", pid, vid, frame_idx, rev_token)`** where `rev_token` is either `("sqlite", cache_revision)` when that frame is served from a DB row or `("legacy", npz_mtime_ns)` for legacy-only frames. `_invalidate_mask_cache(pid, vid)` drops all entries for that video prefix.

## Downstream and classifier

- `downstream_analysis/tracking_io.py` — full-track checks and centroid timelines use **`VideoMaskStorage(video_dir)`** instead of globbing `*.npz`.
- `backend/classifier.py` — takes `video_dir` and loads masks via the same storage.

## Migration CLI

Script: **`scripts/migrate_masks_sqlite.py`**.

Typical usage (from repo root, with deps available e.g. `conda run -n sam3`):

```bash
python scripts/migrate_masks_sqlite.py --video-dir /path/to/project/videos/<video_dir>
python scripts/migrate_masks_sqlite.py --project-dir /path/to/project --video-id <uuid>
python scripts/migrate_masks_sqlite.py --project-dir /path/to/project --all-complete
```

Flags:

- **`--dry-run`** — report counts only.
- **`--batch-size`** — rows per transaction (default 1000).
- **`--verify-sample K`** / **`--verify-full`** — compare SQLite decode vs legacy NPZ (and bbox JSON or derived bboxes).
- **`--delete-legacy --yes`** — after successful verify, remove legacy `masks/*.npz` and `bboxes/*.json` (requires verification enabled).

## Related paths

| Area | File |
|------|------|
| Store + DDL | `backend/mask_store.py` |
| Blob codec | `backend/mask_seg_codec.py` |
| Bbox-from-mask helper | `backend/video_processor.py` (`bbox_norm_xywh_score_from_mask`) |
| HTTP / propagation | `backend/server.py` |
| SQLite path helper | `backend/project_manager.py` (`masks_sqlite_path`) |
| Export overlay | `backend/video_processor.py` (`export_video_with_masks(..., video_dir=...)`) |

## Rollout notes

New writes **dual-write** SQLite + legacy files so unmigrated deployments keep working. After **`migrate_masks_sqlite.py`** + optional **`--delete-legacy`**, reads remain correct via SQLite-first dual-read. Dropping the legacy path entirely would be a later cleanup once all videos are migrated.
