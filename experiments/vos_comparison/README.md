# `vos_comparison` — single-shot VOS vs. anchor-frame labeling

This experiment quantifies the value of **anchor frame labeling**. In the main
app, a human re-labels frames sparsely throughout a video (the *anchors*) so that
when SAM3 propagation collapses at some frame, the error does not propagate across
the whole video. To justify that effort, we measure what *single-shot* video
object segmentation (VOS) does instead: keep only the **first** labeled frame as a
prompt and let SAM3 propagate across the entire video with **no further human
anchors**. We then score the single-shot track against the original anchor-based
track (treated as ground truth) frame-by-frame using IoU / Jaccard, and locate
where quality collapses.

## What it produces

`build` emits a sibling project `single_shot_vos_<source>` whose videos are
**symlinks** to the source video files (no pixel/mask copies). Each video keeps
only the first anchor's point prompts; all propagation state is reset. A
`vos_comparison_manifest.json` records the link back to each ground-truth video.

`track` runs SAM3 propagation on that project through the backend's HTTP/SSE
`/propagate` endpoint (the same path the UI uses, so all SAM3 session workarounds
are reused). Predicted masks are written to each video's `masks.sqlite`.

`evaluate` and `figures` produce, under `<project>/vos_comparison_results/`:

```
per_frame_iou.csv      per_object_iou.csv     video_summary.csv
object_summary.csv     results.json           timelines.npz
figures/  fig_iou_curves  fig_aggregate_band  fig_survival  fig_summary_bars   (.pdf + .png)
tables/   table_per_video.tex   table_aggregate.tex                            (booktabs)
```

## Usage

From the repo root, inside the `sam3` conda env (matplotlib needed for figures):

```bash
pip install -r experiments/vos_comparison/requirements.txt

# 0. Backend must be running for the tracking stage:
bash start_backend.sh                       # uvicorn on :8000

# 1–4 in one shot:
python -m experiments.vos_comparison all \
    --source-project <project-id|folder|path> \
    --backend http://127.0.0.1:8000

# …or stage by stage:
python -m experiments.vos_comparison build    --source-project <ref>
python -m experiments.vos_comparison track    --project single_shot_vos_<ref> --backend http://127.0.0.1:8000
python -m experiments.vos_comparison evaluate --project single_shot_vos_<ref> --threshold 0.5
python -m experiments.vos_comparison figures  --project single_shot_vos_<ref>
```

Projects are resolved the same way as the backend: by filesystem path, folder
name, or short id, searched under `$SAM3_TRACKING_PROJECTS_DIR` /
`$SAM3_PROJECTS_DIR` / `~/.sam3_zero_projects`.

## Experimental design notes

- **What is controlled.** SAM3 propagates the video in `anchor_batch_size`-frame
  batches; across batch boundaries it re-seeds from the *previous batch's
  propagated mask* (machine output, not a human label). This cross-batch handoff
  is identical in both conditions, so the only difference being tested is the
  presence of periodic **human** anchors.
  - `--batch-size inherit` (default) reuses the source video's batch size — a
    controlled comparison where anchors are simply removed.
  - `--batch-size single` runs the whole video as one batch (purest single-shot:
    no cross-batch re-seeding at all). Heavier on disk/RAM for long videos.
  - `--batch-size N` sets an explicit batch size.

- **Ground truth.** The anchor-based track is the reference. Anchor labeling has
  IoU 1.0 against itself by construction, so figures draw it as a reference line;
  the experiment characterizes the single-shot *deviation* from it.

- **Collapse detection** (`metrics.detect_collapse`). After smoothing the
  per-frame mean IoU with a trailing window (~1% of the video), the collapse
  frame is the earliest frame from which at least `--persist-frac` (default 0.8)
  of the remaining smoothed frames sit below `τ` (`--threshold`, default 0.5).
  This captures a sustained drop and ignores brief dips that recover.

- **Multi-instance objects.** Instance mask keys (`"1"`, `"1_2"`, …) are merged
  by base object id (logical union) before IoU, matching how the rest of the
  analysis code treats object identity.
