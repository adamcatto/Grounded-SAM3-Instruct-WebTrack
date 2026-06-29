# Anchor-Frame Memory Experiment

This experiment measures what is lost when SAM propagation uses the standard
recent queue memory bank instead of loading all labeled anchor frames into memory.

The source project is treated as the anchor-frame-memory reference. The experiment
creates a sibling project named `recent_queue_memory_<source name>` that keeps the
same videos, objects, anchor labels, and point prompts, but removes all propagated
tracking state. It then tracks the fork with `use_all_anchors=false` and evaluates
those masks against the source project's anchor-memory masks.

## Commands

```bash
python -m experiments.anchor_frame_memory all \
  --source-project /home/adam/.sam3_zero_projects/0818b9be-Mouse-Mingle-Test \
  --backend http://127.0.0.1:8000 \
  --quiet-stream
```

Individual stages:

```bash
python -m experiments.anchor_frame_memory backup --source-project <project>
python -m experiments.anchor_frame_memory build --source-project <project>
python -m experiments.anchor_frame_memory track --project <recent_queue_project>
python -m experiments.anchor_frame_memory evaluate --project <recent_queue_project>
python -m experiments.anchor_frame_memory figures --project <recent_queue_project>
```

## Outputs

Results are written under `<recent_queue_project>/anchor_frame_memory_results/`:

- `per_frame_iou.csv`: frame-level mean/worst IoU and missing object counts.
- `per_object_iou.csv`: object-level IoU, presence, area, and re-entry flags.
- `object_summary.csv`: per-object loss and re-entry summary statistics.
- `video_summary.csv`: video-level IoU and lost object-frame totals.
- `reentry_events.csv`: object return events after an anchor-GT absence.
- `results.json`: run metadata and aggregate statistics.
- `timelines.npz`: arrays used by the plotting code.
- `figures/*.png`, `figures/*.pdf`, `figures/*.svg`: time-course and summary plots.

The main re-entry definition is: an object is present in the anchor-memory masks
after being absent for at least `--min-absence-frames` frames. Each event is scored
over the following `--reentry-window` frames.

Lost tracking is counted when an object is anchor-present and either absent from
the queue-memory prediction or predicted with IoU below `--low-iou-threshold`
(default `0.2`). The low-IoU case captures identity swaps or badly corrupted
masks where a same-id prediction exists but no longer overlaps the anchor-memory
reference well enough to count as useful tracking.

## Interpretation

Anchor-frame memory is used as the reference because those tracks were generated
with every labeled anchor frame available to SAM's memory. Queue memory is expected
to fail when an object leaves the field of view long enough that recent-frame
memory no longer contains it. The summary tables count these failures as missing
object-frames and missed re-entry events.

## Verified Mouse-Mingle-Test run

This experiment was run on:

- Anchor-memory reference: `/home/adam/.sam3_zero_projects/0818b9be-Mouse-Mingle-Test`
- Queue-memory fork: `/home/adam/.sam3_zero_projects/dcd9b5c1-recent_queue_memory_Mouse-Mingle-Test`
- Backup made before the run: `/home/adam/.sam3_zero_projects/_backups/0818b9be-Mouse-Mingle-Test.backup-20260629T170508Z`
- Results: `/home/adam/.sam3_zero_projects/dcd9b5c1-recent_queue_memory_Mouse-Mingle-Test/anchor_frame_memory_results`

The source project's legacy masks were also migrated into:

`/home/adam/.sam3_zero_projects/0818b9be-Mouse-Mingle-Test/videos/f17961f9/masks.sqlite`

The SQLite migration verified 34,136 frame rows spanning frames 0 through 34,135.

Headline results for `2B_video_2LBFwithdrawal_20250919_130012.mp4`:

- Mean IoU, queue memory vs anchor-frame memory: `0.8696`
- Median IoU: `0.9435`
- Worst-object mean IoU: `0.8064`
- Missing anchor-present object-frames: `3,455`
- Low-IoU anchor-present object-frames (`IoU < 0.2`): `1,262`
- Lost anchor-present object-frames (missing or low-IoU): `4,717`
- Anchor-GT re-entry events: `19`
- Missed/lost re-entry events: `12`
- `CageMouse`: 452 lost frames out of 34,134 anchor-present frames (`1.32%`).
- `FreeMouse`: 4,265 lost frames out of 14,300 anchor-present frames (`29.83%`), with 12 lost re-entry events.

The generated artifacts include CSV tables, `results.json`, `timelines.npz`, and
PNG/PDF/SVG versions of:

- `timecourse_mean_min_iou`
- `object_iou_facets`
- `object_summary_bars`
- `missing_objects_timeline`
- `reentry_event_windows`
