---
name: single-animal-condition-analysis
description: >
  Differential-feature ("DE-style"), temporally-binned, and response-correlated
  analysis of single-animal Grounded-SAM3-WebTrack tracking features across
  experimental condition groups defined by an external metadata CSV. Use when
  features are already extracted (clustering pipeline `results/cluster_assignments.csv`
  or per-video `frame_features_*.npz`) and the goal is to compare conditions,
  look at feature dynamics over time, and correlate features with a per-subject
  response measure. Not for running tracking (see tracking-run-ops) or for the
  clustering/report deck (see behavior-quant-reporting).
---

# Single-animal condition analysis

Grounded-SAM3-WebTrack produces per-window single-animal features (12 features:
`energy, speed_mean, speed_std, speed_max, angular_velocity_mean,
angular_velocity_std, frac_time_moving, area_mean, area_std, area_range,
area_slope, mean_eccentricity`) in
`analysis_of_tracking_data/clustering/results/cluster_assignments.csv` (already
z-scored globally), each row a 90-frame window with `video_id` and `start_frame`.
Raw per-frame geometry lives in `.../clustering/features/frame_features_*.npz`
(cols = `a_[7] + b_[7] + interaction_[7]`; for single animal only the first 7
`a_*` = centroid_x, centroid_y, area, major_axis_length, minor_axis_length,
orientation, eccentricity are meaningful).

This skill compares those features across arbitrary condition groups supplied by a
**metadata CSV** that maps each `video_id` to conditions and (optionally) a
per-subject response value.

## The one rule that matters: pick the right experimental unit

Do the stats at the level of the biological replicate, not the window and not the
video. Windows within a video are autocorrelated; multiple cameras/videos of the
same animal are the same animal. **Aggregate window → video → subject, then run
statistics across subjects (n = #subjects).** Report n honestly. Treating windows
(tens of thousands) as independent samples is pseudoreplication and will make
everything "significant" — do not do it.

## Metadata CSV contract

Required columns: `video_id`, one or more categorical condition columns, and a
`subject_id` (the aggregation unit). Optional: a numeric per-subject response
column (e.g. a behavioral log-fold-change) for the correlation analysis. Never
invent condition labels — if the mapping isn't given, ask the user or derive it
explicitly and show your work.

## What the analysis produces

1. **Differential features across conditions (DE-style).** For each 2-level
   contrast: per-feature mean difference, log2 fold-change, Cohen's d, Welch
   t-test + Mann-Whitney U, Benjamini-Hochberg FDR across features. Volcano plot
   (effect size vs -log10 p), a feature×contrast effect-size heatmap, and boxplots
   of the top features. For >2-level factors, Kruskal-Wallis per feature.
2. **Temporal binning (time series).** Bin windows into fixed real-time bins
   (default 2 min; `bin = start_frame // (bin_minutes*60*fps)`), average per
   (subject, bin, feature), then plot group mean ± SEM over time. Also fit a
   per-subject slope over bins and compare slopes between groups (trajectory
   difference test).
3. **Response correlation.** Correlate per-subject feature means against the
   response column (Spearman + Pearson, BH-FDR), with scatter+fit plots for the
   strongest features.

Caveats to always surface: check whether batch / sex / camera are confounded with
condition (crosstab them), and state it. Features from `cluster_assignments.csv`
are in z-score (SD) units.

## How to run

```bash
conda run -n sam3 python analysis_of_tracking_data/condition_analysis/condition_analysis.py \
  --features   analysis_of_tracking_data/clustering/results/cluster_assignments.csv \
  --metadata   metadata/video_metadata.csv \
  --subject-col subject_id \
  --contrast   ecohiv_vs_control=EcoHIV,Control \
  --contrast   art_vs_control=ART,Control \
  --group-col  full_condition \
  --response-col log2fc_zone2_day1_to_day5 \
  --fps 30 --bin-minutes 2 \
  --outdir     analysis_of_tracking_data/condition_analysis
```

Outputs: `results/*.csv|json` (per-subject feature matrix, DE tables, temporal
series, correlation table, `summary.json` with n / caveats) and `plots/*.png`.
The script is self-contained (numpy/pandas/scipy/matplotlib only; implements its
own BH-FDR — statsmodels is not in the `sam3` env). Re-running never recomputes
tracking or the clustering pipeline; it only reads their outputs.

Environment: the `sam3` conda env (base env has a broken numpy/pandas ABI).
