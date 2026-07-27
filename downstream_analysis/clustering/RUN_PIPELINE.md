# Clustering Pipeline: How to Run

## Overview

The pipeline has two phases:
1. **Frame feature extraction** (slow, parallelizable via bsub) -- ~55 min per video
2. **Clustering + comparison** (fast, single process) -- ~5 min total

## Step 1: Generate and submit bsub jobs for frame feature extraction

```bash
# Generate .lsf files + submit all 47 jobs in one command:
conda run -n sam3 python \
  /sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack/downstream_analysis/clustering/generate_bsub_jobs.py \
  --project-dir /sc/arion/projects/KennyComputational/Behavior/projects/3c9bddf2-Home-Cage-Interactions-0126-test-day \
  --submit-all
```

Or do it in two steps (generate first, review, then submit):
```bash
# Generate only:
conda run -n sam3 python .../generate_bsub_jobs.py --project-dir /path/to/project --dry-run

# Then submit:
bash /path/to/project/analysis_of_tracking_data/clustering/bsub_jobs/submit_all.sh
```

## Step 2: Monitor progress

```bash
# Quick status:
bjobs -w | grep feat_

# Detailed check (cached count, running jobs, errors):
bash /path/to/project/analysis_of_tracking_data/clustering/bsub_jobs/check_status.sh
```

Logs are in `<project>/analysis_of_tracking_data/clustering/bsub_logs/`.

## Step 3: Run clustering (after all jobs complete)

Once all 47 videos have cached features:

```bash
cd /sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack

conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /sc/arion/projects/KennyComputational/Behavior/projects/3c9bddf2-Home-Cage-Interactions-0126-test-day \
  --skip-mask-verification
```

This will:
- Load cached frame features (seconds, not hours)
- Compute 33-dim sequence features per sliding window
- Cluster via Leiden (or spectral fallback)
- Compare isolated vs group-housed mice
- Generate all plots

### Single-animal projects

For projects where each video tracks a **single** animal (no social pair), add
`--single-animal`:

```bash
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/single_animal_project \
  --skip-mask-verification \
  --single-animal
```

This clusters on the focal animal's 12 per-object kinematic/shape features only
(dropping the `b_*` and interaction features, which are undefined with one
animal) and produces **descriptive** outputs instead of the social/housing
comparison:

- `results/cluster_description.json` (in place of `housing_comparison.json`)
- Cluster summaries, per-video ethograms, feature heatmap, feature violins,
  and behavioral transition matrices
- No housing/enrichment plots or sheets (`umap_by_housing`, `enrichment_bars`,
  `feature_enrichment`, `cluster_composition` are skipped)

If you omit the flag but every video has only one object, the pipeline still
runs and logs a warning suggesting `--single-animal`.

## Step 4 (optional): Re-run with different parameters

Cached frame features are reused automatically. Only clustering is re-done:

```bash
# Different resolution (more/fewer clusters):
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/project \
  --skip-mask-verification \
  --resolution 0.5

# Different window size:
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/project \
  --skip-mask-verification \
  --window-size 150 --stride 50
```

## Output

All results go to `<project>/analysis_of_tracking_data/clustering/`:

```
features/                          # cached per-video frame features + combined dataset
  frame_features_<video_id>.npz    # one per video (~10 MB each)
  sequence_features.npz            # combined (N_windows, 33) matrix
results/
  cluster_assignments.json         # per-window cluster labels
  cluster_summary.json             # per-cluster feature centroids
  housing_comparison.json          # isolated vs group-housed stats
  embedding.npz                    # UMAP/t-SNE coordinates
plots/
  umap_by_cluster.png
  umap_by_housing.png
  cluster_composition.png
  feature_heatmap.png
  enrichment_bars.png
  feature_violins.png
  ethograms/ethogram_<vid>.png     # per-video behavioral timelines
bsub_jobs/                         # generated .lsf files
bsub_logs/                         # stdout/stderr from jobs
summary.json                       # run metadata and cluster counts
clustering_report.xlsx             # Excel workbook (tables + embedded plots)
```

Multi-project merged runs use the same layout under ``--output-dir`` (typically
``analysis_of_tracking_data/clustering``), with ``condition_comparison.json`` in
``results/`` instead of ``housing_comparison.json``.

## Troubleshooting

- **Job failed?** Check `bsub_logs/feat_<vid>.err`. Re-submit individual jobs:
  `bsub < bsub_jobs/feat_<vid>.lsf`

- **Skip already-cached videos when re-submitting:**
  `python generate_bsub_jobs.py --project-dir /path --skip-cached --submit-all`

- **No output from pipeline?** Use `conda run --no-capture-output` (conda buffers stderr).
