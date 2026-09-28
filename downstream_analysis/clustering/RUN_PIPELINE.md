# Clustering Pipeline: How to Run

## Overview

The pipeline has two phases:
1. **Frame feature extraction** (slow, one Slurm/LSF job per video) -- ~55 min per video
2. **Clustering + comparison** (fast, single process) -- ~5 min total

Cluster settings (scheduler, account, queue, resources) come from the `hpc:` block
of `configs/env.yaml`; see the "HPC Batch Processing" section of the top-level README.

## Step 1: Submit frame feature extraction jobs

```bash
# Preview the generated #SBATCH / #BSUB scripts:
python scripts/hpc_submit.py features /path/to/project --dry-run

# Submit one job per tracked video:
python scripts/hpc_submit.py features /path/to/project
```

## Step 2: Monitor progress

```bash
squeue --me -n feat_     # Slurm   (LSF: bjobs -w | grep feat_)

# Videos still missing cached features:
python scripts/hpc_submit.py features /path/to/project --skip-cached --dry-run
```

Logs are in `logs/hpc/` (or `hpc.log_dir`); generated job scripts in `logs/hpc/jobs/`.

## Step 3: Run clustering (after all jobs complete)

Once every video has cached features, either submit it as a job:

```bash
python scripts/hpc_submit.py behavior-quant /path/to/project
```

or run it directly:

```bash
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir /path/to/project \
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
summary.json                       # run metadata and cluster counts
clustering_report.xlsx             # Excel workbook (tables + embedded plots)
```

Multi-project merged runs use the same layout under ``--output-dir`` (typically
``analysis_of_tracking_data/clustering``), with ``condition_comparison.json`` in
``results/`` instead of ``housing_comparison.json``.

## Troubleshooting

- **Job failed?** Check `logs/hpc/feat_<vid>_<jobid>.err`, then re-submit only
  what's missing: `python scripts/hpc_submit.py features /path --skip-cached`

- **No output from pipeline?** Use `conda run --no-capture-output` (conda buffers stderr).
