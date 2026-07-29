---
name: behavior-quant-reporting
description: >-
  Run the downstream behavior-quantification (clustering) pipeline on a tracked
  project and produce a FOCUSED report + editable slide deck — not a
  dump-everything PDF. Use when a project under
  /sc/arion/projects/KennyComputational/Behavior/projects/ has completed
  tracking and needs behavioral analysis written up, when asked to "make a
  report/slides" for a behavior project, or when comparing conditions
  (housing/role/treatment) across videos. Covers: resolving which videos belong
  to which condition (from filenames or a user-supplied CSV / text description),
  submitting the quantification LSF job, scaffolding and CURATING a report spec,
  and rendering a two-part PDF (focused brief + full detail) and a .pptx deck.
---

# Behavior quantification reporting

Turn a tracked project into a **focused** write-up: a short brief a collaborator
reads on its own, backed by a longer detail section, plus a matching editable
slide deck. The pipeline discovers behavioral structure; **you** decide what is
worth putting in front of a reader.

## Why this exists (what went wrong before)

The first agent-generated report (`projects/merged_alberto_social_0126-5/generate_report.py`
→ ~65-page PDF) was a **catalog, not a report**: every plot got its own page in
pipeline order, wrapped in dense 9.5pt monospace text pages. Its failure modes —
avoid all of these:

1. **No "focused first" layer.** The only summary was a 2-page, 9-item wall.
2. **Equal weight to everything.** The headline result and a minor boxplot each
   got one page.
3. **Text divorced from figures.** A claim on one page, its figure 1–3 pages later.
4. **Appendix material in the body** (cluster-by-cluster prose for 21+25 clusters).
5. **Monospace-dump aesthetics** — read like stdout.
6. **Not reusable** — hardcoded absolute paths and hand-listed filenames.

The tooling here fixes 1–6 structurally. Your job is the editorial judgment it
can't do: pick the 3–5 findings that matter and say what they mean.

## The deliverable shape

- **Part 1 — Focused Brief (~5 pages):** cover → at-a-glance (N + 3–5 headline
  bullets) → one key figure → one page per finding (figure + plain-language read
  + the stat, *together*) → what-it-means/caveats. Self-contained.
- **Part 2 — Full Detail:** methods, complete cluster/comparison catalog,
  technical appendix. Reference material, clearly demarcated.
- **Slide deck (.pptx, 16:9, editable):** the same Part-1 story, slide-shaped.

## Tooling (all under `downstream_analysis/clustering/reporting/`)

Run everything in the `sam3` conda env from the repo root
(`/sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack`),
prefer `conda run --no-capture-output -n sam3`.

| Step | Command |
|---|---|
| Probe conditions | `python -m downstream_analysis.clustering.reporting.conditions probe --project-dir DIR` |
| Emit conditions template CSV | `python -m ...reporting.conditions template --project-dir DIR --out conditions.csv` |
| Draft a report spec | `python -m ...reporting.scaffold --project-dir DIR --out spec.json` |
| Render two-part PDF | `python -m ...reporting.make_report --spec spec.json --out report.pdf` |
| Render editable deck | `python -m ...reporting.make_slides --spec spec.json --out deck.pptx` |

`make_slides` needs `python-pptx` (installed in the `sam3` env). For merged/
multi-project runs whose outputs live at the project root, pass `--output-dir`
to `scaffold` (and set `output_dir` in the spec).

## Workflow

### 1. Confirm tracking is done

The quantification pipeline consumes `masks.sqlite` per video and needs
`propagation_complete: true`. If tracking isn't finished, use the
`tracking-run-ops` skill first.

### 2. Resolve conditions — ask the user if filenames don't encode them

A behavior report is meaningless until every video is labelled with its
condition. **Always run the probe first:**

```
python -m downstream_analysis.clustering.reporting.conditions probe --project-dir DIR
```

- **`all_resolved: true`** → filenames parse via `experiment_registry.csv`
  (names like `1A_video_test1a_20251014_112131.mp4`). The clustering pipeline
  derives housing/role automatically. Continue.
- **Any `unresolved`** → the filenames don't say what condition each video is.
  **Stop and ask the user** for a description. Accept either form:
  - **Free text**, e.g. *"R1–R4 are saline, R5–R8 are oxycodone; box 1–2 are
    males."* — you then fill the template CSV from it.
  - **A CSV** they provide directly.

  Emit a template keyed by the project's actual videos and fill it:

  ```
  python -m ...reporting.conditions template --project-dir DIR --out DIR/conditions.csv
  ```

  Columns: `video_id, video_name, condition, group, role, housing, treatment,
  notes`. Fill the ones that apply. Sanity-check with
  `conditions summarize --csv DIR/conditions.csv` (counts videos per condition —
  watch for a condition with n=1 or a typo splitting a group).

  Never invent condition labels. If the user's description doesn't cover some
  videos, ask again rather than guessing.

  For datasets that fit the existing experiment schema, prefer **adding rows to
  `downstream_analysis/clustering/experiment_registry.csv`** (keyed by
  `experiment_name, session, box, shave_pattern`) so filename resolution works
  natively; use the standalone `conditions.csv` for ad-hoc/one-off naming.

### 3. Run the quantification pipeline

For LARGE projects, extract frame features in parallel first (cached + reused):

```
python downstream_analysis/clustering/generate_bsub_jobs.py --project-dir DIR --submit-all
```

Then run clustering as an LSF job (add `-- --single-animal` for one-animal-per-video
projects; forward any clustering flags after `--`):

```
export PROJECT_DIR=DIR
bsub < scripts/bsub_behavior_quantification.bsub                 # pair-level
PROJECT_DIR=DIR bsub < scripts/bsub_behavior_quantification.bsub -- --single-animal
```

Small/cached projects run interactively in ~2–3 min:

```
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering \
  --project-dir DIR --skip-mask-verification [--single-animal]
```

Outputs land in `DIR/analysis_of_tracking_data/clustering/` (`summary.json`,
`results/`, `plots/`). Single-animal runs nest plots under `plots/png/` and emit
`results/cluster_description.json`; pair-level runs write `plots/` flat with
`results/housing_comparison.json`; merged runs write `results/condition_comparison.json`.
The reporting tooling auto-detects all three.

### 4. Scaffold the spec

```
python -m ...reporting.scaffold --project-dir DIR --out DIR/report_spec.json
```

This fills real numbers, finds the best headline figure, and drafts finding
STUBS from the strongest feature effects — every editorial field is marked
`TODO`.

### 5. CURATE the spec — this is the actual work

Open `report_spec.json` and replace **every** `TODO`. This is where a focused
report is made:

- **`headline_bullets`**: 3–5 short, declarative findings, most important first.
- **`key_figure`**: the single best figure (usually UMAP-by-condition/role) +
  a caption and an `annotation` ("look at the isolated cluster top-left").
- **`findings`** (keep to 3–5): each has a **declarative `heading`** ("SH
  residents are hyperactive", not "speed_mean"), a real 2–4 sentence
  **`interpretation`** in plain language, the **`stat`** ("Cohen's d = 0.27,
  p < 1e-6, 17/20 features significant"), the supporting **`figure`**, and an
  optional **`annotation`**. Pick the figure that *shows* the claim.
- **`caveats`**: 2–4 real limitations (dorsal camera only, 3 s windows smooth
  fast events, N imbalance across conditions).
- **`implications`**: one paragraph on what this means for the project.
- **`detail_sections`** (Part 2): move the exhaustive material here — full
  cluster heatmaps, all pairwise comparison grids, transition matrices,
  ethograms. Use `type: image | two_image | image_grid | text`. This is where
  the long, project-specific content lives (the "longer part for the project").

Ground every claim in `results/*.json` — read `cluster_summary.json`,
`housing_comparison.json`/`condition_comparison.json`, `feature_tests.csv` for
the actual effect sizes and p-values. Do not carry over a number you didn't see.

### 6. Render

```
python -m ...reporting.make_report --spec DIR/report_spec.json --out DIR/report.pdf
python -m ...reporting.make_slides --spec DIR/report_spec.json --out DIR/deck.pptx
```

### 7. Review

Render a few PDF pages to eyeball layout (`gs` and `convert` are available):

```
gs -sDEVICE=png16m -r90 -dFirstPage=1 -dLastPage=5 -o page_%02d.png DIR/report.pdf
```

Check: Part 1 reads standalone; each finding's figure supports its claim; no
`TODO` survives; Part 2 holds the reference detail. Iterate on the spec and
re-render (cheap — no recompute).

## Guardrails

- **Never fabricate results.** Scaffold stubs are placeholders, not findings.
  If the data doesn't support a claim, drop it.
- **Ask when conditions are ambiguous** rather than guessing a mapping.
- **Keep Part 1 to ~5 pages / the deck to ~12 slides.** If a thing is worth
  saying twice, it belongs in Part 1 once and Part 2 in full.
- The spec is plain JSON — hand-editable and diffable. Re-rendering never
  recomputes the pipeline.
```
