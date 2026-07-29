# Behavior-quantification reporting

Turn a completed clustering run into a **focused** report + editable slide deck,
driven by a small human-curated `ReportSpec` (JSON) rather than dumping every
plot. See the `behavior-quant-reporting` agent skill
(`.agents/skills/behavior-quant-reporting/SKILL.md`) for the full workflow.

## Quick start

```bash
cd /sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack
PD=/path/to/project

# 0. Do videos encode their condition in the filename?
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.reporting.conditions \
    probe --project-dir "$PD"
#    If not: `... conditions template --project-dir "$PD" --out "$PD/conditions.csv"`,
#    then fill it from the user's description.

# 1. Draft a spec from the clustering outputs (fills numbers + figures, marks TODOs)
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.reporting.scaffold \
    --project-dir "$PD" --out "$PD/report_spec.json"

# 2. Edit report_spec.json — replace every TODO with real, plain-language findings.

# 3. Render
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.reporting.make_report \
    --spec "$PD/report_spec.json" --out "$PD/report.pdf"
conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.reporting.make_slides \
    --spec "$PD/report_spec.json" --out "$PD/deck.pptx"
```

## Modules

| File | Role |
|---|---|
| `artifacts.py` | Locate + load a run's `summary.json`/`results/`/`plots/` (pair-level, single-animal, merged layouts) |
| `spec.py` | `ReportSpec` dataclass + `scaffold_spec()` draft-from-artifacts |
| `scaffold.py` | CLI: write a draft spec |
| `conditions.py` | CLI + helpers: probe filename→condition resolution, template/load a conditions CSV |
| `make_report.py` | Two-part PDF (matplotlib) |
| `make_slides.py` | Editable `.pptx` deck (python-pptx) |

## Design

The generators only do **layout**; all editorial content (which findings are the
headline, what each figure shows, the takeaways) lives in the spec. Re-rendering
never recomputes the pipeline. Run the LSF quantification step with
`scripts/bsub_behavior_quantification.bsub`.
