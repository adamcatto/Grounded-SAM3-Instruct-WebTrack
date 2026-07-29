"""Report specification: the small, human-curated JSON that drives a focused
report and slide deck.

Philosophy: the *pipeline* discovers behavioral structure; the *agent* decides
what is worth putting in front of a reader. The ``ReportSpec`` captures that
editorial judgment (which 3-5 findings are the headline, what each figure
actually shows, the one-line takeaways) while the generators guarantee the
layout. ``scaffold_spec`` pre-fills a draft from the pipeline artifacts so the
agent starts from real numbers and real figure paths, then curates.

The spec is deliberately plain JSON (not code) so it can be reviewed, diffed,
and hand-edited without touching the generators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .artifacts import ClusteringOutputs, top_enriched_features


@dataclass
class Finding:
    heading: str                      # short, declarative ("SH residents are hyperactive")
    figure: str | None = None         # path (abs, or relative to plots/) of the supporting figure
    interpretation: str = ""          # 2-4 plain-language sentences
    stat: str = ""                    # the number ("Cohen's d = 0.27, p < 1e-6, 17/20 features sig.")
    annotation: str | None = None     # optional short callout drawn on/near the figure


@dataclass
class DetailSection:
    title: str
    type: str = "text"                # text | image | image_grid | two_image
    body: str = ""                    # for type=text
    image: str | None = None          # for type=image
    caption: str = ""
    images: list = field(default_factory=list)   # for image_grid/two_image: [[path, label], ...]


@dataclass
class ReportSpec:
    project_dir: str
    output_dir: str | None = None
    title: str = "Behavioral Quantification Report"
    subtitle: str = ""
    date: str = ""
    # -- Part 1: focused brief --
    dataset: str = ""                             # one-line dataset description
    conditions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)     # n_videos, n_windows, n_features, n_clusters
    headline_bullets: list = field(default_factory=list)   # 3-5 short bullets
    key_figure: dict = field(default_factory=dict)         # {image, caption, annotation}
    findings: list = field(default_factory=list)           # list[Finding]
    caveats: list = field(default_factory=list)
    implications: str = ""
    # -- Part 2: full detail --
    detail_sections: list = field(default_factory=list)    # list[DetailSection]
    methods: dict = field(default_factory=dict)
    # -- misc --
    references: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "ReportSpec":
        data = json.loads(Path(path).read_text())
        data["findings"] = [Finding(**f) if isinstance(f, dict) else f
                            for f in data.get("findings", [])]
        data["detail_sections"] = [DetailSection(**s) if isinstance(s, dict) else s
                                   for s in data.get("detail_sections", [])]
        return cls(**data)

    def findings_as_objs(self) -> list[Finding]:
        return [f if isinstance(f, Finding) else Finding(**f) for f in self.findings]

    def sections_as_objs(self) -> list[DetailSection]:
        return [s if isinstance(s, DetailSection) else DetailSection(**s) for s in self.detail_sections]


def scaffold_spec(project_dir: str | Path, output_dir: str | Path | None = None) -> ReportSpec:
    """Build a draft ReportSpec from a completed clustering run.

    This fills in the numbers, discovers the best available headline figure, and
    drafts finding STUBS from the strongest feature effects. The stubs are marked
    with 'TODO' interpretations on purpose: the agent must replace them with real,
    plain-language reads of the data. Nothing here invents a scientific claim.
    """
    outputs = ClusteringOutputs.locate(project_dir, output_dir)
    summary = outputs.summary()
    kind, comparison = outputs.comparison()
    config = summary.get("config", {}) or {}

    # Single-animal runs emit cluster_description.json (in place of a housing/
    # condition comparison) and use ~12 individual features rather than ~32.
    single_animal = (
        bool(summary.get("single_animal"))
        or bool((config or {}).get("single_animal"))
        or kind == "cluster_description"
        or "single" in str(kind or "")
    )

    stats = {
        "n_windows": summary.get("n_windows") or summary.get("n_samples"),
        "n_features": summary.get("n_features"),
        "n_clusters": summary.get("n_clusters"),
    }

    # Best-available "key figure": condition/housing/role UMAP, else cluster UMAP.
    key_fig = outputs.first_plot(
        "umap/umap_by_condition_group.png", "umap/umap_by_housing.png",
        "umap/umap_by_role.png", "umap_by_housing.png", "umap_by_condition.png",
        "single_animal/umap/umap_by_role.png",
        "umap/umap_by_cluster.png", "umap_by_cluster.png",
    )

    # Draft finding stubs from the strongest effects.
    findings: list[Finding] = []
    top = top_enriched_features(kind, comparison, n=4)
    enrichment_fig = outputs.first_plot(
        "feature_enrichment.png", "enrichment_bars.png",
        "paired_animals/condition_groups", "single_animal/feature_heatmap.png",
        "feature_heatmap.png",
    )
    for feat in top[:3]:
        pval = feat.get("pvalue")
        stat = f"effect = {feat['effect']:+.2f}"
        if pval is not None:
            stat += f", p = {pval:.1e}"
        findings.append(Finding(
            heading=f"TODO: interpret '{feat['feature']}' ({feat['direction']})",
            figure=enrichment_fig,
            interpretation="TODO: 2-4 plain-language sentences describing what this "
                           "difference means biologically. Replace this stub.",
            stat=stat,
        ))
    if not findings:
        findings.append(Finding(
            heading="TODO: headline finding 1",
            figure=key_fig,
            interpretation="TODO: describe the single most important result.",
            stat="",
        ))

    # Draft Part-2 detail sections from whatever plots exist.
    detail: list[DetailSection] = []
    if outputs.first_plot("feature_heatmap.png", "single_animal/feature_heatmap.png"):
        detail.append(DetailSection(
            title="Cluster feature heatmap",
            type="image",
            image=outputs.first_plot("feature_heatmap.png", "single_animal/feature_heatmap.png"),
            caption="z-scored feature centroids per cluster.",
        ))
    etho = outputs.ethogram_example()
    if etho:
        detail.append(DetailSection(title="Example ethogram", type="image",
                                    image=etho, caption="Per-video behavioral timeline."))

    spec = ReportSpec(
        project_dir=str(project_dir),
        output_dir=str(outputs.output_dir),
        title="Behavioral Quantification Report",
        subtitle="Single-animal analysis" if single_animal else "Pair-level analysis",
        dataset="TODO: one-line dataset description (experiments, N videos, conditions).",
        conditions=[],
        stats={k: v for k, v in stats.items() if v is not None},
        headline_bullets=["TODO: 3-5 short headline findings, most important first."],
        key_figure={"image": key_fig or "",
                    "caption": "TODO: what this embedding shows.",
                    "annotation": None},
        findings=findings,
        caveats=["TODO: 2-4 real limitations (camera view, window length, N imbalance)."],
        implications="TODO: one paragraph on what these results mean for the project.",
        detail_sections=detail,
        methods={
            "window_size": config.get("window_size"),
            "stride": config.get("stride"),
            "n_neighbors": config.get("n_neighbors"),
            "leiden_resolution": config.get("leiden_resolution"),
            "normalize_method": config.get("normalize_method"),
            "clustering_method": summary.get("clustering_method"),
            "embedding_method": summary.get("embedding_method"),
        },
    )
    return spec
