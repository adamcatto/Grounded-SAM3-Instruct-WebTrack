"""Load behavior-quantification pipeline outputs for reporting.

The clustering pipeline (``downstream_analysis.clustering``) writes its results
in one of two layouts:

  Single-project:  <project>/analysis_of_tracking_data/clustering/{summary.json,
                   plots/, results/}
  Merged project:  <output_dir>/{summary.json, plots/, results/}   (output_dir
                   is usually the project root, e.g. the merged_alberto_* dirs)

This module locates that output directory and provides thin, defensive loaders
so the report/slide generators never have to know the on-disk schema. Every
loader degrades gracefully (returns ``{}`` / ``[]`` / ``None``) when an artifact
is absent, so the same code path works for pair-level, single-animal, and merged
runs even though each emits a slightly different set of files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Comparison JSONs, in priority order. Different run modes emit different names:
#   housing_comparison.json     -> pair-level social/housing project
#   condition_comparison.json   -> merged multi-project run
#   cluster_description.json     -> single-animal descriptive run
COMPARISON_FILES = (
    "condition_comparison.json",
    "housing_comparison.json",
    "cluster_description.json",
)


class ClusteringOutputs:
    """Handle to a completed clustering run's artifacts."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        # Plot layout varies: pair-level runs write PNGs flat under plots/, while
        # single-animal runs nest them under plots/png/ (with a sibling svg/).
        # Prefer plots/png/ when it holds images, else plots/.
        plots = self.output_dir / "plots"
        png = plots / "png"
        if png.is_dir() and any(png.rglob("*.png")):
            self.plots_dir = png
        else:
            self.plots_dir = plots

    # -- discovery ---------------------------------------------------------
    @classmethod
    def locate(cls, project_dir: str | os.PathLike, output_dir: str | os.PathLike | None = None) -> "ClusteringOutputs":
        """Find the clustering output dir for a project.

        Pass ``output_dir`` explicitly for merged runs. Otherwise this checks the
        canonical single-project location, then the project root itself.
        """
        project_dir = Path(project_dir)
        if output_dir is not None:
            return cls(Path(output_dir))
        candidates = [
            project_dir / "analysis_of_tracking_data" / "clustering",
            project_dir,
        ]
        for c in candidates:
            if (c / "summary.json").is_file() or (c / "plots").is_dir():
                return cls(c)
        # Fall back to the canonical location even if empty (clearer errors).
        return cls(candidates[0])

    # -- JSON loaders ------------------------------------------------------
    def _load_json(self, *relparts: str) -> Any:
        p = self.output_dir.joinpath(*relparts)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def summary(self) -> dict:
        return self._load_json("summary.json") or {}

    def cluster_summary(self) -> dict:
        return self._load_json("results", "cluster_summary.json") or {}

    def comparison(self) -> tuple[str | None, dict]:
        """Return (kind, data) for whichever comparison file exists."""
        for name in COMPARISON_FILES:
            data = self._load_json("results", name)
            if data is not None:
                return name.replace(".json", ""), data
        return None, {}

    # -- plot discovery ----------------------------------------------------
    def has_plot(self, relpath: str) -> bool:
        return (self.plots_dir / relpath).is_file()

    def plot(self, relpath: str) -> str | None:
        """Absolute path to a plot if it exists, else None."""
        p = self.plots_dir / relpath
        return str(p) if p.is_file() else None

    def list_plots(self) -> list[str]:
        """All PNGs under plots/, as paths relative to plots_dir, sorted."""
        if not self.plots_dir.is_dir():
            return []
        out = []
        for root, _dirs, files in os.walk(self.plots_dir):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    rel = os.path.relpath(os.path.join(root, f), self.plots_dir)
                    out.append(rel)
        return sorted(out)

    def first_plot(self, *candidates: str) -> str | None:
        """First existing plot among candidate relative paths."""
        for c in candidates:
            p = self.plot(c)
            if p:
                return p
        return None

    def ethogram_example(self) -> str | None:
        etho = self.plots_dir / "ethograms"
        if etho.is_dir():
            pngs = sorted(p for p in etho.iterdir() if p.suffix == ".png")
            if pngs:
                return str(pngs[0])
        return None


def top_enriched_features(comparison_kind: str | None, comparison: dict, n: int = 8) -> list[dict]:
    """Best-effort extraction of the strongest feature effects for headline
    drafting. Returns a list of dicts with keys: feature, effect, pvalue,
    direction. Shapes vary by run mode, so this is intentionally forgiving.
    """
    if not comparison:
        return []

    rows: list[dict] = []

    def _push(feature, effect, pvalue=None):
        try:
            eff = float(effect)
        except (TypeError, ValueError):
            return
        rows.append({
            "feature": str(feature),
            "effect": eff,
            "abs_effect": abs(eff),
            "pvalue": pvalue,
            "direction": "higher" if eff >= 0 else "lower",
        })

    fe = comparison.get("feature_enrichment")
    if isinstance(fe, list):
        for item in fe:
            if isinstance(item, dict):
                _push(item.get("feature") or item.get("name"),
                      item.get("cohens_d", item.get("effect_size", item.get("d"))),
                      item.get("pvalue", item.get("p")))
    elif isinstance(fe, dict):
        for feat, item in fe.items():
            if isinstance(item, dict):
                _push(feat, item.get("cohens_d", item.get("effect_size", item.get("d"))),
                      item.get("pvalue", item.get("p")))
            else:
                _push(feat, item)

    pft = comparison.get("per_feature_tests")
    if not rows and isinstance(pft, dict):
        for feat, item in pft.items():
            if isinstance(item, dict):
                _push(feat, item.get("cohens_d", item.get("effect_size", item.get("d"))),
                      item.get("pvalue", item.get("p")))

    rows.sort(key=lambda r: r["abs_effect"], reverse=True)
    return rows[:n]
