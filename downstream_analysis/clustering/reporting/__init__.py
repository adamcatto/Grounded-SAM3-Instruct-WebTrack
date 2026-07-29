"""Focused behavior-quantification reporting: two-part PDF + editable slide deck.

The clustering pipeline finds behavioral structure; this package turns a
completed run into a *focused* report (a short brief a collaborator reads on its
own, backed by a longer detail section) and a matching slide deck -- driven by a
small human-curated ReportSpec rather than dumping every plot.

See the ``behavior-quant-reporting`` agent skill for the end-to-end workflow.
"""

from .artifacts import ClusteringOutputs, top_enriched_features
from .spec import ReportSpec, Finding, DetailSection, scaffold_spec

# Note: submodules (conditions, make_report, make_slides, scaffold) are imported
# on demand — importing them eagerly here trips a runpy warning when they are
# run as `python -m ...reporting.<submodule>`. Use
# `from downstream_analysis.clustering.reporting import conditions`.

__all__ = [
    "ClusteringOutputs", "top_enriched_features",
    "ReportSpec", "Finding", "DetailSection", "scaffold_spec",
]
