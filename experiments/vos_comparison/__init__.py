"""
vos_comparison
==============

Single-shot video object segmentation (VOS) vs. anchor-frame labeling.

The pipeline justifies sparse "anchor frame labeling" by measuring what happens
when a fully-tracked video is re-tracked **single-shot**: only the first labeled
frame is kept as a prompt, and SAM3 propagation runs across the whole video with
no further human anchors. The single-shot tracks are then scored against the
original (anchor-frame) tracks as ground truth, using per-frame IoU / Jaccard,
to locate where prediction quality collapses.

Stages (see ``cli.py``):

1. ``build``     — emit ``single_shot_vos_<project>`` (symlinked videos, first
                   anchor only) next to the source project.
2. ``track``     — run SAM3 propagation on the single-shot project via the
                   backend HTTP/SSE propagate endpoint; masks land in
                   ``masks.sqlite``.
3. ``evaluate``  — per-frame IoU vs. the anchor-based ground truth + collapse
                   detection; writes CSVs and ``results.json``.
4. ``figures``   — conference-ready figures (PDF/PNG) and LaTeX tables.

Run ``python -m experiments.vos_comparison --help`` from the repo root.
"""

from __future__ import annotations

# IoU collapse threshold (Jaccard below this counts as "lost" on that frame).
DEFAULT_IOU_THRESHOLD = 0.5
