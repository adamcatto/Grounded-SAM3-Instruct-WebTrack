#!/usr/bin/env python3
"""Generate a focused, two-part behavioral-quantification report PDF.

Part 1 -- FOCUSED BRIEF (~5 pages): cover, at-a-glance, one key figure, one page
per headline finding (figure + plain-language read + the stat, together), and a
"what it means / caveats" page. A collaborator reads only Part 1.

Part 2 -- FULL DETAIL: methods, all clusters/comparisons, and a technical
appendix. Reference material, clearly demarcated.

The report is driven entirely by a ReportSpec (see spec.py). This script only
handles layout; all editorial content comes from the spec. Uses matplotlib only
(no extra deps), but with a clean sans-serif, hierarchy, and figure+text on the
same page -- deliberately unlike the old monospace page-per-plot dump.

Usage:
    python -m downstream_analysis.clustering.reporting.make_report \
        --spec spec.json --out report.pdf
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import numpy as np

from .spec import ReportSpec, Finding, DetailSection
from .artifacts import ClusteringOutputs

# Portrait US-letter.
PAGE = (8.5, 11.0)
INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#1f5fa8"
RULE = "#c9c9c9"
CALLOUT_BG = "#eef4fb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _blank_page(pdf):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    return fig, ax


def _wrap(text, width):
    out = []
    for para in (text or "").split("\n"):
        out.append("\n".join(textwrap.wrap(para, width=width)) if para.strip() else "")
    return "\n".join(out)


def _resolve(path, plots_dir):
    """Resolve a figure path that may be absolute or relative to plots/."""
    if not path:
        return None
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return str(p)
    cand = Path(plots_dir) / path
    if cand.is_file():
        return str(cand)
    return str(p) if p.is_file() else None


def _draw_image(ax, img_path):
    img = np.asarray(Image.open(img_path))
    ax.imshow(img)
    ax.axis("off")


def _header(fig, part_label, page_title):
    fig.text(0.08, 0.955, part_label, fontsize=8.5, color=ACCENT, weight="bold",
             ha="left", va="center")
    fig.text(0.92, 0.955, page_title, fontsize=8.5, color=MUTED,
             ha="right", va="center")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.945, 0.945], color=RULE, lw=0.8,
                              transform=fig.transFigure))


def _callout(fig, x, y, w, h, title, body, bg=CALLOUT_BG, edge=ACCENT):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=bg, edgecolor=edge, lw=1.0,
                                transform=ax.transAxes, clip_on=False))
    if title:
        ax.text(0.04, 0.90, title, fontsize=9.5, weight="bold", color=ACCENT,
                va="top", ha="left", transform=ax.transAxes)
    ax.text(0.04, 0.72 if title else 0.90, body, fontsize=9, color=INK,
            va="top", ha="left", transform=ax.transAxes, wrap=True)


# ---------------------------------------------------------------------------
# Part 1 pages
# ---------------------------------------------------------------------------

def cover_page(pdf, spec: ReportSpec):
    fig, ax = _blank_page(pdf)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.72, 0.72], color=ACCENT, lw=2.5,
                              transform=fig.transFigure))
    fig.text(0.08, 0.66, spec.title, fontsize=26, weight="bold", color=INK,
             ha="left", va="top")
    if spec.subtitle:
        fig.text(0.08, 0.58, spec.subtitle, fontsize=14, color=MUTED, ha="left", va="top")
    meta = []
    if spec.dataset and not spec.dataset.startswith("TODO"):
        meta.append(spec.dataset)
    if spec.date:
        meta.append(spec.date)
    if meta:
        fig.text(0.08, 0.50, _wrap("  |  ".join(meta), 80), fontsize=10.5,
                 color=INK, ha="left", va="top")
    fig.text(0.08, 0.10,
             "Part 1 (Focused Brief) is intended to be read on its own.\n"
             "Part 2 (Full Detail) is reference material.",
             fontsize=9, color=MUTED, ha="left", va="bottom", style="italic")
    pdf.savefig(fig, dpi=150); plt.close(fig)


def at_a_glance_page(pdf, spec: ReportSpec):
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 1 · FOCUSED BRIEF", "At a glance")
    fig.text(0.08, 0.90, "At a glance", fontsize=18, weight="bold", color=INK,
             ha="left", va="top")

    # stats strip
    stats = spec.stats or {}
    labels = [("Videos", stats.get("n_videos")), ("Windows", stats.get("n_windows")),
              ("Features", stats.get("n_features")), ("Clusters", stats.get("n_clusters"))]
    labels = [(k, v) for k, v in labels if v is not None]
    n = max(len(labels), 1)
    for i, (k, v) in enumerate(labels):
        x = 0.08 + i * (0.84 / n)
        fig.text(x, 0.83, f"{v:,}" if isinstance(v, int) else str(v),
                 fontsize=20, weight="bold", color=ACCENT, ha="left", va="top")
        fig.text(x, 0.785, k.upper(), fontsize=8.5, color=MUTED, ha="left", va="top")

    y = 0.74
    if spec.conditions:
        fig.text(0.08, y, "Conditions: " + ", ".join(str(c) for c in spec.conditions),
                 fontsize=10, color=INK, ha="left", va="top")
        y -= 0.04

    fig.text(0.08, y, "Headline findings", fontsize=13, weight="bold", color=INK,
             ha="left", va="top")
    y -= 0.03
    for b in (spec.headline_bullets or []):
        fig.text(0.10, y, "•", fontsize=11, color=ACCENT, ha="left", va="top")
        wrapped = _wrap(str(b), 92)
        fig.text(0.13, y, wrapped, fontsize=10.5, color=INK, ha="left", va="top",
                 linespacing=1.35)
        y -= 0.035 + 0.022 * wrapped.count("\n")
    pdf.savefig(fig, dpi=150); plt.close(fig)


def key_figure_page(pdf, spec: ReportSpec, plots_dir):
    kf = spec.key_figure or {}
    img = _resolve(kf.get("image"), plots_dir)
    if not img:
        return
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 1 · FOCUSED BRIEF", "Key result")
    fig.text(0.08, 0.90, "The key result", fontsize=18, weight="bold", color=INK,
             ha="left", va="top")
    ax = fig.add_axes([0.08, 0.30, 0.84, 0.55]); _draw_image(ax, img)
    cap = kf.get("caption", "")
    if cap and not cap.startswith("TODO"):
        fig.text(0.08, 0.27, _wrap(cap, 100), fontsize=9.5, color=MUTED,
                 ha="left", va="top", style="italic")
    if kf.get("annotation"):
        _callout(fig, 0.08, 0.10, 0.84, 0.12, "What to look at", kf["annotation"])
    pdf.savefig(fig, dpi=150); plt.close(fig)


def finding_page(pdf, spec: ReportSpec, finding: Finding, idx: int, plots_dir):
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 1 · FOCUSED BRIEF", f"Finding {idx}")
    fig.text(0.08, 0.91, f"{idx}. {finding.heading}", fontsize=15.5, weight="bold",
             color=INK, ha="left", va="top", wrap=True)

    img = _resolve(finding.figure, plots_dir)
    if img:
        ax = fig.add_axes([0.08, 0.40, 0.84, 0.44]); _draw_image(ax, img)
        if finding.annotation:
            fig.text(0.08, 0.375, "→ " + finding.annotation, fontsize=9,
                     color=ACCENT, ha="left", va="top", style="italic")
    interp_top = 0.34 if img else 0.82
    fig.text(0.08, interp_top, _wrap(finding.interpretation, 96), fontsize=10.5,
             color=INK, ha="left", va="top", linespacing=1.4)
    if finding.stat and not finding.stat.startswith("TODO"):
        _callout(fig, 0.08, 0.08, 0.84, 0.09, "The numbers", finding.stat,
                 bg="#f3f0fb", edge="#5b3fa8")
    pdf.savefig(fig, dpi=150); plt.close(fig)


def implications_page(pdf, spec: ReportSpec):
    if not (spec.implications or spec.caveats):
        return
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 1 · FOCUSED BRIEF", "What it means")
    fig.text(0.08, 0.90, "What it means", fontsize=18, weight="bold", color=INK,
             ha="left", va="top")
    y = 0.83
    if spec.implications and not spec.implications.startswith("TODO"):
        fig.text(0.08, y, _wrap(spec.implications, 96), fontsize=10.5, color=INK,
                 ha="left", va="top", linespacing=1.4)
        y = 0.55
    if spec.caveats:
        fig.text(0.08, y, "Caveats", fontsize=13, weight="bold", color=INK,
                 ha="left", va="top"); y -= 0.035
        for c in spec.caveats:
            wrapped = _wrap(str(c), 92)
            fig.text(0.10, y, "•", fontsize=11, color=MUTED, ha="left", va="top")
            fig.text(0.13, y, wrapped, fontsize=10, color=INK, ha="left", va="top",
                     linespacing=1.35)
            y -= 0.035 + 0.022 * wrapped.count("\n")
    pdf.savefig(fig, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2 pages
# ---------------------------------------------------------------------------

def part2_divider(pdf, spec: ReportSpec):
    fig, _ = _blank_page(pdf)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.56, 0.56], color=ACCENT, lw=2.5,
                              transform=fig.transFigure))
    fig.text(0.08, 0.52, "Part 2 — Full Detail", fontsize=24, weight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.08, 0.45,
             "Methods, complete cluster and comparison catalog, and technical\n"
             "appendix. Reference material supporting the brief above.",
             fontsize=11, color=MUTED, ha="left", va="top", linespacing=1.4)
    pdf.savefig(fig, dpi=150); plt.close(fig)


def detail_section(pdf, section: DetailSection, plots_dir):
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 2 · FULL DETAIL", section.title)
    fig.text(0.08, 0.90, section.title, fontsize=15, weight="bold", color=INK,
             ha="left", va="top")

    if section.type == "text":
        fig.text(0.08, 0.84, _wrap(section.body, 98), fontsize=9.8, color=INK,
                 ha="left", va="top", linespacing=1.4)
    elif section.type == "image":
        img = _resolve(section.image, plots_dir)
        if img:
            ax = fig.add_axes([0.06, 0.12, 0.88, 0.72]); _draw_image(ax, img)
        if section.caption:
            fig.text(0.08, 0.09, _wrap(section.caption, 100), fontsize=9,
                     color=MUTED, ha="left", va="top", style="italic")
    elif section.type == "two_image":
        for i, item in enumerate(section.images[:2]):
            path, label = (item + [""])[:2] if isinstance(item, list) else (item, "")
            img = _resolve(path, plots_dir)
            if img:
                ax = fig.add_axes([0.05 + i * 0.48, 0.20, 0.44, 0.60])
                _draw_image(ax, img)
                if label:
                    ax.set_title(label, fontsize=9, color=INK)
    elif section.type == "image_grid":
        pos = [[0.06, 0.46, 0.42, 0.36], [0.52, 0.46, 0.42, 0.36],
               [0.06, 0.08, 0.42, 0.36], [0.52, 0.08, 0.42, 0.36]]
        for i, item in enumerate(section.images[:4]):
            path, label = (item + [""])[:2] if isinstance(item, list) else (item, "")
            img = _resolve(path, plots_dir)
            if img:
                ax = fig.add_axes(pos[i]); _draw_image(ax, img)
                if label:
                    ax.set_title(label, fontsize=8, color=INK)
    pdf.savefig(fig, dpi=150); plt.close(fig)


def methods_page(pdf, spec: ReportSpec):
    if not spec.methods:
        return
    fig, _ = _blank_page(pdf)
    _header(fig, "PART 2 · FULL DETAIL", "Methods")
    fig.text(0.08, 0.90, "Methods & parameters", fontsize=15, weight="bold",
             color=INK, ha="left", va="top")
    y = 0.83
    for k, v in spec.methods.items():
        if v is None:
            continue
        fig.text(0.10, y, f"{k.replace('_', ' ')}:", fontsize=10, color=MUTED,
                 ha="left", va="top")
        fig.text(0.45, y, str(v), fontsize=10, color=INK, ha="left", va="top")
        y -= 0.03
    if spec.references:
        y -= 0.03
        fig.text(0.08, y, "References", fontsize=12, weight="bold", color=INK,
                 ha="left", va="top"); y -= 0.03
        for r in spec.references:
            wrapped = _wrap(str(r), 96)
            fig.text(0.10, y, wrapped, fontsize=8.5, color=INK, ha="left", va="top")
            y -= 0.028 + 0.02 * wrapped.count("\n")
    pdf.savefig(fig, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_report(spec: ReportSpec, out_path: str | Path) -> str:
    outputs = ClusteringOutputs.locate(spec.project_dir, spec.output_dir)
    plots_dir = outputs.plots_dir
    from matplotlib.backends.backend_pdf import PdfPages
    out_path = str(out_path)
    with PdfPages(out_path) as pdf:
        # Part 1
        cover_page(pdf, spec)
        at_a_glance_page(pdf, spec)
        key_figure_page(pdf, spec, plots_dir)
        for i, f in enumerate(spec.findings_as_objs(), start=1):
            finding_page(pdf, spec, f, i, plots_dir)
        implications_page(pdf, spec)
        # Part 2
        sections = spec.sections_as_objs()
        if sections or spec.methods:
            part2_divider(pdf, spec)
        for s in sections:
            detail_section(pdf, s, plots_dir)
        methods_page(pdf, spec)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="Path to ReportSpec JSON")
    ap.add_argument("--out", required=True, help="Output PDF path")
    args = ap.parse_args()
    spec = ReportSpec.from_json(args.spec)
    out = build_report(spec, args.out)
    size = os.path.getsize(out) / 1024 / 1024
    print(f"Report written: {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
