#!/usr/bin/env python3
"""Generate a focused, editable slide deck (.pptx) from a ReportSpec.

Mirrors Part 1 of the report but slide-shaped: title, study design, methods in
one line, one slide per headline finding (big figure + one-line takeaway), and
an implications slide. Editable in PowerPoint/Keynote/Google Slides so a human
can polish wording before presenting.

Same spec, same figures as make_report.py -- the deck is never out of sync with
the brief. Requires python-pptx.

Usage:
    python -m downstream_analysis.clustering.reporting.make_slides \
        --spec spec.json --out deck.pptx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

from .spec import ReportSpec, Finding
from .artifacts import ClusteringOutputs

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0x1F, 0x5F, 0xA8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CALLOUT = RGBColor(0xEE, 0xF4, 0xFB)

SW, SH = Inches(13.333), Inches(7.5)   # 16:9


def _resolve(path, plots_dir):
    if not path:
        return None
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return str(p)
    cand = Path(plots_dir) / path
    if cand.is_file():
        return str(cand)
    return str(p) if p.is_file() else None


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, x, y, w, h, text, size=18, bold=False, color=INK,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        f = r.font
        f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color
        f.name = "Calibri"
    return tb


def _accent_bar(slide, y=Inches(0.0), h=Inches(0.12)):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), y, SW, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = ACCENT
    shp.line.fill.background()
    return shp


def _fit_image(slide, img_path, box_x, box_y, box_w, box_h):
    """Add image scaled to fit inside a box, centered."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    box_ar = box_w / box_h
    img_ar = iw / ih
    if img_ar > box_ar:
        w = box_w; h = Emu(int(box_w / img_ar))
    else:
        h = box_h; w = Emu(int(box_h * img_ar))
    x = box_x + Emu(int((box_w - w) / 2))
    y = box_y + Emu(int((box_h - h) / 2))
    slide.shapes.add_picture(img_path, x, y, width=w, height=h)


def _callout_box(slide, x, y, w, h, title, body):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = CALLOUT
    shp.line.color.rgb = ACCENT; shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]
    if title:
        r = p.add_run(); r.text = title + "  "
        r.font.bold = True; r.font.size = Pt(13); r.font.color.rgb = ACCENT
    r = p.add_run(); r.text = body
    r.font.size = Pt(13); r.font.color.rgb = INK


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------

def title_slide(prs, spec: ReportSpec):
    s = _blank(prs)
    _accent_bar(s, Inches(2.55), Inches(0.06))
    _text(s, Inches(0.8), Inches(2.7), Inches(11.7), Inches(1.6),
          spec.title, size=40, bold=True, color=INK)
    if spec.subtitle:
        _text(s, Inches(0.8), Inches(4.1), Inches(11.7), Inches(0.8),
              spec.subtitle, size=20, color=MUTED)
    meta = [m for m in [spec.dataset if not spec.dataset.startswith("TODO") else "", spec.date] if m]
    if meta:
        _text(s, Inches(0.8), Inches(4.9), Inches(11.7), Inches(0.8),
              "   |   ".join(meta), size=14, color=MUTED)


def at_a_glance_slide(prs, spec: ReportSpec):
    s = _blank(prs)
    _accent_bar(s)
    _text(s, Inches(0.6), Inches(0.35), Inches(12), Inches(0.8),
          "At a glance", size=30, bold=True, color=INK)
    stats = spec.stats or {}
    items = [("Videos", stats.get("n_videos")), ("Windows", stats.get("n_windows")),
             ("Features", stats.get("n_features")), ("Clusters", stats.get("n_clusters"))]
    items = [(k, v) for k, v in items if v is not None]
    for i, (k, v) in enumerate(items):
        x = Inches(0.6 + i * 3.0)
        _text(s, x, Inches(1.4), Inches(2.8), Inches(0.8),
              f"{v:,}" if isinstance(v, int) else str(v), size=32, bold=True, color=ACCENT)
        _text(s, x, Inches(2.2), Inches(2.8), Inches(0.5), k.upper(), size=13, color=MUTED)
    y = 3.1
    if spec.conditions:
        _text(s, Inches(0.6), Inches(y), Inches(12), Inches(0.5),
              "Conditions: " + ", ".join(str(c) for c in spec.conditions), size=15, color=INK)
        y += 0.6
    body = "\n".join(f"•  {b}" for b in (spec.headline_bullets or []))
    _text(s, Inches(0.6), Inches(y), Inches(12.1), Inches(7.3 - y), body, size=17, color=INK)


def finding_slide(prs, spec: ReportSpec, finding: Finding, idx: int, plots_dir):
    s = _blank(prs)
    _accent_bar(s)
    _text(s, Inches(0.6), Inches(0.3), Inches(12.1), Inches(1.0),
          f"{idx}. {finding.heading}", size=24, bold=True, color=INK)
    img = _resolve(finding.figure, plots_dir)
    if img:
        _fit_image(s, img, Inches(0.6), Inches(1.5), Inches(8.2), Inches(5.0))
        # right rail: interpretation + stat
        _text(s, Inches(9.0), Inches(1.5), Inches(3.9), Inches(3.6),
              finding.interpretation, size=14, color=INK)
        if finding.stat and not finding.stat.startswith("TODO"):
            _callout_box(s, Inches(9.0), Inches(5.4), Inches(3.9), Inches(1.1),
                         "", finding.stat)
    else:
        _text(s, Inches(0.6), Inches(1.6), Inches(12.1), Inches(4.5),
              finding.interpretation, size=18, color=INK)
        if finding.stat and not finding.stat.startswith("TODO"):
            _callout_box(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(1.0),
                         "The numbers:", finding.stat)


def key_figure_slide(prs, spec: ReportSpec, plots_dir):
    kf = spec.key_figure or {}
    img = _resolve(kf.get("image"), plots_dir)
    if not img:
        return
    s = _blank(prs)
    _accent_bar(s)
    _text(s, Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.8),
          "The key result", size=24, bold=True, color=INK)
    _fit_image(s, img, Inches(0.6), Inches(1.3), Inches(9.0), Inches(5.6))
    cap = kf.get("caption", "")
    if cap and not cap.startswith("TODO"):
        _text(s, Inches(9.8), Inches(1.5), Inches(3.2), Inches(3.0), cap, size=14,
              color=MUTED, italic=True)
    if kf.get("annotation"):
        _callout_box(s, Inches(9.8), Inches(4.8), Inches(3.2), Inches(1.8),
                     "Look at:", kf["annotation"])


def implications_slide(prs, spec: ReportSpec):
    if not (spec.implications or spec.caveats):
        return
    s = _blank(prs)
    _accent_bar(s)
    _text(s, Inches(0.6), Inches(0.35), Inches(12), Inches(0.8),
          "What it means", size=30, bold=True, color=INK)
    y = 1.4
    if spec.implications and not spec.implications.startswith("TODO"):
        _text(s, Inches(0.6), Inches(y), Inches(12.1), Inches(2.5),
              spec.implications, size=18, color=INK)
        y = 4.2
    if spec.caveats:
        _text(s, Inches(0.6), Inches(y), Inches(12), Inches(0.5),
              "Caveats", size=18, bold=True, color=MUTED); y += 0.55
        body = "\n".join(f"•  {c}" for c in spec.caveats)
        _text(s, Inches(0.6), Inches(y), Inches(12.1), Inches(7.2 - y), body,
              size=14, color=INK)


def build_deck(spec: ReportSpec, out_path: str | Path) -> str:
    outputs = ClusteringOutputs.locate(spec.project_dir, spec.output_dir)
    plots_dir = outputs.plots_dir
    prs = Presentation()
    prs.slide_width = SW; prs.slide_height = SH
    title_slide(prs, spec)
    at_a_glance_slide(prs, spec)
    key_figure_slide(prs, spec, plots_dir)
    for i, f in enumerate(spec.findings_as_objs(), start=1):
        finding_slide(prs, spec, f, i, plots_dir)
    implications_slide(prs, spec)
    out_path = str(out_path)
    prs.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = ReportSpec.from_json(args.spec)
    out = build_deck(spec, args.out)
    print(f"Deck written: {out}")


if __name__ == "__main__":
    main()
