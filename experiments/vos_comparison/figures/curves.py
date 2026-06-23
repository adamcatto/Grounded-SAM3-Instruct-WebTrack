"""Per-video IoU-vs-frame curves, one figure (grid of small multiples) per analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _common as C


def _reduction_collapse_starts(out_dir: Path, reduction: str) -> dict[str, float]:
    """video_id -> collapse_start_frame for a given reduction (mean/min)."""
    out: dict[str, float] = {}
    for r in C.collapse_stats(out_dir):
        if r.get("reduction") == reduction:
            cf = C.i(r.get("collapse_start_frame"))
            if cf is not None:
                out[r["video_id"]] = cf
    return out


def _curve_panel(ax, frames, vals, tau, color, label, collapse_x=None, title=None):
    ax.set_visible(True)
    ax.plot(frames, vals, color=color, label=label)
    ax.axhline(1.0, color=C.C_ANCHOR, lw=1.0, ls="--", alpha=0.7, label="anchor (GT)")
    ax.axhline(tau, color="0.4", lw=0.8, ls=":", label=f"τ={tau:g}")
    if collapse_x is not None:
        ax.axvline(collapse_x, color=C.C_COLLAPSE, lw=1.1, ls="-", label="collapse")
    ax.set_ylim(-0.02, 1.02)
    if title:
        ax.set_title(title)


def _reduction_grid(out_dir, tl, meta, tau, kind, color, fig_dir):
    vids = C.video_ids(tl)
    if not vids:
        return
    starts = _reduction_collapse_starts(out_dir, kind)
    fig, axes, ncols, _ = C.make_grid(len(vids))
    for k, vid in enumerate(vids):
        ax = axes[k]
        frames, vals = C.reduction_series(tl, vid, kind)
        m = meta.get(vid, {})
        name = str(m.get("name", vid))[:26]
        mi = m.get("mean_iou", float("nan"))
        _curve_panel(ax, frames, vals, tau, color, f"single-shot ({kind})",
                     collapse_x=starts.get(vid),
                     title=f"{name}\n(mean IoU={mi:.2f})")
        if k % ncols == 0:
            ax.set_ylabel("IoU vs. GT")
        ax.set_xlabel("frame")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Single-shot VOS quality over time — {kind} over objects (per video)", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    C.save(fig, fig_dir, kind)


def curves_mean(out_dir, tl, meta, tau, fig_dir):
    _reduction_grid(out_dir, tl, meta, tau, "mean", C.C_SINGLE, fig_dir)


def curves_min(out_dir, tl, meta, tau, fig_dir):
    _reduction_grid(out_dir, tl, meta, tau, "min", C.C_MIN, fig_dir)


def curves_objects_overlay(out_dir, tl, meta, tau, fig_dir):
    """Per video: every object's IoU on one panel, different colors."""
    vids = C.video_ids(tl)
    if not vids:
        return
    fig, axes, ncols, _ = C.make_grid(len(vids))
    for k, vid in enumerate(vids):
        ax = axes[k]
        ax.set_visible(True)
        frames, objs = C.object_series(tl, vid)
        colors = C.object_colors(len(objs))
        for (oid, nm, vals), col in zip(objs, colors):
            ax.plot(frames, vals, color=col, lw=1.0, label=str(nm)[:14])
        ax.axhline(tau, color="0.4", lw=0.8, ls=":")
        ax.axhline(1.0, color=C.C_ANCHOR, lw=1.0, ls="--", alpha=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{str(meta.get(vid, {}).get('name', vid))[:26]}\n({len(objs)} objects)")
        if k % ncols == 0:
            ax.set_ylabel("IoU vs. GT")
        ax.set_xlabel("frame")
        if 0 < len(objs) <= 6:
            ax.legend(fontsize=6, loc="lower left", framealpha=0.6)
    fig.suptitle("Per-object single-shot IoU over time (objects overlaid per video)", y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    C.save(fig, fig_dir, "objects_overlay")


def curves_objects_facets(out_dir, tl, meta, tau, fig_dir):
    """One panel per (video, object) — every object track on its own axes."""
    vids = C.video_ids(tl)
    panels = []  # (vid, oid, name, frames, vals)
    for vid in vids:
        frames, objs = C.object_series(tl, vid)
        for oid, nm, vals in objs:
            panels.append((vid, oid, nm, frames, vals))
    if not panels:
        return
    # object collapse frames for annotation
    ocoll: dict[tuple[str, str], int] = {}
    for r in C.read_csv_dicts(out_dir / "object_summary.csv"):
        cf = C.i(r.get("collapse_frame"))
        if cf is not None:
            ocoll[(r["video_id"], r["object_id"])] = cf
    fig, axes, ncols, _ = C.make_grid(len(panels), panel_w=3.0, panel_h=1.9)
    for k, (vid, oid, nm, frames, vals) in enumerate(panels):
        ax = axes[k]
        vname = str(meta.get(vid, {}).get("name", vid))[:16]
        _curve_panel(ax, frames, vals, tau, "#16A085", "object",
                     collapse_x=ocoll.get((vid, oid)),
                     title=f"{vname}:{str(nm)[:12]}")
        if k % ncols == 0:
            ax.set_ylabel("IoU")
        ax.set_xlabel("frame")
    fig.suptitle("Single-shot IoU per object, plotted independently", y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    C.save(fig, fig_dir, "objects_facets")
