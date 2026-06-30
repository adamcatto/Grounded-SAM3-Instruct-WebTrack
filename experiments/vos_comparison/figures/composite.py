"""Assemble the main-text Figure 5 composite from individual VOS panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from . import _common as C
from . import aggregate, representative


def _load(path: Path):
    return mpimg.imread(str(path))


def make_figure5(out_dir: Path, fig_root: Path, *, tau: float | None = None) -> None:
    out_dir = Path(out_dir)
    fig_root = Path(fig_root)
    gen = fig_root / "generated"
    gen.mkdir(parents=True, exist_ok=True)

    tl = C.load_timelines(out_dir)
    if not tl:
        print("[figures] WARNING: no timelines.npz — skipping Figure 5 composite")
        return

    results = C.load_results(out_dir)
    tau = tau if tau is not None else float(results.get("threshold", 0.3))

    # Refresh polished aggregate panels (shorter titles applied in aggregate.py)
    aggregate.band(tl, tau, "mean", fig_root / "aggregate")
    aggregate.survival(tl, tau, "mean", fig_root / "aggregate")

    rep_base = gen / "fig05_panel_d_representative"
    representative.make_representative_panel(out_dir, rep_base, tau=tau)

    # Build composite layout preview from PNG panels
    panel_paths = {
        "band": fig_root / "aggregate" / "band_mean.png",
        "survival": fig_root / "aggregate" / "survival_mean.png",
        "rep": rep_base.with_suffix(".png"),
        "summary": fig_root / "summary" / "summary_overview.png",
    }
    if not all(p.is_file() for p in (panel_paths["band"], panel_paths["survival"], panel_paths["rep"])):
        print("[figures] WARNING: missing panels for Figure 5 composite preview")
        return

    fig = C.plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.2, wspace=0.15)
    ax_band = fig.add_subplot(gs[0, :])
    ax_surv = fig.add_subplot(gs[1, 0])
    ax_rep = fig.add_subplot(gs[1, 1])
    ax_band.imshow(_load(panel_paths["band"]))
    ax_surv.imshow(_load(panel_paths["survival"]))
    ax_rep.imshow(_load(panel_paths["rep"]))
    for ax, lbl in ((ax_band, "b"), (ax_surv, "c"), (ax_rep, "d")):
        ax.set_title(lbl, loc="left", fontsize=9)
        ax.axis("off")
    fig.suptitle("Figure 5 — Single-shot VOS degradation", y=0.98, fontsize=10)
    C.save(fig, gen, "fig05_composite_preview")
