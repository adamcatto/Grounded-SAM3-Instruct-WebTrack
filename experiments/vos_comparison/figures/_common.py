"""Shared styling, IO, and layout helpers for the figures package.

Deliberately backend-free: this package only reads the CSV/npz/json artifacts that
``analyze.py`` writes, so figures can be rendered anywhere matplotlib is installed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palette
C_SINGLE = "#C0392B"    # red   — single-shot degradation
C_ANCHOR = "#27AE60"    # green — anchor ground-truth reference (IoU == 1)
C_BAND = "#5B8DD9"      # blue  — aggregate band
C_COLLAPSE = "#E67E22"  # orange — collapse marker
C_MIN = "#8E44AD"       # purple — worst-object (min) signal
REDUCTION_COLOR = {"mean": C_SINGLE, "min": C_MIN, "object": "#16A085"}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.3,
    })


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = ("pdf", "png", "svg")
    for ext in exts:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"[figures]   {out_dir.name}/{name}.{{{','.join(exts)}}}")


# ── tiny typed-CSV helpers (no pandas) ─────────────────────────────────────
def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(x: Any) -> float:
    """Parse a CSV cell as float; '', 'None' → nan."""
    if x is None or x == "" or x == "None":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def i(x: Any):
    if x is None or x == "" or x == "None":
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def b(x: Any) -> bool:
    return str(x) == "True"


# ── loaders ────────────────────────────────────────────────────────────────
def load_results(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "results.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def load_timelines(out_dir: Path) -> dict[str, np.ndarray]:
    p = out_dir / "timelines.npz"
    if not p.is_file():
        return {}
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def video_ids(tl: dict[str, np.ndarray]) -> list[str]:
    if "__video_ids__" in tl:
        return [str(x) for x in tl["__video_ids__"]]
    # fall back to scanning frame keys
    return sorted({k[: -len("__frames")] for k in tl if k.endswith("__frames")})


def reduction_series(tl: dict[str, np.ndarray], vid: str, kind: str):
    """(frames, vals) for kind in {'mean','min'}."""
    return tl[f"{vid}__frames"], tl[f"{vid}__{kind}"]


def object_series(tl: dict[str, np.ndarray], vid: str):
    """(frames, [(oid, name, vals), …]) for one video."""
    frames = tl[f"{vid}__frames"]
    oids = [str(x) for x in tl.get(f"{vid}__objids", np.array([]))]
    names = [str(x) for x in tl.get(f"{vid}__objnames", np.array([]))]
    out = []
    for oid, nm in zip(oids, names):
        key = f"{vid}__obj__{oid}"
        if key in tl:
            out.append((oid, nm, tl[key]))
    return frames, out


def video_meta(out_dir: Path) -> dict[str, dict[str, Any]]:
    """video_id -> {name, mean_iou, collapse_frame, collapse_fraction, num_frames}."""
    meta: dict[str, dict[str, Any]] = {}
    for r in read_csv_dicts(out_dir / "video_summary.csv"):
        meta[r["video_id"]] = {
            "name": r.get("video_name", r["video_id"]),
            "mean_iou": f(r.get("mean_iou")),
            "collapse_frame": i(r.get("collapse_frame")),
            "collapse_fraction": f(r.get("collapse_fraction")),
            "num_frames": i(r.get("num_frames")),
        }
    return meta


def collapse_stats(out_dir: Path) -> list[dict[str, str]]:
    return read_csv_dicts(out_dir / "collapse_stats.csv")


# ── layout + math ──────────────────────────────────────────────────────────
def make_grid(n: int, ncols: int | None = None, panel_w: float = 3.3,
              panel_h: float = 2.1, sharey: bool = True):
    ncols = ncols or min(8, max(1, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False, sharey=sharey)
    flat = list(axes.flat)
    for ax in flat:
        ax.set_visible(False)
    return fig, flat, ncols, nrows


def normalized_resample(frames: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a per-frame series onto a common [0,1] normalized-position grid."""
    vals = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return np.full(grid.size, np.nan)
    fr = np.asarray(frames, dtype=np.float64)[finite]
    v = vals[finite]
    span = max(1e-9, fr.max() - fr.min())
    p = (fr - fr.min()) / span
    return np.interp(grid, p, v, left=v[0], right=v[-1])


def object_colors(n: int):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(k % cmap.N) for k in range(n)]
