"""Matplotlib figures for locomotion distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .figio import save_figure  # noqa: E402


def _finite(vals: np.ndarray) -> np.ndarray:
    return vals[np.isfinite(vals)]


def plot_overlay_histogram(
    series_by_label: Mapping[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    outfile: Path,
    bins: int = 48,
    density: bool = True,
    figsize: tuple[float, float] = (9, 5.5),
    colors: Mapping[str, str] | None = None,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for label, vals in sorted(series_by_label.items(), key=lambda x: x[0].lower()):
        v = _finite(vals)
        if v.size == 0:
            continue
        kw = dict(bins=bins, alpha=0.45, density=density, label=f"{label} (n={v.size})")
        if colors and label in colors:
            kw["color"] = colors[label]
        ax.hist(v, **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save_figure(fig, outfile, dpi=140)
    plt.close(fig)


def plot_overlay_cdf(
    series_by_label: Mapping[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    outfile: Path,
    figsize: tuple[float, float] = (9, 5.5),
    colors: Mapping[str, str] | None = None,
    log_x: bool = False,
) -> None:
    """Empirical CDF (ECDF) overlay per label — step functions after sorted samples."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, vals in sorted(series_by_label.items(), key=lambda x: x[0].lower()):
        v = np.sort(_finite(vals))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1, dtype=np.float64) / v.size
        kw = dict(where="post", label=f"{label} (n={v.size})", linewidth=2.0, alpha=0.88)
        if colors and label in colors:
            kw["color"] = colors[label]
        ax.step(v, y, **kw)
    ax.set_title(title if title.endswith("(CDF)") else f"{title} (CDF)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative probability")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#333333", linewidth=0.6, linestyle="--", alpha=0.6)
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, outfile, dpi=140)
    plt.close(fig)

def plot_bimodal_moving_fraction(
    names: list[str],
    fractions: list[float | None],
    outfile: Path,
    title: str = "Estimated fraction of high-locomotion chunks (GMM threshold)",
) -> None:
    xs = np.arange(len(names))
    ys: list[float] = []
    colors: list[str] = []
    for y in fractions:
        if y is not None and np.isfinite(y):
            ys.append(float(y))
            colors.append("steelblue")
        else:
            ys.append(0.0)
            colors.append("#4a4a4a")
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    ax.bar(xs, ys, color=colors, alpha=0.88)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction ≥ threshold")
    ax.set_title(title)
    ax.text(
        0.99,
        0.02,
        "Gray bars: insufficient data or sklearn unavailable",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#777",
    )
    fig.tight_layout()
    save_figure(fig, outfile, dpi=140)
    plt.close(fig)
