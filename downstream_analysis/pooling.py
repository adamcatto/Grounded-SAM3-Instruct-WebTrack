"""Pool samples across videos by object name; quantile and absolute views."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NamedSampleBundle:
    """All locomotion chunk values for one object name (across videos and instances)."""

    name: str
    absolute_pixels: list[float] = field(default_factory=list)
    normalized: list[float] = field(default_factory=list)
    quantile_ranks: list[float] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)


def ranks_to_uniform(values: np.ndarray) -> np.ndarray:
    """Empirical quantile rank in [0, 1], midrank method; ignores NaNs."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    srt = np.sort(v)
    out = np.empty_like(values, dtype=np.float64)
    out[:] = np.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            continue
        # fraction of samples <= x
        out[i] = float(np.searchsorted(srt, x, side="right") / max(srt.size, 1))
    return out


def append_video_quantiles(
    bundles: dict[str, NamedSampleBundle],
    object_name: str,
    video_id: str,
    abs_vals: np.ndarray,
    norm_vals: np.ndarray,
) -> None:
    """Add this video's samples; quantile ranks are within-video for this object stream."""
    m = np.isfinite(abs_vals) & np.isfinite(norm_vals)
    if not np.any(m):
        return
    a = abs_vals[m]
    n = norm_vals[m]
    q_abs = ranks_to_uniform(a)
    q_norm = ranks_to_uniform(n)
    # Pool quantiles: average of rank in absolute and normalized space for stability
    q = (q_abs + q_norm) / 2.0
    if object_name not in bundles:
        bundles[object_name] = NamedSampleBundle(name=object_name)
    b = bundles[object_name]
    for i in range(a.size):
        b.absolute_pixels.append(float(a[i]))
        b.normalized.append(float(n[i]))
        b.quantile_ranks.append(float(q[i]))
        b.video_ids.append(video_id)
