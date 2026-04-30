"""Scalar summaries for locomotion arrays."""

from __future__ import annotations

import numpy as np


def distribution_snapshot(values: np.ndarray) -> dict[str, float | int]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }
