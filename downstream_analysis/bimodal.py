"""Two-component mixture for moving vs stationary-ish chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None  # type: ignore[misc, assignment]


@dataclass
class BimodalFit:
    low_mean: float
    high_mean: float
    low_weight: float
    high_weight: float
    threshold: float  # midpoint between component means in sorted order
    labels: np.ndarray  # 0 = lower mean component, 1 = higher
    bic: float | None


def fit_bimodal_gmm(values: np.ndarray, *, random_state: int = 0) -> BimodalFit | None:
    """
    Fit 2-component 1D GMM on log(x + eps). Returns None if too few points or sklearn missing.
    """
    v = values[np.isfinite(values)]
    v = v[v >= 0]
    if v.size < 10:
        return None
    if GaussianMixture is None:
        return None

    eps = 1e-6
    x = np.log(v + eps).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gm.fit(x)
    means_log = gm.means_.ravel()
    order = np.argsort(means_log)
    low_i, high_i = int(order[0]), int(order[1])
    low_mean = float(np.exp(means_log[low_i]))
    high_mean = float(np.exp(means_log[high_i]))
    weights = gm.weights_.ravel()
    low_w, high_w = float(weights[low_i]), float(weights[high_i])
    resp = gm.predict_proba(x)
    labels = np.argmax(resp, axis=1).astype(np.int32)
    # remap so label 0 = low mean component
    raw_low = low_i
    labels01 = np.where(labels == raw_low, 0, 1)
    thr_log = (means_log[low_i] + means_log[high_i]) / 2.0
    threshold = float(np.exp(thr_log))
    bic = float(gm.bic(x))
    return BimodalFit(
        low_mean=low_mean,
        high_mean=high_mean,
        low_weight=low_w,
        high_weight=high_w,
        threshold=threshold,
        labels=labels01,
        bic=bic,
    )


def moving_fraction(fit: BimodalFit | None, values: np.ndarray) -> float | None:
    """Fraction of samples at or above mixture midpoint threshold (higher-locomotion side)."""
    if fit is None:
        return None
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    return float(np.mean(v >= fit.threshold))
