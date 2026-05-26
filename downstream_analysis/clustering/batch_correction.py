"""Batch effect correction for multi-project feature matrices.

Implements parametric ComBat (Johnson et al., 2007) in pure numpy/scipy,
plus a simpler z-score-per-batch fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import polygamma

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComBat (parametric empirical Bayes)
# ---------------------------------------------------------------------------

def _aprior(gamma_hat: np.ndarray) -> float:
    """Estimate hyperprior for gamma (location) via method of moments."""
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 + m ** 2) / s2 if s2 > 1e-12 else 1.0


def _bprior(gamma_hat: np.ndarray) -> float:
    """Estimate hyperprior for gamma (location) via method of moments."""
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m * s2 + m ** 3) / s2 if s2 > 1e-12 else m


def _postmean(g_hat: np.ndarray, g_bar: float, n: np.ndarray, d_star: np.ndarray, t2: float) -> np.ndarray:
    """Posterior mean of gamma (location parameter)."""
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum_sq: np.ndarray, n: np.ndarray, a_prior: float, b_prior: float) -> np.ndarray:
    """Posterior mean of delta (scale parameter) via inverse-gamma."""
    return (0.5 * sum_sq + b_prior) / (0.5 * n + a_prior - 1)


def _it_sol(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 0.0001,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative solution for empirical Bayes batch parameter estimation."""
    n = np.array([s_data.shape[0]] * s_data.shape[1], dtype=np.float64)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum_sq = np.sum((s_data - g_new[np.newaxis, :]) ** 2, axis=0)
        d_new = _postvar(sum_sq, n, a, b)
        change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
                      np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)))
        g_old, d_old = g_new, d_new
        if change < conv:
            break

    return g_new, d_new


def combat_correct(
    features: np.ndarray,
    batch_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply parametric ComBat batch correction.

    Parameters
    ----------
    features : (N, P) float64 feature matrix (NaN should be imputed beforehand)
    batch_labels : (N,) integer or string array identifying the batch for each sample

    Returns
    -------
    (corrected_features, info_dict) where corrected_features has same shape as input
    """
    X = features.copy().astype(np.float64)
    N, P = X.shape

    # Encode batch labels as integers
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([batch_map[b] for b in batch_labels])

    batch_sizes = np.array([np.sum(batch_idx == i) for i in range(n_batches)])
    logger.info(
        "ComBat: %d samples, %d features, %d batches (sizes: %s)",
        N, P, n_batches, ", ".join(f"{b}:{s}" for b, s in zip(unique_batches, batch_sizes)),
    )

    if n_batches < 2:
        logger.warning("Only 1 batch — no correction needed.")
        return X, {"n_batches": 1, "method": "none"}

    # Step 1: Standardize — compute grand mean and pooled variance
    grand_mean = X.mean(axis=0)
    grand_var = X.var(axis=0)
    grand_var[grand_var < 1e-12] = 1.0  # avoid division by zero

    # Create design matrix for batch means
    B_hat = np.zeros((n_batches, P))
    for i in range(n_batches):
        mask = batch_idx == i
        B_hat[i] = X[mask].mean(axis=0)

    # Pooled variance (within-batch variance, pooled across batches)
    var_pooled = np.zeros(P)
    for i in range(n_batches):
        mask = batch_idx == i
        if batch_sizes[i] > 1:
            var_pooled += (batch_sizes[i] - 1) * X[mask].var(axis=0, ddof=1)
    var_pooled /= max(N - n_batches, 1)
    var_pooled[var_pooled < 1e-12] = 1.0

    # Standardize data
    X_std = (X - grand_mean[np.newaxis, :]) / np.sqrt(var_pooled[np.newaxis, :])

    # Step 2: Estimate batch effects (gamma = location, delta = scale)
    gamma_hat = np.zeros((n_batches, P))
    delta_hat = np.zeros((n_batches, P))
    for i in range(n_batches):
        mask = batch_idx == i
        gamma_hat[i] = X_std[mask].mean(axis=0)
        delta_hat[i] = X_std[mask].var(axis=0, ddof=1) if batch_sizes[i] > 1 else np.ones(P)

    # Step 3: Empirical Bayes — estimate hyperparameters and shrink
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i in range(n_batches):
        mask = batch_idx == i
        s_data = X_std[mask]

        # Hyperparameters for gamma
        g_bar = gamma_hat[i].mean()
        t2 = gamma_hat[i].var()

        # Hyperparameters for delta (inverse-gamma)
        a_prior = _aprior(delta_hat[i])
        b_prior = _bprior(delta_hat[i])

        # Iterative EB estimation
        g_star, d_star = _it_sol(
            s_data, gamma_hat[i], delta_hat[i],
            g_bar, t2, a_prior, b_prior,
        )
        gamma_star[i] = g_star
        delta_star[i] = d_star

    # Step 4: Adjust data
    X_corrected = np.zeros_like(X_std)
    for i in range(n_batches):
        mask = batch_idx == i
        denom = np.sqrt(delta_star[i][np.newaxis, :])
        denom[denom < 1e-12] = 1.0
        X_corrected[mask] = (X_std[mask] - gamma_star[i][np.newaxis, :]) / denom

    # Transform back to original scale
    X_corrected = X_corrected * np.sqrt(var_pooled[np.newaxis, :]) + grand_mean[np.newaxis, :]

    info = {
        "method": "combat",
        "n_batches": n_batches,
        "batch_sizes": {str(b): int(s) for b, s in zip(unique_batches, batch_sizes)},
        "gamma_star_range": [float(gamma_star.min()), float(gamma_star.max())],
        "delta_star_range": [float(delta_star.min()), float(delta_star.max())],
    }
    logger.info(
        "ComBat done: gamma range [%.3f, %.3f], delta range [%.3f, %.3f]",
        gamma_star.min(), gamma_star.max(), delta_star.min(), delta_star.max(),
    )
    return X_corrected, info


# ---------------------------------------------------------------------------
# Simpler fallback: z-score per batch
# ---------------------------------------------------------------------------

def zscore_per_batch(
    features: np.ndarray,
    batch_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Z-score within each batch, then globally.

    Simpler than ComBat — useful when some batches are very small.
    """
    X = features.copy().astype(np.float64)
    unique_batches = np.unique(batch_labels)

    # Per-batch standardization
    for b in unique_batches:
        mask = batch_labels == b
        if mask.sum() < 2:
            continue
        mu = X[mask].mean(axis=0)
        sigma = X[mask].std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        X[mask] = (X[mask] - mu) / sigma

    # Global standardization
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    X = (X - mu) / sigma

    info = {
        "method": "zscore_per_batch",
        "n_batches": len(unique_batches),
    }
    return X, info
