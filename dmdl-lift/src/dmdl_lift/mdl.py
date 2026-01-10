"""MDL computation for rank-1 vs rank-2 model comparison.

Rank-1: Global AR(1) model
Rank-2: Per-k-mer AR(1) models

MDL = NLL + (k/2) * log(n)
where k = number of parameters, n = number of observations.

Parameter counting convention:
- We use plug-in MLE for variance (absorbed into NLL)
- Only regression parameters (alpha, beta) are counted in penalty term
- This is consistent across both models for fair comparison

Note on fairness: Both rank-1 and rank-2 models are fit and scored
on the same temporal segment to ensure unbiased comparison.
"""
import numpy as np
from .phase import kmer_id


def fit_ar1(x):
    """Fit AR(1) model: x[t] = a*x[t-1] + c + eps.

    Parameters
    ----------
    x : np.ndarray
        Time series (length n)

    Returns
    -------
    theta : tuple of float
        (a, c) coefficients where:
        - a: autoregressive coefficient
        - c: intercept

    Notes
    -----
    Uses least squares: minimize sum((x[t] - (a*x[t-1] + c))^2)
    """
    if len(x) < 2:
        return (0.0, 0.0)
    X = np.column_stack([x[:-1], np.ones(len(x)-1)])
    y = x[1:]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return tuple(theta)


def nll_gaussian(residuals):
    """Negative log-likelihood for Gaussian residuals.

    Assumes unknown variance (MLE estimate via plug-in).

    Parameters
    ----------
    residuals : np.ndarray
        Residual errors

    Returns
    -------
    float
        Negative log-likelihood

    Notes
    -----
    For n residuals with MLE variance σ² = (1/n)Σe²:
    NLL = (n/2) * (log(2π) + log(σ²) + 1)
        = (n/2) * (log(2πσ²) + 1)
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    var = np.mean(residuals**2) + 1e-12
    return 0.5 * n * (np.log(2 * np.pi * var) + 1.0)


def mdl_rank1(x, start_idx=1):
    """Compute MDL for global AR(1) model.

    IMPORTANT: To ensure fair comparison with rank-2, we fit the model
    on the SAME segment that will be scored (from start_idx onward).

    Parameters
    ----------
    x : np.ndarray
        Time series
    start_idx : int
        First index to include in model fitting and scoring

    Returns
    -------
    mdl : float
        MDL score = NLL + (k/2)*log(n)
    params : tuple
        (a, c) AR(1) coefficients

    Notes
    -----
    Parameter count k=2: (alpha, beta)
    Variance is absorbed via MLE and not counted in penalty.
    """
    # Fit on the segment we'll score (fairness fix)
    x_segment = x[start_idx-1:]
    if len(x_segment) < 2:
        return float('inf'), (0.0, 0.0)

    a, c = fit_ar1(x_segment)

    # Compute residuals on the same segment
    pred = a * x_segment[:-1] + c
    resid = x_segment[1:] - pred

    n = len(resid)
    k = 2  # (a, c) - variance absorbed into NLL

    mdl = nll_gaussian(resid) + 0.5 * k * np.log(n)

    return mdl, (a, c)


def mdl_rank2(x, word, L=8, min_samples=10, a_global=None, c_global=None):
    """Compute MDL for per-k-mer AR(1) model.

    Fits separate AR(1) models for each k-mer phase that has sufficient
    samples. Falls back to global model for sparse phases.

    Parameters
    ----------
    x : np.ndarray
        Time series
    word : np.ndarray
        Symbolic sequence
    L : int
        K-mer length
    min_samples : int
        Minimum samples per phase to fit dedicated model
    a_global, c_global : float, optional
        Global AR(1) params for fallback (if None, computed here)

    Returns
    -------
    mdl : float
        MDL score = NLL + (k/2)*log(n)
    n_phases : int
        Number of phases with dedicated models
    ess_per_phase : float
        Average effective sample size per learned phase

    Notes
    -----
    Parameter count k = 2 * (number of learned phases)
    Each phase contributes 2 params: (alpha_phase, beta_phase)
    Variance absorbed via MLE, not counted.
    """
    start_idx = L - 1

    # Build k-mer phases for each time point
    phases = [None] * len(word)
    for t in range(L-1, len(word)):
        phases[t] = kmer_id(word, t, L=L)

    # Bucket observations by phase
    buckets = {}
    for t in range(start_idx, len(x)):
        ph = phases[t]
        if ph is None:
            continue
        if ph not in buckets:
            buckets[ph] = {"xs": [], "ys": []}
        buckets[ph]["xs"].append(x[t-1])
        buckets[ph]["ys"].append(x[t])

    # Fit per-phase AR(1) models
    thetas = {}
    for ph, d in buckets.items():
        xs = np.asarray(d["xs"])
        ys = np.asarray(d["ys"])
        if len(xs) < min_samples:
            continue
        X = np.column_stack([xs, np.ones(len(xs))])
        theta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        thetas[ph] = tuple(theta)

    # Global fallback (fitted on the segment from start_idx, for consistency)
    if a_global is None or c_global is None:
        x_segment = x[start_idx-1:]
        a_global, c_global = fit_ar1(x_segment)

    # Compute residuals using per-phase models where available
    residuals = []
    for t in range(start_idx, len(x)):
        ph = phases[t]
        if ph is not None and ph in thetas:
            a, c = thetas[ph]
        else:
            a, c = a_global, c_global
        pred = a * x[t-1] + c
        residuals.append(x[t] - pred)

    resid = np.asarray(residuals)
    n = len(resid)
    k = 2 * len(thetas)  # 2 params per phase (alpha, beta)

    mdl = nll_gaussian(resid) + 0.5 * k * np.log(n)

    # ESS per phase (only for learned phases)
    if thetas:
        counts = [len(buckets[ph]["xs"]) for ph in thetas]
        ess_per_phase = float(np.mean(counts))
    else:
        ess_per_phase = 0.0

    return mdl, len(thetas), ess_per_phase
