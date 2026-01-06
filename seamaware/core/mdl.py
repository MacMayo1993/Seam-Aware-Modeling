"""
Minimum Description Length (MDL) calculations for model selection.

This module implements Rissanen's MDL principle with proper bit counting,
forming the foundation for seam-aware model selection.

Reference:
    Rissanen, J. (1978). Modeling by shortest data description.
    Automatica, 14(5), 465-471.
"""

from typing import Union

import numpy as np


def compute_mdl(data: np.ndarray, prediction: np.ndarray, num_params: int) -> float:
    """
    Compute Minimum Description Length in bits.

    The MDL for a model M given data x is:
        MDL(x | M) = NLL(x | M) + (k/2)·log₂(N)

    where:
        - NLL = negative log-likelihood (assuming Gaussian noise)
        - k = number of model parameters
        - N = sample size

    Args:
        data: Observed signal (length N)
        prediction: Model prediction (same length as data)
        num_params: Number of model parameters k

    Returns:
        Total description length in bits

    Raises:
        ValueError: If data and prediction have different lengths
        ValueError: If data contains non-finite values

    Examples:
        >>> data = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> prediction = data + 0.1 * np.random.randn(100)
        >>> mdl = compute_mdl(data, prediction, num_params=2)
        >>> mdl > 0
        True
    """
    # Validate inputs
    if len(data) != len(prediction):
        raise ValueError(
            f"Data length ({len(data)}) != prediction length ({len(prediction)})"
        )

    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or inf values")

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or inf values")

    n = len(data)
    if n == 0:
        raise ValueError("Data is empty")

    # Compute residuals and variance
    residuals = data - prediction
    sigma2 = np.var(residuals) + 1e-10  # Regularization to avoid log(0)

    # Negative log-likelihood (Gaussian assumption)
    # For Gaussian: -log₂ p(x | μ, σ²) = (1/2)·log₂(2πeσ²) + (x-μ)²/(2σ²·ln(2))
    # Summing over all samples:
    nll = (n / 2) * np.log2(2 * np.pi * np.e * sigma2)

    # Parameter cost (Rissanen's normalized maximum likelihood)
    # The cost to encode k parameters with precision log₂(N)
    param_cost = (num_params / 2) * np.log2(n) if n > 1 else 0.0

    mdl = nll + param_cost

    return mdl


def delta_mdl(mdl_baseline: float, mdl_seam: float) -> float:
    """
    Compute MDL improvement (negative = better).

    Args:
        mdl_baseline: MDL without seam
        mdl_seam: MDL with seam

    Returns:
        ΔMDL = mdl_seam - mdl_baseline
        Accept seam if ΔMDL < 0

    Examples:
        >>> mdl_base = 1000.0
        >>> mdl_seam = 850.0
        >>> improvement = delta_mdl(mdl_base, mdl_seam)
        >>> improvement < 0  # Seam improves model
        True
    """
    return mdl_seam - mdl_baseline


def compute_bic(data: np.ndarray, prediction: np.ndarray, num_params: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC) for comparison.

    BIC is closely related to MDL but uses different parameter penalty:
        BIC = -2·ln(L) + k·ln(N)

    Converting to bits (divide by ln(2)):
        BIC_bits = N·log₂(σ²) + k·log₂(N)

    Args:
        data: Observed signal
        prediction: Model prediction
        num_params: Number of parameters

    Returns:
        BIC in bits (for consistency with MDL)

    Note:
        BIC and MDL differ in constant factors but have same asymptotic behavior.
    """
    n = len(data)
    residuals = data - prediction
    sigma2 = np.var(residuals) + 1e-10

    # BIC in bits
    bic = (n / 2) * np.log2(sigma2) + num_params * np.log2(n)

    return bic


def compute_aic(data: np.ndarray, prediction: np.ndarray, num_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC) for comparison.

    AIC = -2·ln(L) + 2k

    Converting to bits:
        AIC_bits = N·log₂(σ²) + 2k/ln(2)

    Args:
        data: Observed signal
        prediction: Model prediction
        num_params: Number of parameters

    Returns:
        AIC in bits

    Note:
        AIC penalizes parameters less than MDL/BIC, often overfitting.
    """
    n = len(data)
    residuals = data - prediction
    sigma2 = np.var(residuals) + 1e-10

    # AIC in bits
    aic = (n / 2) * np.log2(sigma2) + (2 * num_params) / np.log(2)

    return aic


def mdl_per_sample(data: np.ndarray, prediction: np.ndarray, num_params: int) -> float:
    """
    Compute MDL normalized by sample size (for cross-length comparison).

    Args:
        data: Observed signal
        prediction: Model prediction
        num_params: Number of parameters

    Returns:
        MDL / N (bits per sample)
    """
    mdl = compute_mdl(data, prediction, num_params)
    return mdl / len(data)


def residual_variance(data: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute residual variance σ².

    Args:
        data: Observed signal
        prediction: Model prediction

    Returns:
        Variance of residuals
    """
    residuals = data - prediction
    return float(np.var(residuals))


def effective_snr(
    data: np.ndarray, prediction_baseline: np.ndarray, prediction_seam: np.ndarray
) -> float:
    """
    Compute effective SNR from variance reduction.

    SNR_eff = (σ²_baseline - σ²_seam) / σ²_seam

    This is the signal-to-noise ratio that justifies the seam.

    Args:
        data: Observed signal
        prediction_baseline: Baseline prediction (no seam)
        prediction_seam: Seam-aware prediction

    Returns:
        Effective SNR (compare to k* ≈ 0.721)

    Examples:
        >>> # If SNR > k*, seam is justified
        >>> k_star = 1.0 / (2.0 * np.log(2))
        >>> snr = effective_snr(data, baseline_pred, seam_pred)
        >>> accept_seam = (snr > k_star)
    """
    sigma2_baseline = residual_variance(data, prediction_baseline)
    sigma2_seam = residual_variance(data, prediction_seam)

    if sigma2_seam == 0:
        return np.inf

    snr_eff = (sigma2_baseline - sigma2_seam) / sigma2_seam
    return max(0.0, snr_eff)  # SNR cannot be negative
