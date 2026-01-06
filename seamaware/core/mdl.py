"""
Minimum Description Length (MDL) computation with pluggable likelihoods.

The MDL principle states that the best model minimizes:
    L(model) + L(data | model)

where L denotes description length in bits.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional

import numpy as np


class LikelihoodType(Enum):
    """Supported likelihood models."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"  # For heavy-tailed / sparse residuals
    CAUCHY = "cauchy"    # For very heavy tails


@dataclass
class MDLResult:
    """Complete MDL computation result."""
    total_bits: float
    data_bits: float
    model_bits: float
    num_params: int
    num_samples: int
    bits_per_sample: float
    likelihood_type: str

    def __repr__(self) -> str:
        return (f"MDLResult(total={self.total_bits:.2f} bits, "
                f"data={self.data_bits:.2f}, model={self.model_bits:.2f}, "
                f"k={self.num_params}, n={self.num_samples})")


def _gaussian_nll_bits(residuals: np.ndarray) -> float:
    """Negative log-likelihood for Gaussian in bits."""
    n = len(residuals)
    variance = np.var(residuals)

    # Handle edge cases
    if variance < 1e-15:
        # Near-perfect fit: use minimum representable variance
        variance = 1e-15

    # NLL = (n/2) * log(2*pi*var) + (1/(2*var)) * sum(residuals^2)
    # In bits (divide by ln(2)):
    nll_nats = (n / 2) * np.log(2 * np.pi * variance) + np.sum(residuals**2) / (2 * variance)
    return nll_nats / np.log(2)


def _laplace_nll_bits(residuals: np.ndarray) -> float:
    """Negative log-likelihood for Laplace (double exponential) in bits."""
    n = len(residuals)
    # MLE for Laplace scale parameter
    b = np.mean(np.abs(residuals))
    if b < 1e-15:
        b = 1e-15

    # NLL = n * log(2b) + sum(|residuals|) / b
    nll_nats = n * np.log(2 * b) + np.sum(np.abs(residuals)) / b
    return nll_nats / np.log(2)


def _cauchy_nll_bits(residuals: np.ndarray) -> float:
    """Negative log-likelihood for Cauchy in bits (approximate MLE)."""
    n = len(residuals)
    # Use median absolute deviation as robust scale estimate
    gamma = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
    if gamma < 1e-15:
        gamma = 1e-15

    # NLL = n * log(pi * gamma) + sum(log(1 + (x/gamma)^2))
    nll_nats = n * np.log(np.pi * gamma) + np.sum(np.log(1 + (residuals / gamma)**2))
    return nll_nats / np.log(2)


_LIKELIHOOD_FUNCTIONS = {
    LikelihoodType.GAUSSIAN: _gaussian_nll_bits,
    LikelihoodType.LAPLACE: _laplace_nll_bits,
    LikelihoodType.CAUCHY: _cauchy_nll_bits,
}


def compute_mdl(
    data: np.ndarray,
    predictions: np.ndarray,
    num_params: int,
    likelihood: LikelihoodType = LikelihoodType.GAUSSIAN,
    param_precision_bits: float = 32.0,
) -> MDLResult:
    """
    Compute two-part MDL score.

    MDL = L(model) + L(data | model)

    Parameters
    ----------
    data : np.ndarray
        Original data (1D)
    predictions : np.ndarray
        Model predictions (same shape as data)
    num_params : int
        Number of model parameters
    likelihood : LikelihoodType
        Likelihood model for residuals
    param_precision_bits : float
        Bits per parameter (default 32 for float32)

    Returns
    -------
    MDLResult
        Complete MDL breakdown

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    # Validate inputs
    data = np.asarray(data, dtype=np.float64).ravel()
    predictions = np.asarray(predictions, dtype=np.float64).ravel()

    if len(data) == 0:
        raise ValueError("Empty data array")
    if len(data) != len(predictions):
        raise ValueError(f"Shape mismatch: data={len(data)}, predictions={len(predictions)}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains NaN or Inf")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain NaN or Inf")
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    n = len(data)
    residuals = data - predictions

    # Model cost: bits to encode parameters
    # Using refined MDL: k/2 * log2(n) + k * param_precision
    # The log2(n) term accounts for parameter precision scaling with sample size
    model_bits = (num_params / 2) * np.log2(n) if n > 1 else 0
    model_bits += num_params * param_precision_bits

    # Data cost: negative log-likelihood in bits
    nll_func = _LIKELIHOOD_FUNCTIONS[likelihood]
    data_bits = nll_func(residuals)

    total_bits = model_bits + data_bits

    return MDLResult(
        total_bits=total_bits,
        data_bits=data_bits,
        model_bits=model_bits,
        num_params=num_params,
        num_samples=n,
        bits_per_sample=total_bits / n,
        likelihood_type=likelihood.value
    )


def compute_k_star(likelihood: LikelihoodType = LikelihoodType.GAUSSIAN) -> float:
    """
    Compute the theoretical k* threshold for seam detection.

    For Gaussian: k* = 1 / (2 * ln(2)) ≈ 0.721

    This is the SNR threshold above which tracking orientation
    (1 bit per seam) reduces total MDL.

    Parameters
    ----------
    likelihood : LikelihoodType
        Likelihood model (affects threshold)

    Returns
    -------
    float
        k* threshold value
    """
    if likelihood == LikelihoodType.GAUSSIAN:
        return 1 / (2 * np.log(2))  # ≈ 0.7213
    elif likelihood == LikelihoodType.LAPLACE:
        # For Laplace, threshold is different (derived separately)
        return 1 / np.log(2)  # ≈ 1.4427
    elif likelihood == LikelihoodType.CAUCHY:
        # Cauchy has infinite variance, threshold is approximate
        return 2.0  # Empirical estimate
    else:
        return 1 / (2 * np.log(2))  # Default to Gaussian


def mdl_improvement(
    baseline_mdl: MDLResult,
    seam_mdl: MDLResult,
) -> dict:
    """
    Compute improvement metrics between baseline and seam-aware MDL.

    Returns
    -------
    dict with keys:
        - absolute_reduction: bits saved
        - relative_reduction: fraction reduction (0-1)
        - compression_ratio: baseline/seam ratio
        - effective: bool, whether seam model is better
    """
    abs_reduction = baseline_mdl.total_bits - seam_mdl.total_bits
    rel_reduction = abs_reduction / baseline_mdl.total_bits if baseline_mdl.total_bits > 0 else 0
    ratio = baseline_mdl.total_bits / seam_mdl.total_bits if seam_mdl.total_bits > 0 else float('inf')

    return {
        "absolute_reduction": abs_reduction,
        "relative_reduction": rel_reduction,
        "compression_ratio": ratio,
        "effective": abs_reduction > 0
    }


# Legacy compatibility functions
def compute_bic(
    data: np.ndarray, prediction: np.ndarray, num_params: int
) -> float:
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
        BIC and MDL differ in constant factors but have same asymptotic
        behavior.
    """
    n = len(data)
    residuals = data - prediction
    sigma2 = np.var(residuals) + 1e-10

    # BIC in bits
    bic = (n / 2) * np.log2(sigma2) + num_params * np.log2(n)

    return bic


def compute_aic(
    data: np.ndarray, prediction: np.ndarray, num_params: int
) -> float:
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


def delta_mdl(mdl_baseline: float, mdl_seam: float) -> float:
    """
    Compute MDL improvement (negative = better).

    Args:
        mdl_baseline: MDL without seam
        mdl_seam: MDL with seam

    Returns:
        ΔMDL = mdl_seam - mdl_baseline
        Accept seam if ΔMDL < 0
    """
    return mdl_seam - mdl_baseline
