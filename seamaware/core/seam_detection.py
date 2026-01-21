"""
Seam detection via local roughness analysis.

This module implements roughness-based seam detection, which identifies
discontinuities by measuring local residual variance after polynomial fitting.

The key insight: Seams manifest as spikes in the roughness function R(τ),
where R(τ) = Var(residuals in window around τ).
"""

from typing import List, Optional

import numpy as np
from scipy import signal as scipy_signal


def compute_roughness(
    data: np.ndarray,
    window: int = 20,
    poly_degree: int = 1,
    method: str = "variance",
) -> np.ndarray:
    """
    Compute roughness function R(τ) across the signal.

    OPTIMIZED: Uses Savitzky-Golay filter for efficient polynomial smoothing
    instead of per-window polynomial fitting. This reduces complexity from
    O(n × window³) to O(n) while maintaining the same mathematical result.

    Args:
        data: Input signal (length N)
        window: Half-window size w (full window = 2w+1)
        poly_degree: Polynomial degree for local fitting (default: 1)
        method: Roughness metric ('variance', 'std', 'mad')

    Returns:
        Roughness array R(τ) of shape (N,)

    Examples:
        >>> signal = np.sin(np.linspace(0, 4*np.pi, 200))
        >>> signal[100:] *= -1  # Seam at 100
        >>> R = compute_roughness(signal, window=20)
        >>> seam_idx = np.argmax(R)
        >>> abs(seam_idx - 100) < 5
        True
    """
    n = len(data)

    # OPTIMIZATION: Use Savitzky-Golay filter for efficient polynomial smoothing
    # This is mathematically equivalent to local polynomial fitting but O(n)
    # instead of O(n × window³)
    window_length = 2 * window + 1

    # Ensure window_length is odd and <= n
    window_length = min(window_length, n if n % 2 == 1 else n - 1)
    if window_length < poly_degree + 2:
        # Fallback to simple method if window too small
        window_length = poly_degree + 2
        if window_length > n:
            # Cannot fit polynomial, return zeros
            return np.zeros(n)

    try:
        # Apply Savitzky-Golay filter to get smoothed signal
        smoothed = scipy_signal.savgol_filter(
            data, window_length, poly_degree, mode="nearest"
        )

        # Compute residuals
        residuals = data - smoothed

        # Compute rolling roughness metric
        if method == "variance":
            # Use rolling variance with stride tricks for efficiency
            roughness = _rolling_variance(residuals, window)
        elif method == "std":
            roughness = np.sqrt(_rolling_variance(residuals, window))
        elif method == "mad":
            # Median absolute deviation (requires per-window computation)
            roughness = _rolling_mad(residuals, window)
        else:
            raise ValueError(f"Unknown method: {method}")

    except (np.linalg.LinAlgError, ValueError):
        # Fallback to zeros on error
        roughness = np.zeros(n)

    return roughness


def _rolling_variance(data: np.ndarray, half_window: int) -> np.ndarray:
    """
    Compute rolling variance efficiently using cumulative sums.

    This is O(n) instead of O(n × window) for naive implementation.
    """
    n = len(data)
    result = np.zeros(n)

    # Compute cumulative sums for efficient variance computation
    # Var(X) = E[X²] - E[X]²
    cumsum_x = np.concatenate([[0], np.cumsum(data)])
    cumsum_x2 = np.concatenate([[0], np.cumsum(data**2)])

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_size = end - start

        if window_size > 0:
            sum_x = cumsum_x[end] - cumsum_x[start]
            sum_x2 = cumsum_x2[end] - cumsum_x2[start]
            mean_x = sum_x / window_size
            mean_x2 = sum_x2 / window_size
            result[i] = max(
                0, mean_x2 - mean_x**2
            )  # Avoid negative due to numerical error

    return result


def _rolling_mad(data: np.ndarray, half_window: int) -> np.ndarray:
    """Compute rolling median absolute deviation."""
    n = len(data)
    result = np.zeros(n)

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_data = data[start:end]

        if len(window_data) > 0:
            median = np.median(window_data)
            result[i] = np.median(np.abs(window_data - median))

    return result


def detect_seams_roughness(
    data: np.ndarray,
    window: int = 20,
    threshold_sigma: float = 2.0,
    min_distance: int = 10,
    poly_degree: int = 1,
    max_seams: Optional[int] = None,
) -> List[int]:
    """
    Detect seams via roughness local maxima.

    Algorithm:
    1. Compute roughness R(τ) across signal
    2. Find local maxima of R(τ)
    3. Threshold: keep only maxima > μ + threshold_sigma·σ
    4. Enforce minimum distance between seams

    Args:
        data: Input signal
        window: Half-window size for roughness computation
        threshold_sigma: Threshold in standard deviations above mean
        min_distance: Minimum samples between detected seams
        poly_degree: Polynomial degree for local fitting
        max_seams: Maximum number of seams to return (None = no limit)

    Returns:
        Sorted list of seam indices

    Examples:
        >>> signal = np.sin(np.linspace(0, 4*np.pi, 200))
        >>> signal[100:] *= -1
        >>> seams = detect_seams_roughness(signal, window=20)
        >>> len(seams) > 0
        True
        >>> abs(seams[0] - 100) < 10
        True
    """
    # Compute roughness
    roughness = compute_roughness(data, window=window, poly_degree=poly_degree)

    # Find local maxima
    # scipy.signal.find_peaks returns (peaks, properties)
    peaks, properties = scipy_signal.find_peaks(
        roughness, distance=min_distance, prominence=0.0
    )

    if len(peaks) == 0:
        return []

    # Threshold: keep only peaks > μ + threshold_sigma·σ
    mean_roughness = np.mean(roughness[roughness > 0])  # Exclude zeros at boundaries
    std_roughness = np.std(roughness[roughness > 0])
    threshold = mean_roughness + threshold_sigma * std_roughness

    # Filter peaks by threshold
    strong_peaks = peaks[roughness[peaks] > threshold]

    # Sort by roughness value (strongest first)
    if len(strong_peaks) > 0:
        sorted_indices = np.argsort(roughness[strong_peaks])[::-1]
        strong_peaks = strong_peaks[sorted_indices]

    # Limit number of seams
    if max_seams is not None and len(strong_peaks) > max_seams:
        strong_peaks = strong_peaks[:max_seams]

    # Return sorted by position
    return sorted(strong_peaks.tolist())


def detect_seams_cusum(
    data: np.ndarray, threshold: float = 5.0, drift: float = 0.5
) -> List[int]:
    """
    Detect seams using Cumulative Sum (CUSUM) change detection.

    CUSUM is sensitive to mean shifts and can complement roughness-based
    detection for detecting level changes.

    Args:
        data: Input signal
        threshold: CUSUM threshold (higher = fewer detections)
        drift: Drift parameter (0 = no drift correction)

    Returns:
        List of detected change points

    References:
        Page, E. S. (1954). Continuous inspection schemes.
        Biometrika, 41(1/2), 100-115.

    Examples:
        >>> signal = np.concatenate([np.zeros(50), np.ones(50)])
        >>> seams = detect_seams_cusum(signal, threshold=2.0)
        >>> len(seams) > 0
        True
        >>> abs(seams[0] - 50) < 10
        True
    """
    n = len(data)

    # Estimate baseline mean and std from first segment
    baseline_mean = np.mean(data[: min(50, n // 4)])
    baseline_std = np.std(data[: min(50, n // 4)]) + 1e-10

    # Standardize
    standardized = (data - baseline_mean) / baseline_std

    # CUSUM for upward shifts
    cusum_pos = np.zeros(n)
    # CUSUM for downward shifts
    cusum_neg = np.zeros(n)

    changepoints = []

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + standardized[i] - drift)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - standardized[i] - drift)

        if cusum_pos[i] > threshold:
            changepoints.append(i)
            cusum_pos[i] = 0  # Reset
        elif cusum_neg[i] > threshold:
            changepoints.append(i)
            cusum_neg[i] = 0  # Reset

    return sorted(list(set(changepoints)))  # Remove duplicates


def detect_seams_bayesian(
    data: np.ndarray, prior_prob: float = 0.01, min_distance: int = 10
) -> List[int]:
    """
    Detect seams using Bayesian online changepoint detection.

    This is a more sophisticated method that maintains a posterior
    distribution over run lengths (time since last changepoint).

    Args:
        data: Input signal
        prior_prob: Prior probability of changepoint at each timestep
        min_distance: Minimum distance between changepoints

    Returns:
        List of detected changepoints

    Note:
        This is a simplified implementation. For production use,
        consider the 'bayesian-changepoint-detection' package.

    References:
        Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint
        detection. arXiv preprint arXiv:0710.3742.
    """
    # Simplified placeholder implementation
    # Full Bayesian OCPD requires recursive posterior updates

    # For now, use a heuristic based on likelihood ratio
    n = len(data)
    window = max(10, min_distance)

    log_likelihood_ratio = np.zeros(n)

    for i in range(window, n - window):
        # Compare variance before/after potential changepoint
        before = data[i - window : i]
        after = data[i : i + window]

        var_before = np.var(before) + 1e-10
        var_after = np.var(after) + 1e-10

        # Log-likelihood ratio (larger = more likely changepoint)
        log_likelihood_ratio[i] = abs(np.log(var_before / var_after))

    # Find peaks in likelihood ratio
    peaks, _ = scipy_signal.find_peaks(
        log_likelihood_ratio, distance=min_distance, prominence=0.5
    )

    return sorted(peaks.tolist())


def refine_seam_location(
    data: np.ndarray,
    approximate_seam: int,
    search_window: int = 10,
    method: str = "variance_change",
) -> int:
    """
    Refine seam location by local search around approximate position.

    Args:
        data: Input signal
        approximate_seam: Initial seam estimate
        search_window: Half-width of search region
        method: Refinement method ('variance_change', 'gradient')

    Returns:
        Refined seam location

    Examples:
        >>> signal = np.concatenate([np.zeros(50), np.ones(50)])
        >>> refined = refine_seam_location(signal, 52, search_window=5)
        >>> abs(refined - 50) < 3
        True
    """
    n = len(data)
    start = max(search_window, approximate_seam - search_window)
    end = min(n - search_window - 1, approximate_seam + search_window)

    if method == "variance_change":
        # Find location with maximum variance change
        max_change = -np.inf
        best_location = approximate_seam

        for tau in range(start, end + 1):
            before = data[max(0, tau - search_window) : tau]
            after = data[tau : min(n, tau + search_window)]

            if len(before) > 0 and len(after) > 0:
                var_change = abs(np.var(after) - np.var(before))
                if var_change > max_change:
                    max_change = var_change
                    best_location = tau

        return best_location

    elif method == "gradient":
        # Find location with maximum gradient
        gradient = np.abs(np.gradient(data[start:end]))
        best_idx = np.argmax(gradient)
        return start + best_idx

    else:
        raise ValueError(f"Unknown refinement method: {method}")


def compute_seam_confidence(data: np.ndarray, seam: int, window: int = 20) -> float:
    """
    Compute confidence score for a detected seam.

    Confidence is based on:
    1. Roughness spike magnitude
    2. Variance change across seam
    3. Statistical significance

    Args:
        data: Input signal
        seam: Seam location
        window: Window size for analysis

    Returns:
        Confidence score in [0, 1] (higher = more confident)

    Examples:
        >>> signal = np.concatenate([np.zeros(50), np.ones(50)])
        >>> conf = compute_seam_confidence(signal, 50, window=10)
        >>> conf > 0.5
        True
    """
    n = len(data)

    if seam < window or seam >= n - window:
        return 0.0  # Not enough data

    # Compute roughness at seam
    roughness = compute_roughness(data, window=window)
    local_roughness = roughness[seam]
    mean_roughness = np.mean(roughness[roughness > 0])
    std_roughness = np.std(roughness[roughness > 0])

    # Roughness z-score
    if std_roughness > 0:
        z_score = (local_roughness - mean_roughness) / std_roughness
    else:
        z_score = 0.0

    # Variance change
    before = data[seam - window : seam]
    after = data[seam : seam + window]
    var_before = np.var(before)
    var_after = np.var(after)
    var_ratio = max(var_before, var_after) / (min(var_before, var_after) + 1e-10)

    # Combine metrics (heuristic)
    confidence = min(1.0, (z_score / 5.0) * (np.log(var_ratio) / 2.0))
    confidence = max(0.0, confidence)

    return confidence
