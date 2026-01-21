"""
Seam detection using robust statistical methods.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class SeamDetectionResult:
    """Result of seam detection."""

    position: int
    confidence: float
    method: str
    all_candidates: List[Tuple[int, float]]  # (position, score) pairs


def detect_seam_cusum(
    signal: np.ndarray, threshold: Optional[float] = None, min_segment_length: int = 10
) -> SeamDetectionResult:
    """
    Detect seam using CUSUM (Cumulative Sum) change-point detection.

    More robust than argmax(diff) for:
    - Zero-crossings
    - Gradual transitions
    - Noisy signals

    Parameters
    ----------
    signal : np.ndarray
        1D input signal
    threshold : float, optional
        Detection threshold. If None, uses 3*std(signal)
    min_segment_length : int
        Minimum samples before/after seam

    Returns
    -------
    SeamDetectionResult
        Detection result with position, confidence, and candidates
    """
    # Validate min_segment_length
    if min_segment_length < 1:
        raise ValueError(f"min_segment_length must be >= 1, got {min_segment_length}")

    if len(signal) < 2 * min_segment_length:
        raise ValueError(f"Signal too short: {len(signal)} < {2 * min_segment_length}")

    n = len(signal)
    signal = np.asarray(signal, dtype=np.float64)

    # Validate input
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or Inf values")

    # Compute CUSUM statistic
    # Find maximum deviation from expected cumsum under no-change hypothesis
    # Use the CUSUM range statistic
    #
    # Optimization: Use cumulative sums to compute segment means in O(1) per split point
    # This makes the entire detection O(n) instead of O(n²)
    cumsum = np.cumsum(signal)
    total_sum = cumsum[-1]

    # OPTIMIZATION: Vectorized computation instead of loop
    valid_indices = np.arange(min_segment_length, n - min_segment_length)
    n_left = valid_indices
    n_right = n - valid_indices

    # Compute means using cumulative sum (O(1) per position)
    sum_left = cumsum[valid_indices - 1]  # Sum of signal[0:i]
    sum_right = total_sum - sum_left  # Sum of signal[i:]
    mean_left = sum_left / n_left
    mean_right = sum_right / n_right

    # Welch-Satterthwaite style statistic (vectorized)
    cusum_range = np.zeros(n)
    cusum_range[valid_indices] = np.abs(mean_right - mean_left) * np.sqrt(
        n_left * n_right / n
    )

    signal_std = np.std(signal)

    # Find candidates above threshold
    if threshold is None:
        threshold = signal_std * 1.5

    # OPTIMIZATION: Vectorized candidate finding
    mask = cusum_range[valid_indices] > threshold
    candidate_indices = valid_indices[mask]
    candidate_scores = cusum_range[candidate_indices]
    candidates = list(zip(candidate_indices.tolist(), candidate_scores.tolist()))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        # No significant change point found
        # Return the maximum anyway with low confidence
        valid_range = cusum_range[min_segment_length : n - min_segment_length]
        if len(valid_range) == 0:
            # Signal too short, return midpoint
            best_idx = n // 2
        else:
            best_idx = int(np.argmax(valid_range)) + min_segment_length
        return SeamDetectionResult(
            position=best_idx,
            confidence=0.0,
            method="cusum",
            all_candidates=[
                (
                    best_idx,
                    cusum_range[best_idx] if best_idx < len(cusum_range) else 0.0,
                )
            ],
        )

    best_pos, best_score = candidates[0]
    max_possible_score = signal_std * np.sqrt(n) / 2  # Approximate max
    confidence = (
        min(1.0, best_score / max_possible_score) if max_possible_score > 0 else 0.0
    )
    flat_signal_std_threshold = 0.05
    if signal_std < flat_signal_std_threshold:
        confidence *= signal_std / flat_signal_std_threshold

    return SeamDetectionResult(
        position=best_pos,
        confidence=confidence,
        method="cusum",
        all_candidates=candidates[:5],  # Top 5 candidates
    )


def detect_seam_roughness(
    signal: np.ndarray, window_size: int = 5, min_segment_length: int = 10
) -> SeamDetectionResult:
    """
    Detect seam using local roughness (rolling variance of differences).

    Better than raw diff for noisy signals.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal
    window_size : int
        Window for computing local roughness
    min_segment_length : int
        Minimum samples before/after seam

    Returns
    -------
    SeamDetectionResult
    """
    if len(signal) < 2 * min_segment_length:
        raise ValueError(f"Signal too short: {len(signal)} < {2 * min_segment_length}")

    n = len(signal)
    signal = np.asarray(signal, dtype=np.float64)

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or Inf values")

    # Compute first differences
    diff = np.diff(signal)

    # Compute rolling variance of differences (roughness)
    roughness = np.zeros(n - 1)
    half_win = window_size // 2

    for i in range(len(diff)):
        start = max(0, i - half_win)
        end = min(len(diff), i + half_win + 1)
        roughness[i] = np.var(diff[start:end]) if end > start else 0

    # Also look at absolute jump magnitude
    jump_magnitude = np.abs(diff)

    # Combined score: high jump AND change in local roughness
    combined_score = jump_magnitude * (1 + roughness / (np.mean(roughness) + 1e-10))

    # Find candidates
    candidates = []
    for i in range(min_segment_length, n - min_segment_length - 1):
        candidates.append((i, combined_score[i]))

    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        return SeamDetectionResult(
            position=n // 2, confidence=0.0, method="roughness", all_candidates=[]
        )

    best_pos, best_score = candidates[0]
    max_score = np.max(combined_score)
    confidence = best_score / max_score if max_score > 0 else 0.0

    return SeamDetectionResult(
        position=best_pos,
        confidence=confidence,
        method="roughness",
        all_candidates=candidates[:5],
    )


def detect_seam(
    signal: np.ndarray, method: str = "cusum", **kwargs: Any
) -> SeamDetectionResult:
    """
    Main seam detection entry point.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal
    method : str
        Detection method: "cusum", "roughness", or "ensemble"
    **kwargs
        Method-specific parameters

    Returns
    -------
    SeamDetectionResult
    """
    methods = {
        "cusum": detect_seam_cusum,
        "roughness": detect_seam_roughness,
    }

    if method == "ensemble":
        # Run both methods, take highest confidence
        results = [
            detect_seam_cusum(signal, **kwargs),
            detect_seam_roughness(signal, **kwargs),
        ]
        return max(results, key=lambda r: r.confidence)

    if method not in methods:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(methods.keys())}"
        )

    return methods[method](signal, **kwargs)
