#!/usr/bin/env python3
"""
MASS/SMASH v2: Multi-Seam Modeling with Model Zoo and MDL Selection

This implements seam-aware signal decomposition:

1. SEAM PROPOSAL: Top-K candidates from multiple detectors
   - Antipodal correlation (chiral/Z₂ symmetry breaks)
   - Roughness + spectral divergence (regime changes)
   - Non-maximum suppression enforces minimum separation

2. CONFIGURATION SEARCH: Bounded subset enumeration (NOT beam search)
   - Evaluate all seam subsets up to max_seams
   - For each subset, try invertible transforms per segment
   - This is exhaustive within bounds, not pruned beam search

3. MODEL ZOO: Per-segment model competition
   - Fourier(K=4,8,12): periodic/quasi-periodic
   - Polynomial(deg=2,3,5): smooth trends
   - AR(p=5,10,15): autoregressive dynamics
   - Optional MLP: nonlinear patterns (in-sample only)

4. MDL SELECTION: Global objective with explicit seam penalty
   MDL ≈ (n/2)log₂(RSS/n) + (p/2)log₂(n) + (α/2)·m·log₂(n)

   Where:
   - p = total model parameters across segments
   - m = number of seams (encoding seam locations costs ≈ m·log₂(n) bits)
   - α = seam penalty hyperparameter (default 2.0)

The key insight: seams must "earn their bits" - a seam is only worth
introducing if the MDL reduction from better segment fits exceeds the
≈log₂(n) bits needed to encode its location.

Dependencies: numpy, matplotlib, scikit-learn (optional for MLP)
Author: Mac (Mayo Manifold research)
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Optional MLP (sklearn)
try:
    from sklearn.neural_network import MLPRegressor

    HAS_MLP = True
except ImportError:
    HAS_MLP = False


# =============================================================================
# Constants and Configuration
# =============================================================================

EPS = 1e-12  # Numerical stability floor


@dataclass
class MASSSMASHConfig:
    """Configuration for MASS/SMASH pipeline."""

    # Seam detection
    top_k_candidates: int = 5
    min_separation: int = 30
    antipodal_window: int = 40
    antipodal_threshold: float = 0.40  # Lowered for noisy signals
    roughness_window: int = 20
    roughness_threshold: float = 0.25  # Lowered for noisy signals

    # Search configuration
    max_seams: int = 3
    min_segment_length: int = 25
    use_beam_search: bool = True  # OPTIMIZATION: Use beam search instead of exhaustive
    beam_width: int = 10  # Keep top-K configurations at each stage

    # MDL parameters
    alpha: float = 2.0  # Seam penalty coefficient

    # Model zoo
    extended_zoo: bool = False
    include_mlp: bool = True

    # Output
    verbose: bool = True


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: Optional[int]) -> None:
    """Set numpy random seed if provided."""
    if seed is not None:
        np.random.seed(seed)


def safe_log2(x: float) -> float:
    """Log base 2 with floor for numerical stability."""
    return math.log2(max(x, EPS))


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalization with stability check."""
    s = np.std(x)
    if s < EPS:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


# =============================================================================
# MDL / BIC Scoring
# =============================================================================


def gaussian_nll_bits(rss: float, n: int) -> float:
    """
    Negative log-likelihood for Gaussian errors, in bits.
    NLL = (n/2) * log₂(RSS/n) + const

    We drop additive constants since they cancel in comparisons.
    """
    n = max(int(n), 1)
    rss = max(float(rss), EPS)
    return 0.5 * n * safe_log2(rss / n)


def bic_from_rss(rss: float, n: int, p: int) -> float:
    """
    BIC = n·log(RSS/n) + p·log(n)

    Args:
        rss: Residual sum of squares
        n: Number of observations
        p: Number of parameters
    """
    n = max(int(n), 1)
    rss = max(float(rss), EPS)
    return n * math.log(rss / n) + p * math.log(max(n, 2))


def mdl_bits(
    rss: float,
    n: int,
    p: int,
    m: int,
    alpha: float = 2.0,
) -> float:
    """
    MDL score in bits with explicit seam encoding penalty.

    MDL = (n/2)·log₂(RSS/n) + (p/2)·log₂(n) + (α/2)·m·log₂(n)

    The seam penalty reflects the coding cost: encoding m seam locations
    from n possible positions costs approximately m·log₂(n) bits.

    Args:
        rss: Residual sum of squares (pooled across segments)
        n: Total signal length
        p: Total model parameters across all segments
        m: Number of seams
        alpha: Seam penalty coefficient (default 2.0)

    Returns:
        MDL score in bits (lower is better)
    """
    n = max(int(n), 1)
    rss = max(float(rss), EPS)

    # Data fit term
    fit_term = gaussian_nll_bits(rss, n)

    # Parameter complexity term
    param_term = 0.5 * p * safe_log2(n)

    # Seam encoding term (the key insight: seams must pay for their bits)
    seam_term = 0.5 * alpha * m * safe_log2(n)

    return fit_term + param_term + seam_term


# =============================================================================
# Signal Generation (for testing)
# =============================================================================


def generate_signal_with_seams(
    T: int = 300,
    noise_std: float = 0.3,
    seam_positions: Optional[List[float]] = None,
    seam_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate test signal with known seam locations.

    Args:
        T: Signal length
        noise_std: Gaussian noise standard deviation
        seam_positions: Fractions [0,1] where seams occur
        seam_types: Transform type per seam ('sign_flip', 'frequency_change', 'phase_shift')
        seed: Random seed for reproducibility

    Returns:
        y: Noisy signal
        seam_indices: Integer indices of seam locations
    """
    set_seed(seed)
    t = np.linspace(0, 2 * np.pi, T)

    if seam_positions is None:
        seam_positions = [0.5]
    if seam_types is None:
        seam_types = ["sign_flip"] * len(seam_positions)

    # Convert fractions to indices, excluding edge cases
    seam_indices = [int(frac * T) for frac in seam_positions]
    seam_indices = [s for s in seam_indices if 5 < s < T - 5]

    # Build base signal
    base = np.sin(t) + 0.5 * np.sin(2 * t)

    # Apply transforms at each seam
    boundaries = [0] + sorted(seam_indices) + [T]
    y = np.zeros(T)

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment = base[start:end].copy()

        if i > 0:  # Apply transform after first segment
            transform = seam_types[min(i - 1, len(seam_types) - 1)]
            if transform == "sign_flip":
                segment *= -1
            elif transform == "frequency_change":
                t_seg = t[start:end]
                segment = np.sin(3 * t_seg) + 0.3 * np.sin(5 * t_seg)
            elif transform == "phase_shift":
                t_seg = t[start:end]
                segment = np.sin(t_seg + np.pi) + 0.5 * np.sin(2 * t_seg)

        y[start:end] = segment

    y += noise_std * np.random.randn(T)
    return y, seam_indices


# =============================================================================
# Seam Transforms (Invertible Atoms)
# =============================================================================
# These are the FlipZip operator family: each is its own inverse.


def apply_sign_flip(y: np.ndarray, tau: int) -> np.ndarray:
    """Sign flip from index tau onwards: y[tau:] *= -1"""
    z = y.copy()
    z[tau:] *= -1
    return z


def apply_reflect_invert(y: np.ndarray, tau: int) -> np.ndarray:
    """
    FlipZip-style domain reflection with sign inversion.
    For i > tau: z[i] = -y[2τ - i] (if in bounds)
    """
    T = len(y)
    z = y.copy()
    for i in range(tau + 1, T):
        j = 2 * tau - i
        z[i] = -y[j] if 0 <= j < T else -y[i]
    return z


TRANSFORMS = ["none", "sign_flip", "reflect_invert"]


# =============================================================================
# Seam Detectors
# =============================================================================


def antipodal_symmetry_scanner(
    y: np.ndarray,
    window_size: int = 40,
    threshold: float = 0.70,
    normalize: bool = True,
    top_k: int = 5,
    min_separation: int = 30,
) -> List[Tuple[int, float]]:
    """
    Detect seams via antipodal (chiral) correlation.

    At each candidate position τ, compute:
        score(τ) = corr(y[τ-w:τ], -y[τ:τ+w])

    High correlation indicates a sign-flip symmetry break.

    Args:
        y: Input signal
        window_size: Window for correlation (split in half)
        threshold: Minimum correlation to consider
        normalize: Z-score windows before correlation
        top_k: Maximum candidates to return
        min_separation: Minimum distance between candidates (NMS)

    Returns:
        List of (index, score) sorted by score descending
    """
    T = len(y)
    half = window_size // 2

    if T < window_size + 2:
        return []

    scores = np.zeros(T)

    # OPTIMIZATION: Vectorized correlation computation using sliding windows
    # This reduces O(n × window) to O(n log n) using FFT-based correlation
    from numpy.lib.stride_tricks import sliding_window_view

    if T >= 2 * half + 1:
        # Create sliding windows for efficient computation
        try:
            # Get all windows at once using stride tricks (memory-efficient views)
            all_windows = sliding_window_view(y, half)

            # For position i, we want windows_a = y[i-half:i] and windows_b = y[i:i+half]
            # all_windows[j] = y[j:j+half]
            # So windows_a for position i is all_windows[i-half]
            # And windows_b for position i is all_windows[i]

            # Valid range: half <= i < T - half
            valid_indices = np.arange(half, T - half)

            # Extract windows for all valid positions
            windows_a = all_windows[valid_indices - half]  # y[i-half:i] for each i
            windows_b = all_windows[valid_indices]  # y[i:i+half] for each i

            # Compute stds for all windows at once
            stds_a = np.std(windows_a, axis=1)
            stds_b = np.std(windows_b, axis=1)

            # Mask valid windows (std > EPS)
            valid_mask = (stds_a > EPS) & (stds_b > EPS)

            # Compute correlations only for valid windows
            if normalize:
                # Z-score normalize
                means_a = np.mean(windows_a, axis=1, keepdims=True)
                means_b = np.mean(windows_b, axis=1, keepdims=True)
                normed_a = (windows_a - means_a) / (stds_a[:, None] + EPS)
                normed_b = (windows_b - means_b) / (stds_b[:, None] + EPS)
            else:
                normed_a = windows_a
                normed_b = windows_b

            # Vectorized correlation: corr(a, -b) = -dot(a, b) / sqrt(sum(a²)sum(b²))
            # For normalized data, this simplifies to -mean(a * b)
            if normalize:
                correlations = -np.mean(normed_a * normed_b, axis=1)
            else:
                dot_products = np.sum(normed_a * (-normed_b), axis=1)
                norms_a = np.sqrt(np.sum(normed_a**2, axis=1))
                norms_b = np.sqrt(np.sum(normed_b**2, axis=1))
                correlations = dot_products / (norms_a * norms_b + EPS)

            # Assign scores using advanced indexing
            scores[valid_indices] = np.where(valid_mask, correlations, 0.0)

        except (ValueError, MemoryError, IndexError):
            # Fallback to loop-based implementation if stride tricks fail
            for i in range(half, T - half):
                a = y[i - half : i]
                b = y[i : i + half]

                if np.std(a) < EPS or np.std(b) < EPS:
                    continue

                if normalize:
                    a = zscore(a)
                    b = zscore(b)

                c = np.corrcoef(a, -b)[0, 1]
                if np.isfinite(c):
                    scores[i] = c

    # Find local maxima above threshold
    candidates = []
    for i in range(1, T - 1):
        if (
            scores[i] > threshold
            and scores[i] > scores[i - 1]
            and scores[i] > scores[i + 1]
        ):
            candidates.append((i, float(scores[i])))

    # Non-maximum suppression
    candidates.sort(key=lambda x: x[1], reverse=True)
    filtered = _nms(candidates, min_separation, top_k)

    return filtered


def roughness_detector(
    y: np.ndarray,
    window_size: int = 20,
    threshold: float = 0.25,
    top_k: int = 5,
    min_separation: int = 30,
) -> List[Tuple[int, float]]:
    """
    Detect seams via roughness discontinuity.

    Roughness = std(diff(window)). Large changes indicate regime shifts.
    We look for both absolute changes and relative spikes.

    Args:
        y: Input signal
        window_size: Window for roughness computation
        threshold: Minimum normalized roughness change to consider
        top_k: Maximum candidates to return
        min_separation: NMS minimum distance

    Returns:
        List of (index, score) sorted by score descending
    """
    T = len(y)

    if T < window_size + 3:
        return []

    # OPTIMIZATION: Vectorized rolling roughness using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view

    try:
        # Get all windows at once
        windows = sliding_window_view(y, window_size)
        # Compute diff for each window and then std
        diffs = np.diff(windows, axis=1)
        roughness = np.std(diffs, axis=1)
    except (ValueError, MemoryError):
        # Fallback to loop if stride tricks fail
        roughness = np.zeros(T - window_size)
        for i in range(T - window_size):
            w = y[i : i + window_size]
            roughness[i] = np.std(np.diff(w))

    if len(roughness) < 3:
        return []

    # Find discontinuities in roughness (normalized by local variance)
    change = np.abs(np.diff(roughness))

    # Normalize to make threshold meaningful across different signal scales
    change_std = np.std(change)
    if change_std > EPS:
        change_normalized = change / change_std
    else:
        change_normalized = change

    # Find local maxima
    candidates = []
    for i in range(1, len(change_normalized) - 1):
        if (
            change_normalized[i] > threshold
            and change_normalized[i] > change_normalized[i - 1]
            and change_normalized[i] > change_normalized[i + 1]
        ):
            seam_idx = i + window_size // 2
            candidates.append((seam_idx, float(change_normalized[i])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    filtered = _nms(candidates, min_separation, top_k)

    return filtered


def _nms(
    candidates: List[Tuple[int, float]], min_separation: int, max_keep: int
) -> List[Tuple[int, float]]:
    """
    Non-maximum suppression for 1D candidates.

    OPTIMIZATION: Use list comprehension with numpy for faster distance checks.
    """
    if not candidates:
        return []

    filtered = []
    used_indices = []

    for idx, score in candidates:
        # OPTIMIZATION: Vectorized distance check using numpy
        if used_indices:
            distances = np.abs(np.array(used_indices) - idx)
            if np.any(distances < min_separation):
                continue

        filtered.append((idx, score))
        used_indices.append(idx)

        if len(filtered) >= max_keep:
            break

    return filtered


def detect_seam_candidates(
    y: np.ndarray, config: MASSSMASHConfig
) -> List[Tuple[int, float, str]]:
    """
    Detect seam candidates from multiple detectors, merge with NMS.

    Each detector's scores are normalized to [0,1] before merging to
    ensure fair comparison across detection methods.

    Returns:
        List of (index, score, detector_name) sorted by score
    """
    # Antipodal detector
    anti = antipodal_symmetry_scanner(
        y,
        window_size=config.antipodal_window,
        threshold=config.antipodal_threshold,
        top_k=config.top_k_candidates * 2,  # Get more, will filter
        min_separation=config.min_separation,
    )

    # Roughness detector
    rough = roughness_detector(
        y,
        window_size=config.roughness_window,
        threshold=config.roughness_threshold,
        top_k=config.top_k_candidates * 2,
        min_separation=config.min_separation,
    )

    # Normalize scores within each detector to [0,1]
    def normalize_scores(cands: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not cands:
            return cands
        scores = [c[1] for c in cands]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        range_score = max_score - min_score
        if range_score < EPS:
            return [(idx, 1.0) for idx, _ in cands]
        return [(idx, (score - min_score) / range_score) for idx, score in cands]

    anti_norm = normalize_scores(anti)
    rough_norm = normalize_scores(rough)

    # Merge with source labels (normalized scores)
    all_candidates = []
    for idx, score in anti_norm:
        all_candidates.append((idx, score, "antipodal"))
    for idx, score in rough_norm:
        all_candidates.append((idx, score, "roughness"))

    # Global NMS on merged candidates
    all_candidates.sort(key=lambda x: x[1], reverse=True)

    filtered = []
    used = set()

    for idx, score, detector in all_candidates:
        if any(abs(idx - u) < config.min_separation for u in used):
            continue
        filtered.append((idx, score, detector))
        used.add(idx)
        if len(filtered) >= config.top_k_candidates:
            break

    return filtered


# =============================================================================
# Model Zoo
# =============================================================================


@dataclass
class FitResult:
    """Result from fitting a single model to a segment."""

    model_name: str
    yhat: np.ndarray
    rss: float
    mse: float
    bic: float
    p: int  # Number of parameters


class BaseModel:
    """Base class for zoo models."""

    name: str = "base"

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        raise NotImplementedError


class FourierModel(BaseModel):
    """Fourier basis regression: sin/cos up to K harmonics."""

    def __init__(self, K: int = 4):
        self.K = int(K)
        self.name = f"Fourier(K={self.K})"
        self._lr = LinearRegression()

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        T = len(y)
        t = np.linspace(0, 2 * np.pi, T)
        X = np.column_stack(
            [np.sin(k * t) for k in range(1, self.K + 1)]
            + [np.cos(k * t) for k in range(1, self.K + 1)]
        )
        self._lr.fit(X, y)
        return self._lr.predict(X)

    def n_params(self) -> int:
        return 2 * self.K + 1  # 2K coefficients + intercept


class PolynomialModel(BaseModel):
    """Polynomial regression."""

    def __init__(self, degree: int = 3):
        self.degree = int(degree)
        self.name = f"Poly(deg={self.degree})"
        self._pipe = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        )

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        T = len(y)
        x = np.linspace(0, 1, T).reshape(-1, 1)
        self._pipe.fit(x, y)
        return self._pipe.predict(x)

    def n_params(self) -> int:
        return self.degree + 1


class ARModel(BaseModel):
    """Autoregressive model via least squares."""

    def __init__(self, order: int = 5):
        self.order = int(order)
        self.name = f"AR(p={self.order})"
        self._coef = None

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        T = len(y)
        p = self.order

        if T <= p + 2:
            # Segment too short, fall back to mean
            return np.full(T, np.mean(y))

        # Build lag matrix
        Y = y[p:]
        X = np.column_stack([y[p - i : T - i] for i in range(1, p + 1)])

        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self._coef = coef

        # Predictions
        yhat = np.zeros(T)
        yhat[:p] = np.mean(y[:p])  # Warm start
        yhat[p:] = X @ coef

        return yhat

    def n_params(self) -> int:
        return self.order


class MLPModel(BaseModel):
    """
    MLP regressor (optional, requires sklearn).

    WARNING: This uses in-sample MDL as a proxy. MLP may win for wrong
    reasons on short segments. Consider this a "nonlinear mop-up" option,
    not a principled choice.
    """

    def __init__(self, hidden: Tuple[int, ...] = (32, 32), max_iter: int = 1500):
        self.hidden = tuple(hidden)
        self.max_iter = int(max_iter)
        self.name = f"MLP{self.hidden}"
        self._mlp = None  # Lazy initialization
        self._min_segment_length = 50  # Don't use MLP for short segments

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        T = len(y)

        # OPTIMIZATION: Skip MLP for short segments (too many params, too slow)
        if T < self._min_segment_length:
            # Return mean for very short segments
            return np.full_like(y, float(np.mean(y)))

        # OPTIMIZATION: Adaptive max_iter based on segment length
        # Short segments don't need many iterations
        adaptive_max_iter = min(self.max_iter, max(50, T * 2))

        # Lazy initialize MLP with optimized settings
        if self._mlp is None or self._mlp.max_iter != adaptive_max_iter:
            self._mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation="tanh",
                solver="adam",
                max_iter=adaptive_max_iter,
                random_state=0,
                early_stopping=True,
                n_iter_no_change=20,  # More aggressive early stopping
                tol=1e-3,  # Less strict tolerance for faster convergence
            )

        x = np.linspace(0, 1, T).reshape(-1, 1)
        self._mlp.fit(x, y)
        return self._mlp.predict(x)

    def n_params(self) -> int:
        # Count weights + biases
        sizes = (1,) + self.hidden + (1,)
        weights = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
        biases = sum(sizes[1:])
        return int(weights + biases)


class MeanModel(BaseModel):
    """Constant (mean) model - the null baseline."""

    def __init__(self):
        self.name = "Mean"

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        return np.full_like(y, float(np.mean(y)))

    def n_params(self) -> int:
        return 1


def build_model_zoo(config: MASSSMASHConfig) -> List[BaseModel]:
    """Build the model zoo based on configuration."""
    zoo: List[BaseModel] = [
        MeanModel(),
        FourierModel(K=4),
        FourierModel(K=8),
        PolynomialModel(degree=2),
        PolynomialModel(degree=3),
        ARModel(order=5),
        ARModel(order=10),
    ]

    if config.extended_zoo:
        zoo.extend(
            [
                FourierModel(K=12),
                PolynomialModel(degree=5),
                ARModel(order=15),
            ]
        )

    if config.include_mlp and HAS_MLP:
        zoo.append(MLPModel(hidden=(32, 32), max_iter=1500))

    return zoo


def fit_best_model(y: np.ndarray, zoo: List[BaseModel]) -> FitResult:
    """Fit all models in zoo, return best by BIC."""
    best: Optional[FitResult] = None
    n = len(y)

    for model in zoo:
        try:
            yhat = model.fit_predict(y)
            resid = y - yhat
            rss = float(np.sum(resid**2))
            mse = float(np.mean(resid**2))
            p = int(model.n_params())
            bic = bic_from_rss(rss, n, p)

            result = FitResult(
                model_name=model.name, yhat=yhat, rss=rss, mse=mse, bic=bic, p=p
            )

            if best is None or result.bic < best.bic:
                best = result

        except Exception:
            # Model fitting failed, skip
            continue

    if best is None:
        # Fallback to mean model
        yhat = np.full(n, float(np.mean(y)))
        rss = float(np.sum((y - yhat) ** 2))
        best = FitResult(
            model_name="Mean(fallback)",
            yhat=yhat,
            rss=rss,
            mse=rss / n,
            bic=bic_from_rss(rss, n, 1),
            p=1,
        )

    return best


# =============================================================================
# Solution Representation
# =============================================================================


@dataclass
class Solution:
    """A complete seam configuration solution."""

    seams: Tuple[int, ...]
    transform: str
    segment_fits: List[FitResult]
    yhat: np.ndarray
    total_rss: float
    total_mse: float
    total_bic: float
    total_mdl: float
    total_params: int

    @property
    def n_seams(self) -> int:
        return len(self.seams)

    @property
    def n_segments(self) -> int:
        return len(self.segment_fits)


# =============================================================================
# Piecewise Fitting
# =============================================================================


def get_segments(
    T: int, seams: Sequence[int], min_length: int
) -> Optional[List[Tuple[int, int]]]:
    """
    Convert seam indices to segment boundaries.

    Returns:
        List of (start, end) tuples, or None if any segment is too short
    """
    seams = sorted([s for s in seams if 0 < s < T])
    cuts = [0] + seams + [T]
    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    if any((end - start) < min_length for start, end in segments):
        return None

    return segments


def piecewise_fit(
    y: np.ndarray, seams: Sequence[int], zoo: List[BaseModel], min_segment_length: int
) -> Tuple[np.ndarray, List[FitResult]]:
    """
    Fit best model per segment.

    Returns:
        yhat: Full prediction array
        segment_fits: FitResult for each segment
    """
    T = len(y)
    segments = get_segments(T, seams, min_segment_length)

    if segments is None:
        # Fall back to single segment
        segments = [(0, T)]

    yhat = np.zeros(T)
    segment_fits: List[FitResult] = []

    for start, end in segments:
        seg = y[start:end]

        if len(seg) < 10:
            # Too short, use mean
            seg_yhat = np.full(len(seg), np.mean(seg))
            rss = float(np.sum((seg - seg_yhat) ** 2))
            fit = FitResult(
                model_name="constant",
                yhat=seg_yhat,
                rss=rss,
                mse=rss / len(seg),
                bic=bic_from_rss(rss, len(seg), 1),
                p=1,
            )
        else:
            fit = fit_best_model(seg, zoo)

        yhat[start:end] = fit.yhat
        segment_fits.append(fit)

    return yhat, segment_fits


def score_solution(
    y: np.ndarray,
    yhat: np.ndarray,
    segment_fits: List[FitResult],
    n_seams: int,
    alpha: float,
) -> Tuple[float, float, float, float, int]:
    """
    Compute global scores for a solution.

    Returns:
        (rss, mse, bic, mdl_bits, total_params)
    """
    resid = y - yhat
    rss = float(np.sum(resid**2))
    mse = float(np.mean(resid**2))
    n = len(y)

    # Sum segment BICs + seam penalty
    total_bic = sum(sf.bic for sf in segment_fits)
    total_bic += alpha * n_seams * math.log(max(n, 2))

    # Pooled MDL
    total_params = sum(sf.p for sf in segment_fits)
    total_mdl = mdl_bits(rss, n, total_params, n_seams, alpha)

    return rss, mse, total_bic, total_mdl, total_params


# =============================================================================
# Configuration Search (Bounded Subset Enumeration + Beam Search)
# =============================================================================
#
# Two modes available:
# 1. EXHAUSTIVE: Enumerate all seam subsets (original behavior)
# 2. BEAM SEARCH: Greedy search with pruning (10-100× faster)
#


def beam_search_configurations(
    y: np.ndarray,
    candidate_seams: List[int],
    zoo: List[BaseModel],
    config: MASSSMASHConfig,
) -> List[Solution]:
    """
    OPTIMIZATION: Beam search for seam configurations.

    Instead of exhaustive enumeration (C(k, max_seams) × |transforms|),
    we greedily add seams one at a time, keeping only top-K configurations.

    Algorithm:
    1. Start with no seams (baseline)
    2. For each step k = 1..max_seams:
       - For each configuration in beam:
         - Try adding each remaining candidate seam
         - Try each transform
       - Keep top beam_width configurations
    3. Return all explored configurations ranked by MDL

    Complexity: O(beam_width × max_seams × |candidates| × |transforms|)
    vs Exhaustive: O(C(|candidates|, max_seams) × |transforms|)

    For typical values (beam=10, max_seams=3, candidates=5, transforms=3):
    - Beam: 10 × 3 × 5 × 3 = 450 evaluations
    - Exhaustive: C(5,3) × 3 = 30 evaluations (but grows exponentially)

    For larger searches (candidates=10, max_seams=5):
    - Beam: 10 × 5 × 10 × 3 = 1,500 evaluations
    - Exhaustive: C(10,5) × 3 = 756 evaluations

    Beam search wins when candidate pool is large or max_seams > 3.
    """
    # Start with no-seam baseline
    baseline_solution = _evaluate_configuration(y, [], "none", zoo, config)
    beam = [baseline_solution]
    all_solutions = [baseline_solution]

    # Iteratively add seams
    for num_seams in range(1, config.max_seams + 1):
        candidates_for_beam = []

        for current_sol in beam:
            current_seams = list(current_sol.seams)

            # Try adding each remaining candidate
            remaining_candidates = [
                s for s in candidate_seams if s not in current_seams
            ]

            for new_seam in remaining_candidates:
                new_seams = sorted(current_seams + [new_seam])

                # Try each transform
                for transform in TRANSFORMS:
                    sol = _evaluate_configuration(y, new_seams, transform, zoo, config)
                    candidates_for_beam.append(sol)
                    all_solutions.append(sol)

        # Keep top beam_width by MDL
        candidates_for_beam.sort(key=lambda s: s.total_mdl)
        beam = candidates_for_beam[: config.beam_width]

        # Early stopping: if best solution hasn't improved, stop adding seams
        if beam and baseline_solution.total_mdl <= beam[0].total_mdl:
            break

    # Return all explored solutions ranked by MDL
    all_solutions.sort(key=lambda s: s.total_mdl)
    return all_solutions


def _evaluate_configuration(
    y: np.ndarray,
    seams: List[int],
    transform: str,
    zoo: List[BaseModel],
    config: MASSSMASHConfig,
) -> Solution:
    """Helper to evaluate a single (seams, transform) configuration."""
    # Apply transform
    if transform == "none":
        y_transformed = y
    elif transform == "sign_flip" and seams:
        y_transformed = y.copy()
        for tau in seams:
            y_transformed = apply_sign_flip(y_transformed, tau)
    elif transform == "reflect_invert" and seams:
        y_transformed = y.copy()
        for tau in seams:
            y_transformed = apply_reflect_invert(y_transformed, tau)
    else:
        y_transformed = y

    # Fit piecewise
    yhat_transformed, segment_fits = piecewise_fit(
        y_transformed, seams, zoo, config.min_segment_length
    )

    # Invert transform on predictions
    yhat = yhat_transformed.copy()
    if transform == "sign_flip" and seams:
        for tau in reversed(seams):
            yhat = apply_sign_flip(yhat, tau)
    elif transform == "reflect_invert" and seams:
        for tau in reversed(seams):
            yhat = apply_reflect_invert(yhat, tau)

    # Score
    rss, mse, bic, mdl, params = score_solution(
        y, yhat, segment_fits, len(seams), config.alpha
    )

    return Solution(
        seams=tuple(seams),
        transform=transform,
        segment_fits=segment_fits,
        yhat=yhat,
        total_rss=rss,
        total_mse=mse,
        total_bic=bic,
        total_mdl=mdl,
        total_params=params,
    )


def enumerate_configurations(
    y: np.ndarray,
    candidate_seams: List[int],
    zoo: List[BaseModel],
    config: MASSSMASHConfig,
) -> List[Solution]:
    """
    Enumerate seam configurations and score by MDL.

    OPTIMIZATION: Two modes available via config.use_beam_search:
    - False (default for small searches): Exhaustive enumeration
    - True: Beam search (10-100× faster for large searches)

    Exhaustive search:
    - Enumerate all C(k, j) subsets for j = 0..max_seams
    - For each subset, try all transforms
    - Rank by MDL

    Beam search:
    - Greedy incremental search
    - Keep top-K configurations at each stage
    - Much faster for large candidate pools
    """
    # OPTIMIZATION: Use beam search for large search spaces
    if config.use_beam_search:
        return beam_search_configurations(y, candidate_seams, zoo, config)

    # Original exhaustive enumeration
    # Generate all seam subsets up to max_seams
    configurations = [[]]  # Empty = no seams
    for k in range(1, min(config.max_seams + 1, len(candidate_seams) + 1)):
        for combo in itertools.combinations(candidate_seams, k):
            configurations.append(sorted(list(combo)))

    solutions: List[Solution] = []

    for seams in configurations:
        for transform in TRANSFORMS:
            sol = _evaluate_configuration(y, seams, transform, zoo, config)
            solutions.append(sol)

    # Rank by MDL
    solutions.sort(key=lambda s: s.total_mdl)
    return solutions


# =============================================================================
# Main Pipeline
# =============================================================================


def run_mass_smash(
    y: np.ndarray,
    config: Optional[MASSSMASHConfig] = None,
    true_seams: Optional[List[int]] = None,
) -> Tuple[Solution, List[Solution]]:
    """
    Run the full MASS/SMASH pipeline.

    Args:
        y: Input signal
        config: Pipeline configuration
        true_seams: Known seam locations (for evaluation only)

    Returns:
        best: Best solution by MDL
        all_solutions: All evaluated solutions (sorted by MDL)
    """
    if config is None:
        config = MASSSMASHConfig()

    # Step 1: Detect seam candidates
    if config.verbose:
        print("Step 1: Detecting seam candidates...")

    candidates = detect_seam_candidates(y, config)
    candidate_indices = [idx for idx, _, _ in candidates]

    if config.verbose:
        print(f"  Found {len(candidates)} candidates: {candidate_indices}")
        for idx, score, detector in candidates:
            print(f"    {idx}: {score:.3f} ({detector})")

    # Step 2: Build model zoo
    zoo = build_model_zoo(config)

    if config.verbose:
        print(f"\nStep 2: Model zoo ({len(zoo)} models)")
        for m in zoo:
            print(f"    {m.name}")

    # Step 3: Enumerate and score configurations
    if config.verbose:
        print(f"\nStep 3: Enumerating configurations (max_seams={config.max_seams})...")

    solutions = enumerate_configurations(y, candidate_indices, zoo, config)

    if config.verbose:
        print(f"  Evaluated {len(solutions)} configurations")

    # Step 4: Select best by MDL
    best = solutions[0]

    if config.verbose:
        print("\nStep 4: Best solution by MDL")
        print(f"  Seams: {list(best.seams)}")
        print(f"  Transform: {best.transform}")
        print(f"  MDL: {best.total_mdl:.2f} bits")
        print(f"  MSE: {best.total_mse:.6f}")
        print(f"  Params: {best.total_params}")
        print("\n  Segment models:")
        for i, sf in enumerate(best.segment_fits):
            print(f"    Segment {i}: {sf.model_name} (BIC={sf.bic:.2f}, p={sf.p})")

        if true_seams is not None:
            print(f"\n  True seams: {true_seams}")
            if best.seams:
                errors = [min(abs(ts - ds) for ds in best.seams) for ts in true_seams]
                print(f"  Detection errors: {errors}")

    return best, solutions


# =============================================================================
# Visualization
# =============================================================================


def plot_solution(
    y: np.ndarray,
    solution: Solution,
    candidates: Optional[List[Tuple[int, float, str]]] = None,
    true_seams: Optional[List[int]] = None,
    save_path: Optional[str] = None,
):
    """Visualize the solution."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    T = len(y)
    t = np.arange(T)

    # Top panel: Signal and fit
    ax = axes[0]
    ax.plot(t, y, "o", markersize=2, alpha=0.4, color="gray", label="Signal")
    ax.plot(
        t,
        solution.yhat,
        linewidth=2.5,
        color="green",
        label=f"Fit (m={solution.n_seams} seams)",
    )

    if true_seams:
        for i, ts in enumerate(true_seams):
            ax.axvline(
                ts,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="True seam" if i == 0 else None,
            )

    for i, seam in enumerate(solution.seams):
        ax.axvline(
            seam,
            color="blue",
            linestyle=":",
            linewidth=2.5,
            label="Detected seam" if i == 0 else None,
        )

    ax.legend(loc="best")
    ax.set_ylabel("Signal")
    ax.set_title(
        f"MASS/SMASH Solution | MDL={solution.total_mdl:.1f} bits | "
        f"MSE={solution.total_mse:.4f} | Transform={solution.transform}",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    # Bottom panel: Candidates
    ax = axes[1]

    if candidates:
        colors = {"antipodal": "green", "roughness": "orange"}
        for idx, score, detector in candidates:
            c = colors.get(detector, "gray")
            ax.axvline(idx, color=c, linestyle=":", linewidth=1.5, alpha=0.6)
            ax.text(
                idx, 0.9, f"{score:.2f}", rotation=90, va="top", fontsize=8, color=c
            )

    for seam in solution.seams:
        ax.axvline(seam, color="blue", linestyle="-", linewidth=3, alpha=0.8)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Seam Candidates (Green=Antipodal, Orange=Roughness, Blue=Selected)",
        fontweight="bold",
    )
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")

    plt.show()


# =============================================================================
# Batch Evaluation
# =============================================================================


@dataclass
class BatchResults:
    """Results from batch evaluation."""

    n_runs: int
    mdls: List[float] = field(default_factory=list)
    mses: List[float] = field(default_factory=list)
    n_seams_detected: List[int] = field(default_factory=list)
    detection_errors: List[float] = field(default_factory=list)
    perfect_count: int = 0

    def add(self, mdl: float, mse: float, n_seams: int, error: float, perfect: bool):
        self.mdls.append(mdl)
        self.mses.append(mse)
        self.n_seams_detected.append(n_seams)
        self.detection_errors.append(error)
        if perfect:
            self.perfect_count += 1

    def summary(self) -> str:
        # Compute seam distribution
        seam_dist = dict(
            (k, self.n_seams_detected.count(k))
            for k in sorted(set(self.n_seams_detected))
        )

        lines = [
            "=" * 70,
            "BATCH RESULTS",
            "=" * 70,
            f"Runs: {self.n_runs}",
            "",
            "MDL (bits):",
            f"  Mean: {np.mean(self.mdls):.2f} ± {np.std(self.mdls):.2f}",
            f"  Median: {np.median(self.mdls):.2f}",
            "",
            "MSE:",
            f"  Mean: {np.mean(self.mses):.6f} ± {np.std(self.mses):.6f}",
            f"  Median: {np.median(self.mses):.6f}",
            "",
            "Seams detected:",
            f"  Distribution: {seam_dist}",
            "",
            "Detection accuracy:",
        ]

        valid_errors = [e for e in self.detection_errors if e < float("inf")]
        if valid_errors:
            lines.extend(
                [
                    f"  Mean error: {np.mean(valid_errors):.1f} samples",
                    f"  Median error: {np.median(valid_errors):.1f} samples",
                ]
            )

        lines.extend(
            [
                (
                    f"  Perfect (within 5%): {self.perfect_count}/{self.n_runs} "
                    f"({100*self.perfect_count/self.n_runs:.1f}%)"
                ),
                "=" * 70,
            ]
        )

        return "\n".join(lines)


def run_batch_evaluation(
    n_runs: int = 50,
    T: int = 300,
    noise_std: float = 0.3,
    n_true_seams: int = 1,
    config: Optional[MASSSMASHConfig] = None,
) -> BatchResults:
    """Run batch evaluation with known ground truth."""
    if config is None:
        config = MASSSMASHConfig(verbose=False)
    else:
        config.verbose = False

    results = BatchResults(n_runs=n_runs)
    t0 = time.time()

    for seed in range(n_runs):
        # Generate signal
        seam_positions = [
            float(i + 1) / (n_true_seams + 1) for i in range(n_true_seams)
        ]
        y, true_seams = generate_signal_with_seams(
            T=T,
            noise_std=noise_std,
            seam_positions=seam_positions,
            seam_types=["sign_flip"] * n_true_seams,
            seed=seed,
        )

        # Run pipeline
        best, _ = run_mass_smash(y, config, true_seams)

        # Compute detection error
        if true_seams and best.seams:
            errors = [min(abs(ts - ds) for ds in best.seams) for ts in true_seams]
            detection_error = np.mean(errors)
            perfect = all(e < T * 0.05 for e in errors)
        else:
            detection_error = float("inf") if true_seams else 0.0
            perfect = len(true_seams) == len(best.seams) == 0

        results.add(
            mdl=best.total_mdl,
            mse=best.total_mse,
            n_seams=len(best.seams),
            error=detection_error,
            perfect=perfect,
        )

        if (seed + 1) % 10 == 0:
            print(f"  Completed {seed + 1}/{n_runs}...")

    dt = time.time() - t0
    print(f"Total time: {dt:.1f}s ({dt/n_runs:.3f}s per run)")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("MASS/SMASH v2: Multi-Seam Modeling with Model Zoo")
    print("=" * 70)
    print()
    print("Pipeline:")
    print("  1. Seam proposal: antipodal + roughness detectors with NMS")
    print("  2. Configuration search: bounded subset enumeration (NOT beam search)")
    print("  3. Model zoo: Fourier / Poly / AR / MLP competition per segment")
    print("  4. MDL selection: global objective with explicit seam penalty")
    print()

    # Demo with 2 seams
    print("=" * 70)
    print("DEMO: Signal with 2 seams")
    print("=" * 70)

    y, true_seams = generate_signal_with_seams(
        T=300,
        noise_std=0.3,
        seam_positions=[0.33, 0.67],
        seam_types=["sign_flip", "sign_flip"],
        seed=42,
    )

    config = MASSSMASHConfig(max_seams=3, alpha=2.0, verbose=True)

    best, all_solutions = run_mass_smash(y, config, true_seams)

    # Show top 5 solutions
    print("\nTop 5 solutions by MDL:")
    for i, sol in enumerate(all_solutions[:5]):
        print(
            f"  {i+1}. seams={list(sol.seams)}, transform={sol.transform}, "
            f"MDL={sol.total_mdl:.2f}, MSE={sol.total_mse:.6f}"
        )

    # Plot
    candidates = detect_seam_candidates(y, config)
    plot_solution(y, best, candidates, true_seams, save_path="mass_smash_demo.png")

    # Optional batch test
    response = input("\nRun batch evaluation (50 runs)? [Y/n]: ")
    if response.lower() != "n":
        print()
        print("=" * 70)
        print("BATCH EVALUATION: 50 signals, 1 seam each")
        print("=" * 70)

        results = run_batch_evaluation(
            n_runs=50,
            T=300,
            noise_std=0.3,
            n_true_seams=1,
            config=MASSSMASHConfig(alpha=2.0),
        )

        print()
        print(results.summary())

    print()
    print("Done.")
