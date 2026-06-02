"""
strong_baselines.py
-------------------
Multi-baseline comparison module for MASS/SMASH evaluation.

Each baseline detector returns (peaks: np.ndarray, scores: np.ndarray).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nms(indices: np.ndarray, scores: np.ndarray, min_separation: int = 10):
    """Greedy NMS: keep highest-score peak; suppress neighbours within min_separation."""
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(len(indices)):
            if abs(indices[j] - indices[i]) < min_separation:
                suppressed.add(j)
    keep = sorted(keep, key=lambda i: indices[i])
    return indices[keep], scores[keep]


# ---------------------------------------------------------------------------
# Baseline 1: vector |dB/dt| gradient
# ---------------------------------------------------------------------------

def vector_gradient_baseline(
    B: np.ndarray,
    times: np.ndarray,
    threshold_sigma: float = 3.0,
    min_separation: int = 10,
):
    """
    |dB/dt| vector gradient detector.

    B: (T, 3) array. Compute ||dB/dt|| pointwise, then threshold at
    mean + threshold_sigma * std, find local maxima, return sorted peaks.

    Returns
    -------
    peaks  : np.ndarray of int
    scores : np.ndarray of float
    """
    dt = np.diff(times)
    dt = np.where(dt == 0, 1e-12, dt)
    dBdt = np.diff(B, axis=0) / dt[:, None]          # (T-1, 3)
    mag = np.linalg.norm(dBdt, axis=1)                # (T-1,)

    threshold = mag.mean() + threshold_sigma * mag.std()
    candidates, props = find_peaks(mag, height=threshold)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    heights = props["peak_heights"]
    peaks, scores = _nms(candidates, heights, min_separation=min_separation)
    return peaks, scores


# ---------------------------------------------------------------------------
# Baseline 2: angular rotation threshold
# ---------------------------------------------------------------------------

def angular_rotation_baseline(
    B: np.ndarray,
    times: np.ndarray,
    window: int = 20,
    threshold_deg: float = 45.0,
    min_separation: int = 10,
):
    """
    Angular rotation threshold detector.

    At each position τ compute:
        angle(τ) = arccos(clip(B̂[τ-w] · B̂[τ+w], -1, 1)) * 180/π
    where B̂ = B / (||B|| + ε). Return local maxima where angle > threshold_deg.

    Returns
    -------
    peaks  : np.ndarray of int
    scores : np.ndarray of float
    """
    eps = 1e-12
    norm = np.linalg.norm(B, axis=1, keepdims=True)
    Bhat = B / (norm + eps)

    T = len(B)
    angles = np.zeros(T)
    for tau in range(window, T - window):
        dot = np.clip(np.dot(Bhat[tau - window], Bhat[tau + window]), -1.0, 1.0)
        angles[tau] = np.degrees(np.arccos(dot))

    candidates, props = find_peaks(angles, height=threshold_deg)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    heights = props["peak_heights"]
    peaks, scores = _nms(candidates, heights, min_separation=min_separation)
    return peaks, scores


# ---------------------------------------------------------------------------
# Baseline 3: PVI (single lag)
# ---------------------------------------------------------------------------

def pvi_baseline(
    B: np.ndarray,
    times: np.ndarray,
    lag: int | None = None,
    threshold: float = 3.0,
    min_separation: int = 10,
):
    """
    Standard PVI: PVI(t) = |ΔB(t,τ)| / sqrt(<|ΔB|²>)
    where ΔB(t,τ) = B(t+τ) - B(t), τ = lag samples (default 1).

    B: (T, 3) array of vector field.

    Returns
    -------
    peaks  : np.ndarray of int
    scores : np.ndarray of float
    """
    if lag is None:
        lag = 1

    T = len(B)
    if lag >= T:
        return np.array([], dtype=int), np.array([], dtype=float)

    delta = B[lag:] - B[:-lag]                        # (T-lag, 3)
    mag = np.linalg.norm(delta, axis=1)               # (T-lag,)
    rms = np.sqrt(np.mean(mag ** 2))
    if rms == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    pvi = mag / rms

    candidates, props = find_peaks(pvi, height=threshold)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    heights = props["peak_heights"]
    peaks, scores = _nms(candidates, heights, min_separation=min_separation)
    return peaks, scores


# ---------------------------------------------------------------------------
# Baseline 4: PVI multi-lag
# ---------------------------------------------------------------------------

def pvi_multi_lag_baseline(
    B: np.ndarray,
    times: np.ndarray,
    lags: tuple = (1, 5, 10, 20),
    threshold: float = 3.0,
    min_separation: int = 10,
):
    """
    PVI evaluated at multiple lags; score = max PVI across lags at each point.
    Returns local maxima where max_PVI > threshold.

    Returns
    -------
    peaks  : np.ndarray of int
    scores : np.ndarray of float
    """
    T = len(B)
    combined = np.zeros(T)

    for lag in lags:
        if lag >= T:
            continue
        delta = B[lag:] - B[:-lag]
        mag = np.linalg.norm(delta, axis=1)
        rms = np.sqrt(np.mean(mag ** 2))
        if rms == 0:
            continue
        pvi = mag / rms
        # align to full-length array (use the midpoint convention: index i = t+lag//2)
        offset = lag // 2
        end = offset + len(pvi)
        if end > T:
            end = T
            pvi = pvi[: end - offset]
        combined[offset:end] = np.maximum(combined[offset:end], pvi)

    candidates, props = find_peaks(combined, height=threshold)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    heights = props["peak_heights"]
    peaks, scores = _nms(candidates, heights, min_separation=min_separation)
    return peaks, scores


# ---------------------------------------------------------------------------
# Baseline 5: PELT changepoint (ruptures)
# ---------------------------------------------------------------------------

def pelt_vector_baseline(
    B_1d: np.ndarray,
    penalty_scale: float = 1.0,
    min_separation: int = 10,
):
    """
    PELT changepoint detection on a 1D signal (e.g., Bz) using ruptures library.
    Uses l2 cost with penalty = penalty_scale * log(N) * var(signal).
    Returns changepoint positions. Falls back to empty array if ruptures unavailable.

    Note: B_1d is a 1-D array (pass B[:, 2] for Bz).

    Returns
    -------
    peaks  : np.ndarray of int
    scores : np.ndarray of float  (uniform score = 1.0 for each changepoint)
    """
    try:
        import ruptures as rpt
    except ImportError:
        return np.array([], dtype=int), np.array([], dtype=float)

    N = len(B_1d)
    signal_var = np.var(B_1d)
    if signal_var == 0:
        signal_var = 1.0
    penalty = penalty_scale * np.log(N) * signal_var

    try:
        algo = rpt.Pelt(model="l2").fit(B_1d.astype(float).reshape(-1, 1))
        result = algo.predict(pen=penalty)
        # ruptures returns indices in (1-indexed end of segment); last entry == N
        cps = np.array([r - 1 for r in result if r < N], dtype=int)
        if len(cps) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        scores = np.ones(len(cps), dtype=float)
        peaks, scores = _nms(cps, scores, min_separation=min_separation)
        return peaks, scores
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=float)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _precision_recall_f1(
    detected: np.ndarray,
    true_peaks: np.ndarray,
    tolerance: int,
):
    if len(detected) == 0:
        precision = 0.0
        recall = 0.0
    else:
        tp = 0
        matched_true = set()
        for d in detected:
            for j, t in enumerate(true_peaks):
                if j not in matched_true and abs(d - t) <= tolerance:
                    tp += 1
                    matched_true.add(j)
                    break
        precision = tp / len(detected) if len(detected) > 0 else 0.0
        recall = tp / len(true_peaks) if len(true_peaks) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Master comparison
# ---------------------------------------------------------------------------

def compare_all_baselines(
    B: np.ndarray,
    times: np.ndarray,
    true_peaks: list | np.ndarray,
    tolerance_samples: int = 10,
) -> dict:
    """
    Run all five baselines and MASS/SMASH on B (T, 3) signal and return a
    comparison dict of {name: {precision, recall, f1, n_detected}}.

    MASS/SMASH operates on Bz = B[:, 2].
    """
    from seamaware.pipeline import run_mass_smash, MASSSMASHConfig

    true_peaks = np.asarray(true_peaks, dtype=int)

    results = {}

    # --- vector gradient ---
    peaks, scores = vector_gradient_baseline(B, times)
    p, r, f = _precision_recall_f1(peaks, true_peaks, tolerance_samples)
    results["VectorGradient"] = dict(precision=p, recall=r, f1=f, n_detected=len(peaks))

    # --- angular rotation ---
    peaks, scores = angular_rotation_baseline(B, times)
    p, r, f = _precision_recall_f1(peaks, true_peaks, tolerance_samples)
    results["AngularRotation"] = dict(precision=p, recall=r, f1=f, n_detected=len(peaks))

    # --- PVI single lag ---
    peaks, scores = pvi_baseline(B, times)
    p, r, f = _precision_recall_f1(peaks, true_peaks, tolerance_samples)
    results["PVI_lag1"] = dict(precision=p, recall=r, f1=f, n_detected=len(peaks))

    # --- PVI multi-lag ---
    peaks, scores = pvi_multi_lag_baseline(B, times)
    p, r, f = _precision_recall_f1(peaks, true_peaks, tolerance_samples)
    results["PVI_MultiLag"] = dict(precision=p, recall=r, f1=f, n_detected=len(peaks))

    # --- PELT (Bz) ---
    peaks, scores = pelt_vector_baseline(B[:, 2])
    p, r, f = _precision_recall_f1(peaks, true_peaks, tolerance_samples)
    results["PELT_Bz"] = dict(precision=p, recall=r, f1=f, n_detected=len(peaks))

    # --- MASS/SMASH (Bz) ---
    try:
        cfg = MASSSMASHConfig(verbose=False)
        best, _ = run_mass_smash(B[:, 2], config=cfg)
        ms_peaks = np.array(best.seams, dtype=int) if best.seams else np.array([], dtype=int)
    except Exception as exc:
        print(f"  [MASS/SMASH error: {exc}]")
        ms_peaks = np.array([], dtype=int)
    p, r, f = _precision_recall_f1(ms_peaks, true_peaks, tolerance_samples)
    results["MASS_SMASH"] = dict(precision=p, recall=r, f1=f, n_detected=len(ms_peaks))

    return results


# ---------------------------------------------------------------------------
# __main__ benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    np.random.seed(42)

    T = 500
    seam_fracs = [0.25, 0.5, 0.75]

    from seamaware.pipeline import generate_signal_with_seams

    Bz, true_seams = generate_signal_with_seams(
        T=T,
        noise_std=0.25,
        seam_positions=seam_fracs,
        seam_types=["sign_flip", "frequency_change", "phase_shift"],
        seed=42,
    )

    t = np.linspace(0, 4 * np.pi, T)
    Bx = np.sin(t + 0.3) + 0.15 * np.random.randn(T)
    By = np.cos(t + 0.7) + 0.15 * np.random.randn(T)
    B = np.stack([Bx, By, Bz], axis=1)   # (T, 3)
    times = np.linspace(0.0, 100.0, T)

    print(f"Synthetic benchmark  T={T}  true seams={true_seams}")
    print()

    results = compare_all_baselines(B, times, true_seams, tolerance_samples=10)

    # Print table
    col_w = 16
    header = (
        f"{'Detector':<{col_w}} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N_det':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<{col_w}} {m['precision']:>10.3f} {m['recall']:>8.3f}"
            f" {m['f1']:>8.3f} {m['n_detected']:>7d}"
        )
    print()
    print(f"True seam count: {len(true_seams)}")
    sys.exit(0)
