"""PVI-based current sheet catalog (ground truth)."""
import numpy as np
import os


def compute_pvi(B_vec, lag=1, window=30):
    """
    Partial Variance of Increments (PVI) — standard current sheet detector.

    PVI(t) = |ΔB(t)| / sqrt(<|ΔB|²>)

    Current sheets: PVI > threshold (typically 3.0)

    Args:
        B_vec: (N, 3) array of magnetic field vectors
        lag: increment lag in samples
        window: normalization window in samples

    Returns:
        pvi: (N,) PVI time series
    """
    dB = np.diff(B_vec, axis=0, prepend=B_vec[:1])
    dB_mag = np.sqrt(np.sum(dB**2, axis=1))

    from numpy.lib.stride_tricks import sliding_window_view
    dB_sq = dB_mag**2
    pad = window // 2
    dB_sq_padded = np.pad(dB_sq, pad, mode='edge')
    windows = sliding_window_view(dB_sq_padded, window)
    local_var = np.sqrt(np.mean(windows, axis=1))[:len(dB_mag)]

    pvi = dB_mag / (local_var + 1e-12)
    return pvi


def get_pvi_events(pvi, times, threshold=3.0, min_separation_s=300):
    """
    Extract current sheet crossing times from PVI series.

    Returns list of (time, pvi_score) tuples, plus peak indices array.
    """
    from scipy.signal import find_peaks
    dt = np.median(np.diff(times))
    min_sep_samples = int(min_separation_s / dt)

    peaks, _ = find_peaks(pvi, height=threshold, distance=min_sep_samples)
    events = [(times[p], pvi[p]) for p in peaks]
    return events, peaks


if __name__ == '__main__':
    B = np.load('outputs/wind_mfi_B.npy')[:, :3]
    times = np.load('outputs/wind_mfi_times.npy')

    pvi = compute_pvi(B)
    events, peaks = get_pvi_events(pvi, times)

    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/catalog_peaks.npy', peaks)
    np.save('outputs/pvi_series.npy', pvi)
    print(f"Found {len(events)} current sheet events (PVI > 3.0)")
