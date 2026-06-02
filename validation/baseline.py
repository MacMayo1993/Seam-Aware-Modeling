"""Simple |dB/dt| threshold baseline current sheet detector."""
import numpy as np


def baseline_detector(B_mag, times, threshold_sigma=2.5, min_separation_s=300):
    """
    Baseline current sheet detector: threshold on |dB/dt|.

    Returns: (detected_peaks array, dBdt array)
    """
    from scipy.signal import find_peaks

    dBdt = np.abs(np.gradient(B_mag, times))

    mu, sigma = np.mean(dBdt), np.std(dBdt)
    height = mu + threshold_sigma * sigma

    dt = np.median(np.diff(times))
    min_sep = int(min_separation_s / dt)

    peaks, _ = find_peaks(dBdt, height=height, distance=min_sep)
    return peaks, dBdt
