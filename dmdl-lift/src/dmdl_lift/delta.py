"""ΔMDL trajectory computation.

ΔMDL = MDL_rank1 - MDL_rank2

Positive ΔMDL means rank-2 (phase-conditioned) model is preferred.

This is the core lift detection signal: when ΔMDL > 0, the data exhibits
structure that is better explained by phase-dependent dynamics than by
a single global model.
"""
import numpy as np
from .mdl import mdl_rank1, mdl_rank2


def delta_mdl_trajectory(x, word, L=8, window=200, step=10, min_samples=10):
    """Compute ΔMDL over sliding windows.

    Parameters
    ----------
    x : np.ndarray
        Time series
    word : np.ndarray
        Symbolic sequence
    L : int
        K-mer length
    window : int
        Window size for local MDL computation
    step : int
        Step size between windows
    min_samples : int
        Minimum samples per phase for rank-2 model

    Returns
    -------
    ts : np.ndarray
        Time points (window end positions)
    deltas : np.ndarray
        ΔMDL values (positive = rank-2 preferred)
    phase_counts : np.ndarray
        Number of learned phases per window
    ess_per_phase : np.ndarray
        Average ESS per phase per window

    Notes
    -----
    For each window:
    1. Compute rank-1 MDL (global AR(1))
    2. Compute rank-2 MDL (per-k-mer AR(1))
    3. ΔMDL = MDL_1 - MDL_2

    Both models use the same start_idx (L-1) to ensure fair comparison.
    """
    deltas = []
    phase_counts = []
    ess_per_phase_list = []
    ts = []

    start_t = max(window, L)

    for t in range(start_t, len(x)+1, step):
        xw = x[t-window:t]
        ww = word[t-window:t]

        # Rank-1 MDL
        start_idx = L - 1
        mdl1, (a_global, c_global) = mdl_rank1(xw, start_idx=start_idx)

        # Rank-2 MDL (using same start_idx for fair comparison)
        mdl2, n_phases, ess = mdl_rank2(
            xw, ww, L=L, min_samples=min_samples,
            a_global=a_global, c_global=c_global
        )

        deltas.append(mdl1 - mdl2)
        phase_counts.append(n_phases)
        ess_per_phase_list.append(ess)
        ts.append(t)

    return (np.array(ts), np.array(deltas),
            np.array(phase_counts), np.array(ess_per_phase_list))
