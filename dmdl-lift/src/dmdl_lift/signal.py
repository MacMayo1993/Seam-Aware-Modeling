"""Signal generation with coherent k-mer-based lift.

The key insight: phase space size must match available samples.
Using k-mer alone (not k-mer × time mod M2) keeps phase count manageable:
- Periodic word: 2 k-mers regardless of L
- Fibonacci word: L+1 k-mers (Sturmian property)

This ensures sufficient samples per phase for reliable AR(1) estimation.
"""
import numpy as np
from .phase import kmer_id


def generate_signal(word, lifted=True, noise=0.05, seed=0, L=8):
    """Generate AR(1) signal with optional k-mer-dependent dynamics.

    Parameters
    ----------
    word : np.ndarray
        Symbolic sequence ('A' or 'B')
    lifted : bool
        If True, AR(1) coefficients depend on k-mer context
    noise : float
        Innovation noise scale
    seed : int
        Random seed
    L : int
        K-mer length for phase computation

    Returns
    -------
    x : np.ndarray
        Generated time series

    Notes
    -----
    When lifted=True, dynamics are:
    - alpha = alpha_base + 0.08 * sin(2π * kmer / 2^L)
    - beta = 0.15 * cos(2π * kmer / 2^L)

    This creates smooth phase-dependent modulation across k-mer space.
    """
    rng = np.random.default_rng(seed)
    T = len(word)
    x = np.zeros(T)
    x[0] = 1.0

    # Base AR(1) dynamics
    alpha_base = 0.85

    for t in range(1, T):
        if lifted and t >= L - 1:
            # K-mer dependent modulation
            kmer = kmer_id(word, t, L)
            if kmer is not None:
                # Smooth phase modulation - different alpha per k-mer
                phase_angle = 2 * np.pi * kmer / (2 ** L)
                alpha = alpha_base + 0.08 * np.sin(phase_angle)
                # Also modulate the intercept
                beta = 0.15 * np.cos(phase_angle)
            else:
                alpha = alpha_base
                beta = 0.0
        else:
            alpha = alpha_base
            beta = 0.0

        x[t] = alpha * x[t-1] + beta + noise * rng.normal()

    return x


def generate_null_signal(word, noise=0.05, seed=0):
    """Generate pure AR(1) signal with NO word-dependent structure.

    This is the true ground truth for rank-1 dynamics.
    The word is passed for API consistency but NOT used.

    Parameters
    ----------
    word : np.ndarray
        Symbolic sequence (not used, for API consistency)
    noise : float
        Innovation noise scale
    seed : int
        Random seed

    Returns
    -------
    x : np.ndarray
        Generated time series with pure AR(1) dynamics
    """
    rng = np.random.default_rng(seed)
    T = len(word)
    x = np.zeros(T)
    x[0] = 1.0

    alpha = 0.85

    for t in range(1, T):
        x[t] = alpha * x[t-1] + noise * rng.normal()

    return x
