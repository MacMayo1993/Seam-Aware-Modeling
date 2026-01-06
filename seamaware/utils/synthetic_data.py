"""
Synthetic data generation for benchmarking and validation.

Generates signals with known seam locations and properties,
enabling controlled testing of detection and modeling algorithms.
"""

from typing import List, Optional, Tuple

import numpy as np


def generate_sign_flip_signal(
    length: int = 200,
    seam: Optional[int] = None,
    noise_std: float = 0.1,
    base_signal: str = "sin",
    frequency: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate signal with sign flip at seam.

    Args:
        length: Signal length
        seam: Seam location (default: middle)
        noise_std: Noise standard deviation
        base_signal: Base signal type ('sin', 'cos', 'chirp', 'sawtooth')
        frequency: Base frequency (cycles per signal length)
        seed: Random seed

    Returns:
        (signal, true_seam_location)

    Examples:
        >>> signal, seam = generate_sign_flip_signal(200, noise_std=0.1)
        >>> len(signal)
        200
        >>> seam
        100
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if seam is None:
        seam = length // 2

    # Generate base signal
    t = np.linspace(0, 2 * np.pi * frequency, length)

    if base_signal == "sin":
        clean = np.sin(t)
    elif base_signal == "cos":
        clean = np.cos(t)
    elif base_signal == "chirp":
        # Frequency increases linearly
        phase = np.cumsum(t / length)
        clean = np.sin(2 * np.pi * phase)
    elif base_signal == "sawtooth":
        clean = 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))
    else:
        raise ValueError(f"Unknown base_signal: {base_signal}")

    # Apply sign flip
    clean[seam:] *= -1

    # Add noise
    noise = rng.normal(0, noise_std, length)
    signal = clean + noise

    return signal, seam


def generate_variance_shift_signal(
    length: int = 200,
    seam: Optional[int] = None,
    variance_ratio: float = 4.0,
    base_signal: str = "noise",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate signal with variance shift at seam.

    Args:
        length: Signal length
        seam: Seam location (default: middle)
        variance_ratio: Ratio of post-seam to pre-seam variance
        base_signal: Base signal type ('noise', 'ar1')
        seed: Random seed

    Returns:
        (signal, true_seam_location)

    Examples:
        >>> signal, seam = generate_variance_shift_signal(200, variance_ratio=4.0)
        >>> pre_var = np.var(signal[:seam])
        >>> post_var = np.var(signal[seam:])
        >>> 3.0 < post_var / pre_var < 5.0
        True
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if seam is None:
        seam = length // 2

    if base_signal == "noise":
        # White noise with variance shift
        pre_seam = rng.normal(0, 1.0, seam)
        post_seam = rng.normal(0, np.sqrt(variance_ratio), length - seam)
        signal = np.concatenate([pre_seam, post_seam])

    elif base_signal == "ar1":
        # AR(1) process with variance shift
        phi = 0.7  # AR coefficient

        # Pre-seam
        pre_seam = np.zeros(seam)
        pre_seam[0] = rng.normal(0, 1.0)
        for i in range(1, seam):
            pre_seam[i] = phi * pre_seam[i - 1] + rng.normal(0, 1.0)

        # Post-seam
        post_seam = np.zeros(length - seam)
        post_seam[0] = phi * pre_seam[-1] + rng.normal(0, np.sqrt(variance_ratio))
        for i in range(1, length - seam):
            post_seam[i] = phi * post_seam[i - 1] + rng.normal(
                0, np.sqrt(variance_ratio)
            )

        signal = np.concatenate([pre_seam, post_seam])

    else:
        raise ValueError(f"Unknown base_signal: {base_signal}")

    return signal, seam


def generate_polynomial_kink_signal(
    length: int = 200,
    seam: Optional[int] = None,
    noise_std: float = 0.1,
    pre_degree: int = 1,
    post_degree: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate signal with polynomial regime change (kink).

    Args:
        length: Signal length
        seam: Seam location (default: middle)
        noise_std: Noise standard deviation
        pre_degree: Polynomial degree before seam
        post_degree: Polynomial degree after seam
        seed: Random seed

    Returns:
        (signal, true_seam_location)

    Examples:
        >>> signal, seam = generate_polynomial_kink_signal(200)
        >>> len(signal)
        200
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if seam is None:
        seam = length // 2

    # Pre-seam: polynomial of degree pre_degree
    t_pre = np.linspace(0, 1, seam)
    if pre_degree == 0:
        pre_signal = np.ones(seam)
    elif pre_degree == 1:
        pre_signal = t_pre
    elif pre_degree == 2:
        pre_signal = t_pre**2
    else:
        pre_signal = t_pre**pre_degree

    # Post-seam: polynomial of degree post_degree
    t_post = np.linspace(0, 1, length - seam)
    if post_degree == 0:
        post_signal = np.ones(length - seam)
    elif post_degree == 1:
        post_signal = t_post
    elif post_degree == 2:
        post_signal = -(t_post**2)  # Negative curvature
    else:
        post_signal = (-1) ** post_degree * t_post**post_degree

    # Ensure continuity at seam
    offset = pre_signal[-1] - post_signal[0]
    post_signal += offset

    # Concatenate
    clean = np.concatenate([pre_signal, post_signal])

    # Add noise
    noise = rng.normal(0, noise_std, length)
    signal = clean + noise

    return signal, seam


def generate_multi_seam_signal(
    length: int = 200,
    num_seams: int = 3,
    seam_type: str = "sign_flip",
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate signal with multiple seams.

    Args:
        length: Signal length
        num_seams: Number of seams
        seam_type: Type of seam ('sign_flip', 'variance_shift')
        noise_std: Noise standard deviation
        seed: Random seed

    Returns:
        (signal, list_of_seam_locations)

    Examples:
        >>> signal, seams = generate_multi_seam_signal(300, num_seams=3)
        >>> len(seams)
        3
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Evenly space seams
    segment_length = length // (num_seams + 1)
    seam_locations = [segment_length * (i + 1) for i in range(num_seams)]

    # Generate base signal
    t = np.linspace(0, 4 * np.pi, length)
    signal = np.sin(t)

    # Apply seams
    if seam_type == "sign_flip":
        for i, seam in enumerate(seam_locations):
            # Alternate flips
            if i % 2 == 0:
                signal[seam:] *= -1

    elif seam_type == "variance_shift":
        current_var = 1.0
        for i, seam in enumerate(seam_locations):
            # Increase variance at each seam
            current_var *= 1.5
            signal[seam:] *= current_var

    else:
        raise ValueError(f"Unknown seam_type: {seam_type}")

    # Add noise
    noise = rng.normal(0, noise_std, length)
    signal = signal + noise

    return signal, seam_locations


def generate_hvac_like_signal(
    length: int = 1000,
    on_duration: int = 100,
    off_duration: int = 150,
    temp_setpoint: float = 20.0,
    temp_swing: float = 2.0,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate HVAC-like signal with regime switching (ON/OFF cycles).

    Simulates temperature oscillations from heating/cooling cycles.

    Args:
        length: Signal length (timesteps)
        on_duration: Duration of ON phase
        off_duration: Duration of OFF phase
        temp_setpoint: Target temperature
        temp_swing: Temperature variation during ON phase
        noise_std: Measurement noise
        seed: Random seed

    Returns:
        (temperature_signal, list_of_regime_switches)

    Examples:
        >>> signal, switches = generate_hvac_like_signal(1000)
        >>> len(switches) > 0
        True
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    signal = np.zeros(length)
    seams = []

    t = 0
    phase = "heating"  # Start with heating

    while t < length:
        if phase == "heating":
            duration = min(on_duration, length - t)
            # Temperature rises exponentially toward setpoint + swing
            tau = on_duration / 3  # Time constant
            signal[t : t + duration] = temp_setpoint + temp_swing * (
                1 - np.exp(-np.arange(duration) / tau)
            )
            t += duration
            phase = "cooling"
            if t < length:
                seams.append(t)

        else:  # cooling
            duration = min(off_duration, length - t)
            # Temperature falls exponentially toward setpoint - swing
            tau = off_duration / 3
            signal[t : t + duration] = temp_setpoint - temp_swing * (
                1 - np.exp(-np.arange(duration) / tau)
            )
            t += duration
            phase = "heating"
            if t < length:
                seams.append(t)

    # Add noise
    noise = rng.normal(0, noise_std, length)
    signal = signal + noise

    return signal, seams[:-1] if seams else []  # Remove last if at boundary


def generate_known_snr_signal(
    length: int = 200,
    seam: Optional[int] = None,
    target_snr: float = 1.0,
    base_signal: str = "sin",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Generate signal with precisely controlled SNR for k* validation.

    Args:
        length: Signal length
        seam: Seam location (default: middle)
        target_snr: Target signal-to-noise ratio
        base_signal: Base signal type
        seed: Random seed

    Returns:
        (signal, true_seam, actual_snr)

    Examples:
        >>> signal, seam, snr = generate_known_snr_signal(200, target_snr=0.721)
        >>> 0.7 < snr < 0.75
        True
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if seam is None:
        seam = length // 2

    # Generate base signal
    t = np.linspace(0, 4 * np.pi, length)
    if base_signal == "sin":
        clean = np.sin(t)
    else:
        clean = np.cos(t)

    # Apply sign flip
    clean[seam:] *= -1

    # Compute signal power
    signal_power = np.var(clean)

    # Compute noise power to achieve target SNR
    noise_power = signal_power / target_snr
    noise_std = np.sqrt(noise_power)

    # Add noise
    noise = rng.normal(0, noise_std, length)
    signal = clean + noise

    # Verify actual SNR
    actual_snr = np.var(clean) / np.var(noise)

    return signal, seam, actual_snr
