"""
Input validation and edge case handling.
"""

from typing import Optional, Tuple

import numpy as np


class ValidationError(ValueError):
    """Custom error for validation failures."""

    pass


def validate_signal(
    signal: np.ndarray,
    min_length: int = 2,
    allow_complex: bool = False,
    allow_nan: bool = False,
    name: str = "signal",
) -> np.ndarray:
    """
    Validate and normalize input signal.

    Parameters
    ----------
    signal : array-like
        Input signal
    min_length : int
        Minimum required length
    allow_complex : bool
        Whether complex values are allowed
    allow_nan : bool
        Whether NaN values are allowed
    name : str
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated 1D float64 array

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Convert to array
    try:
        signal = np.asarray(signal)
    except Exception as e:
        raise ValidationError(f"{name}: Cannot convert to array: {e}")

    # Check dimensionality
    if signal.ndim == 0:
        raise ValidationError(f"{name}: Scalar input not allowed")
    if signal.ndim > 1:
        if signal.shape[0] == 1 or signal.shape[1] == 1:
            signal = signal.ravel()
        else:
            raise ValidationError(
                f"{name}: Multi-dimensional input not supported, got shape "
                f"{signal.shape}"
            )

    # Check length
    if len(signal) < min_length:
        raise ValidationError(f"{name}: Length {len(signal)} < minimum {min_length}")

    # Check for empty
    if len(signal) == 0:
        raise ValidationError(f"{name}: Empty array")

    # Check complex
    if np.iscomplexobj(signal):
        if not allow_complex:
            raise ValidationError(f"{name}: Complex values not supported")
        # Keep as complex
    else:
        signal = signal.astype(np.float64)

    # Check for NaN/Inf
    if not allow_nan:
        if np.any(np.isnan(signal)):
            raise ValidationError(f"{name}: Contains NaN values")
    if np.any(np.isinf(signal)):
        raise ValidationError(f"{name}: Contains Inf values")

    return signal


def validate_seam_position(
    position: int, signal_length: int, min_segment: int = 1
) -> int:
    """
    Validate seam position.

    Parameters
    ----------
    position : int
        Seam position index
    signal_length : int
        Total signal length
    min_segment : int
        Minimum points required on each side

    Returns
    -------
    int
        Validated position

    Raises
    ------
    ValidationError
    """
    if not isinstance(position, (int, np.integer)):
        raise ValidationError(f"Seam position must be integer, got {type(position)}")

    position = int(position)

    if position < min_segment:
        raise ValidationError(
            f"Seam position {position} too close to start (min_segment={min_segment})"
        )
    if position >= signal_length - min_segment:
        raise ValidationError(
            f"Seam position {position} too close to end "
            f"(signal_length={signal_length}, min_segment={min_segment})"
        )

    return position


def handle_constant_signal(signal: np.ndarray) -> Tuple[bool, Optional[float]]:
    """
    Check if signal is constant or near-constant.

    Returns
    -------
    Tuple[bool, Optional[float]]
        (is_constant, constant_value)
    """
    if np.all(signal == signal[0]):
        return True, float(signal[0])

    # Check for numerical constancy
    if np.std(signal) < 1e-15 * np.abs(np.mean(signal)):
        return True, float(np.mean(signal))

    return False, None


def safe_log2(x: float, floor: float = 1e-300) -> float:
    """Log2 with floor to prevent -inf."""
    return np.log2(max(x, floor))


def safe_variance(x: np.ndarray, floor: float = 1e-15) -> float:
    """Variance with floor to prevent zero."""
    return max(np.var(x), floor)
