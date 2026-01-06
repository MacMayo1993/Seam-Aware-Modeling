"""
Utility functions for data generation and visualization.

Modules:
    - synthetic_data: Generate benchmark signals with known seams
    - visualization: Plotting utilities for seams and orientations
"""

from seamaware.utils.synthetic_data import (
    generate_sign_flip_signal,
    generate_variance_shift_signal,
    generate_polynomial_kink_signal,
)

__all__ = [
    "generate_sign_flip_signal",
    "generate_variance_shift_signal",
    "generate_polynomial_kink_signal",
]
