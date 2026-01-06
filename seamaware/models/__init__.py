"""
Baseline models for seam-aware time series analysis.

Modules:
    - baselines: Fourier, polynomial, and AR baseline models

Note: MASSFramework is now in seamaware.mass (package root).
"""

from seamaware.models.baselines import (
    ARBaseline,
    FourierBaseline,
    PolynomialBaseline,
)

__all__ = [
    "FourierBaseline",
    "PolynomialBaseline",
    "ARBaseline",
]
