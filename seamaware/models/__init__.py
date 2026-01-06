"""
Complete modeling frameworks for seam-aware time series analysis.

Modules:
    - mass_framework: Main MASS (Manifold-Aware Seam Segmentation) implementation
    - baselines: Fourier, polynomial, and AR baseline models
"""

from seamaware.models.mass_framework import MASSFramework, MASSResult
from seamaware.models.baselines import (
    FourierBaseline,
    PolynomialBaseline,
    ARBaseline,
)

__all__ = [
    "MASSFramework",
    "MASSResult",
    "FourierBaseline",
    "PolynomialBaseline",
    "ARBaseline",
]
