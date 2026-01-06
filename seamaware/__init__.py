"""
SeamAware: Non-Orientable Modeling for Time Series Analysis

A Python package for detecting and exploiting orientation discontinuities
(seams) in time series data through the lens of quotient space topology.
"""

__version__ = "0.1.0"
__author__ = "Mac Mayo"
__license__ = "Apache-2.0"

# Import key classes for convenience
from seamaware.core.flip_atoms import FlipAtom, SignFlipAtom
from seamaware.core.mdl import compute_mdl, delta_mdl
from seamaware.core.orientation import OrientationTracker
from seamaware.models.mass_framework import MASSFramework, MASSResult
from seamaware.theory.k_star import compute_k_star

__all__ = [
    "FlipAtom",
    "SignFlipAtom",
    "compute_mdl",
    "delta_mdl",
    "OrientationTracker",
    "MASSFramework",
    "MASSResult",
    "compute_k_star",
]
