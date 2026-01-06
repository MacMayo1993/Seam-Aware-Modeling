"""
Core mathematical operations for seam-aware modeling.

Modules:
    - flip_atoms: Transformation operators (FlipAtom base class)
    - mdl: Minimum Description Length calculations
    - orientation: OrientationTracker for quotient space navigation
    - seam_detection: Roughness-based seam detection algorithms
"""

from seamaware.core.flip_atoms import (
    FlipAtom,
    PolynomialDetrendAtom,
    SignFlipAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
)
from seamaware.core.mdl import compute_mdl, delta_mdl
from seamaware.core.orientation import OrientationTracker
from seamaware.core.seam_detection import compute_roughness, detect_seams_roughness

__all__ = [
    "FlipAtom",
    "SignFlipAtom",
    "TimeReversalAtom",
    "VarianceScaleAtom",
    "PolynomialDetrendAtom",
    "compute_mdl",
    "delta_mdl",
    "OrientationTracker",
    "detect_seams_roughness",
    "compute_roughness",
]
