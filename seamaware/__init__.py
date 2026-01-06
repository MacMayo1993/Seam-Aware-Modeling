"""
SeamAware: Non-Orientable Modeling for Time Series Analysis

Detects and exploits orientation discontinuities (seams) in time series
data using MDL-justified transformations.

Example
-------
>>> from seamaware import MASSFramework
>>> mass = MASSFramework()
>>> result = mass.fit(signal)
>>> print(f"MDL reduction: {result.mdl_reduction_percent:.1f}%")
"""

__version__ = "0.2.0"
__author__ = "Mac Mayo"

from .core.atoms import (
    AUXILIARY_ATOMS,
    INVOLUTION_ATOMS,
    FlipAtom,
    PolynomialDetrendAtom,
    SignFlipAtom,
    SignTimeReversalAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
    get_atom,
)
from .core.detection import (
    SeamDetectionResult,
    detect_seam,
    detect_seam_cusum,
    detect_seam_roughness,
)
from .core.mdl import (
    LikelihoodType,
    MDLResult,
    compute_k_star,
    compute_mdl,
    mdl_improvement,
)
from .core.validation import (
    ValidationError,
    validate_seam_position,
    validate_signal,
)
from .mass import MASSFramework, MASSResult
from .models.baselines import FourierBaseline

# Convenient constants
K_STAR = compute_k_star()  # ≈ 0.7213

__all__ = [
    # Main API
    "MASSFramework",
    "MASSResult",
    # MDL
    "compute_mdl",
    "MDLResult",
    "LikelihoodType",
    "compute_k_star",
    "mdl_improvement",
    # Detection
    "detect_seam",
    "detect_seam_cusum",
    "detect_seam_roughness",
    "SeamDetectionResult",
    # Atoms
    "FlipAtom",
    "SignFlipAtom",
    "TimeReversalAtom",
    "SignTimeReversalAtom",
    "VarianceScaleAtom",
    "PolynomialDetrendAtom",
    "get_atom",
    "INVOLUTION_ATOMS",
    "AUXILIARY_ATOMS",
    # Validation
    "validate_signal",
    "validate_seam_position",
    "ValidationError",
    # Models
    "FourierBaseline",
    # Constants
    "K_STAR",
]
