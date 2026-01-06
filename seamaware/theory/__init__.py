"""
Theoretical foundations and mathematical derivations.

Modules:
    - k_star: The k* ≈ 0.721 constant and validation
    - quotient_spaces: ℤ₂ eigenspace decomposition mathematics
    - information_geometry: Fisher metric on ℝℙⁿ
"""

from seamaware.theory.k_star import compute_k_star, validate_k_star_convergence

__all__ = [
    "compute_k_star",
    "validate_k_star_convergence",
]
