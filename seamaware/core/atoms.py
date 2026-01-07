"""
Flip atoms: transformations that exploit orientation symmetry.

True ℤ₂ involutions satisfy F(F(x)) = x (applying twice gives identity).
Auxiliary transforms may not be involutions but expose hidden structure.

NOTE: This module provides the main framework API used by MASSFramework.
For advanced usage with explicit inverse() and fit_params() methods,
see flip_atoms.py. These modules will be unified in a future release.

Current Usage:
- Use this module (atoms.py) when working with MASSFramework
- Use flip_atoms.py for lower-level atom composition and testing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AtomResult:
    """Result of applying a flip atom."""

    transformed: np.ndarray
    is_involution: bool
    atom_name: str
    seam_position: int


class FlipAtom(ABC):
    """Abstract base class for flip atoms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Atom identifier."""
        pass

    @property
    @abstractmethod
    def is_involution(self) -> bool:
        """True if F(F(x)) = x."""
        pass

    @abstractmethod
    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        """Apply the transformation at seam_position."""
        pass

    def verify_involution(
        self, signal: np.ndarray, seam_position: int, tol: float = 1e-10
    ) -> bool:
        """Verify that applying twice returns original (for true involutions)."""
        if not self.is_involution:
            return False
        result1 = self.apply(signal, seam_position)
        result2 = self.apply(result1.transformed, seam_position)
        return np.allclose(signal, result2.transformed, atol=tol)

    def num_params(self) -> int:
        """Parameter count for MDL calculation."""
        return 0


class SignFlipAtom(FlipAtom):
    """
    Sign inversion: x → -x after seam.

    This is a true ℤ₂ involution: (-(-x)) = x.
    """

    @property
    def name(self) -> str:
        return "sign_flip"

    @property
    def is_involution(self) -> bool:
        return True

    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        signal = np.asarray(signal, dtype=np.float64).copy()
        n = len(signal)

        if seam_position < 0 or seam_position >= n:
            raise ValueError(f"seam_position {seam_position} out of bounds [0, {n})")

        # Flip sign after seam
        signal[seam_position:] *= -1

        return AtomResult(
            transformed=signal,
            is_involution=True,
            atom_name=self.name,
            seam_position=seam_position,
        )


class TimeReversalAtom(FlipAtom):
    """
    Time reversal: reverse the segment after seam.

    This is a true ℤ₂ involution: reversing twice gives original.
    """

    @property
    def name(self) -> str:
        return "time_reversal"

    @property
    def is_involution(self) -> bool:
        return True

    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        signal = np.asarray(signal, dtype=np.float64).copy()
        n = len(signal)

        if seam_position < 0 or seam_position >= n:
            raise ValueError(f"seam_position {seam_position} out of bounds [0, {n})")

        # Reverse after seam
        signal[seam_position:] = signal[seam_position:][::-1]

        return AtomResult(
            transformed=signal,
            is_involution=True,
            atom_name=self.name,
            seam_position=seam_position,
        )


class SignTimeReversalAtom(FlipAtom):
    """
    Combined sign flip + time reversal.

    This is a true ℤ₂ involution: (PT)² = P²T² = I (since P and T commute here).
    """

    @property
    def name(self) -> str:
        return "sign_time_reversal"

    @property
    def is_involution(self) -> bool:
        return True

    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        signal = np.asarray(signal, dtype=np.float64).copy()
        n = len(signal)

        if seam_position < 0 or seam_position >= n:
            raise ValueError(f"seam_position {seam_position} out of bounds [0, {n})")

        # Reverse and flip sign after seam
        signal[seam_position:] = -signal[seam_position:][::-1]

        return AtomResult(
            transformed=signal,
            is_involution=True,
            atom_name=self.name,
            seam_position=seam_position,
        )


class VarianceScaleAtom(FlipAtom):
    """
    Variance scaling: normalize variance of segment after seam to match before.

    NOT an involution in general (σ → σ₀/σ applied twice doesn't give identity
    unless σ = σ₀).

    This is an auxiliary transform for heteroskedastic signals.
    """

    @property
    def name(self) -> str:
        return "variance_scale"

    @property
    def is_involution(self) -> bool:
        return False  # NOT an involution

    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        signal = np.asarray(signal, dtype=np.float64).copy()
        n = len(signal)

        if seam_position < 1 or seam_position >= n - 1:
            raise ValueError(
                f"seam_position {seam_position} requires segments on both sides"
            )

        # Compute variances
        var_before = np.var(signal[:seam_position])
        var_after = np.var(signal[seam_position:])

        if var_after < 1e-15:
            # Can't scale near-zero variance
            return AtomResult(
                transformed=signal,
                is_involution=False,
                atom_name=self.name,
                seam_position=seam_position,
            )

        # Scale after segment to match before
        scale = np.sqrt(var_before / var_after) if var_before > 1e-15 else 1.0

        mean_after = np.mean(signal[seam_position:])
        signal[seam_position:] = (
            signal[seam_position:] - mean_after
        ) * scale + mean_after

        return AtomResult(
            transformed=signal,
            is_involution=False,
            atom_name=self.name,
            seam_position=seam_position,
        )

    def num_params(self) -> int:
        """One parameter (scale factor)."""
        return 1


class PolynomialDetrendAtom(FlipAtom):
    """
    Polynomial detrending: remove polynomial trend from segment after seam.

    NOT an involution (detrending is not invertible without storing the trend).

    This is an auxiliary transform for non-stationary signals.
    """

    def __init__(self, degree: int = 1):
        self.degree = degree

    @property
    def name(self) -> str:
        return f"polynomial_detrend_deg{self.degree}"

    @property
    def is_involution(self) -> bool:
        return False  # NOT an involution

    def apply(self, signal: np.ndarray, seam_position: int) -> AtomResult:
        signal = np.asarray(signal, dtype=np.float64).copy()
        n = len(signal)

        if seam_position < 1 or seam_position >= n - self.degree - 1:
            raise ValueError(
                f"seam_position {seam_position} leaves insufficient points for degree "
                f"{self.degree}"
            )

        # Fit polynomial to segment after seam
        segment = signal[seam_position:]
        t = np.arange(len(segment))

        coeffs = np.polyfit(t, segment, self.degree)
        trend = np.polyval(coeffs, t)

        # Remove trend, preserving mean of segment before
        mean_before = np.mean(signal[:seam_position])
        signal[seam_position:] = segment - trend + mean_before

        return AtomResult(
            transformed=signal,
            is_involution=False,
            atom_name=self.name,
            seam_position=seam_position,
        )

    def num_params(self) -> int:
        """Parameters = degree + 1 polynomial coefficients."""
        return self.degree + 1


# Registry of all atoms
ATOM_REGISTRY = {
    "sign_flip": SignFlipAtom,
    "time_reversal": TimeReversalAtom,
    "sign_time_reversal": SignTimeReversalAtom,
    "variance_scale": VarianceScaleAtom,
    "polynomial_detrend": PolynomialDetrendAtom,
}

INVOLUTION_ATOMS = ["sign_flip", "time_reversal", "sign_time_reversal"]
AUXILIARY_ATOMS = ["variance_scale", "polynomial_detrend"]


def get_atom(name: str, **kwargs) -> FlipAtom:
    """Get atom instance by name."""
    if name not in ATOM_REGISTRY:
        raise ValueError(
            f"Unknown atom: {name}. Available: {list(ATOM_REGISTRY.keys())}"
        )
    return ATOM_REGISTRY[name](**kwargs) if kwargs else ATOM_REGISTRY[name]()
