"""
Flip atoms: Transformation operators for seam-aware modeling.

A flip atom is a transformation F : ℂᴺ → ℂᴺ that:
1. Commutes with the antipodal map S: x → -x  (i.e., [F, S] = 0)
2. Has an inverse F⁻¹ (or is an involution: F² = I)
3. Can be applied at a seam location to exploit ℤ₂ symmetry

This module provides the abstract FlipAtom base class and concrete
implementations for common transformations.

NOTE: This is the comprehensive atom implementation with explicit inverse()
and fit_params() methods. For the simplified API used by MASSFramework,
see atoms.py. These modules will be unified in a future release.

Current Usage:
- Use atoms.py when working with MASSFramework (returns AtomResult)
- Use this module (flip_atoms.py) for advanced composition and testing
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FlipAtom(ABC):
    """
    Abstract base class for seam-aware transformations.

    Subclasses must implement:
    - apply(): Transform signal at seam
    - inverse(): Undo transformation
    - num_params(): Return parameter count for MDL

    Optionally can override:
    - fit_params(): Learn transformation parameters from data
    """

    @abstractmethod
    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """
        Apply transformation at seam location.

        Args:
            data: Input signal (length N)
            seam: Seam location τ ∈ [0, N)

        Returns:
            Transformed signal (same shape as input)

        Notes:
            Must return a COPY, not modify input array.
        """
        pass

    @abstractmethod
    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """
        Inverse transformation (must satisfy F⁻¹(F(x)) = x).

        Args:
            data: Transformed signal
            seam: Original seam location

        Returns:
            Original signal

        Notes:
            Must return a COPY, not modify input array.
        """
        pass

    @abstractmethod
    def num_params(self) -> int:
        """
        Parameter count for MDL calculation.

        Returns:
            Number of parameters (seam location excluded)

        Notes:
            The seam location itself costs log₂(N) bits and is
            counted separately in the OrientationTracker.
        """
        pass

    def fit_params(self, data: np.ndarray, seam: int) -> None:
        """
        Optional: Learn transformation parameters from data.

        Default implementation does nothing (for parameter-free atoms).

        Args:
            data: Signal to analyze
            seam: Seam location
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class SignFlipAtom(FlipAtom):
    """
    Sign flip: F(x) = -x after seam.

    This is the simplest flip atom, reversing the sign of all
    samples at and after the seam location.

    Properties:
        - Involution: F² = I
        - Commutes with S: FS = SF ✓
        - Parameters: 0

    Examples:
        >>> atom = SignFlipAtom()
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> flipped = atom.apply(signal, seam=2)
        >>> flipped
        array([ 1,  2, -3, -4, -5])
        >>> recovered = atom.inverse(flipped, seam=2)
        >>> np.array_equal(recovered, signal)
        True
    """

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Apply sign flip at seam."""
        result = data.copy()
        result[seam:] *= -1
        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Inverse is the same as apply (involution)."""
        return self.apply(data, seam)

    def num_params(self) -> int:
        """Zero parameters."""
        return 0


class TimeReversalAtom(FlipAtom):
    """
    Time reversal: Reverse order of samples after seam.

    F(x₁:τ, xτ:N) = (x₁:τ, reverse(xτ:N))

    This preserves energy but changes temporal orientation.

    Properties:
        - Involution: F² = I (reversing twice gives original)
        - Energy preserving: ‖Fx‖ = ‖x‖
        - Parameters: 0

    Examples:
        >>> atom = TimeReversalAtom()
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> reversed = atom.apply(signal, seam=2)
        >>> reversed
        array([1, 2, 5, 4, 3])
    """

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Reverse time after seam."""
        result = data.copy()
        result[seam:] = result[seam:][::-1]
        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Inverse is the same (involution)."""
        return self.apply(data, seam)

    def num_params(self) -> int:
        """Zero parameters."""
        return 0


class VarianceScaleAtom(FlipAtom):
    """
    Variance scaling: Homogenize variance across seam.

    F(xτ:N) = α·xτ:N where α = √(σ²pre / σ²post)

    This makes the variance uniform across the entire signal,
    which can improve modeling when there's a variance shift.

    Properties:
        - Invertible: F⁻¹(y) = y/α
        - Energy scaling: ‖Fx‖² = α²·‖x‖²
        - Parameters: 1 (scaling factor α)

    Attributes:
        scale_factor: Learned scaling α (set via fit_params)

    Examples:
        >>> atom = VarianceScaleAtom()
        >>> signal = np.concatenate([np.ones(50), 2*np.ones(50)])
        >>> atom.fit_params(signal, seam=50)
        >>> scaled = atom.apply(signal, seam=50)
        >>> np.var(scaled[:50])  # Pre-seam variance
        0.0
        >>> np.var(scaled[50:])  # Post-seam variance (scaled)
        0.0
    """

    def __init__(self) -> None:
        """Initialize with default scale factor of 1.0."""
        self.scale_factor: float = 1.0

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Apply variance scaling at seam."""
        result = data.copy()
        result[seam:] *= self.scale_factor
        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Inverse scaling."""
        result = data.copy()
        if self.scale_factor != 0:
            result[seam:] /= self.scale_factor
        return result

    def num_params(self) -> int:
        """One parameter (scale factor)."""
        return 1

    def fit_params(self, data: np.ndarray, seam: int) -> None:
        """
        Learn scale factor from data.

        Computes α = √(σ²pre / σ²post) to homogenize variance.

        Args:
            data: Signal to analyze
            seam: Seam location
        """
        if seam <= 0 or seam >= len(data):
            self.scale_factor = 1.0
            return

        pre_seam = data[:seam]
        post_seam = data[seam:]

        var_pre = np.var(pre_seam) + 1e-10
        var_post = np.var(post_seam) + 1e-10

        self.scale_factor = np.sqrt(var_pre / var_post)

    def __repr__(self) -> str:
        """String representation with scale factor."""
        return f"VarianceScaleAtom(scale={self.scale_factor:.3f})"


class PolynomialDetrendAtom(FlipAtom):
    """
    Polynomial detrending: Remove polynomial trend after seam.

    F(xτ:N) = xτ:N - poly_fit(xτ:N)

    This projects the post-seam segment onto the zero-mean subspace,
    which can help when there's a level shift or trend change.

    Properties:
        - Invertible: F⁻¹(y) = y + poly_fit
        - Mean-removing: mean(Fx) ≈ 0 (for degree ≥ 0)
        - Parameters: degree + 1

    Attributes:
        degree: Polynomial degree (0 = constant, 1 = linear, etc.)
        coeffs: Learned polynomial coefficients

    Examples:
        >>> atom = PolynomialDetrendAtom(degree=1)
        >>> t = np.linspace(0, 1, 100)
        >>> signal = np.concatenate([t[:50], t[50:] + 5])  # Level shift
        >>> atom.fit_params(signal, seam=50)
        >>> detrended = atom.apply(signal, seam=50)
        >>> np.mean(detrended[50:])  # Should be close to 0
        0.0...
    """

    def __init__(self, degree: int = 1):
        """
        Initialize with polynomial degree.

        Args:
            degree: Polynomial degree (default: 1 for linear detrending)

        Raises:
            ValueError: If degree < 0
        """
        if degree < 0:
            raise ValueError(f"Degree must be non-negative, got {degree}")

        self.degree = degree
        self.coeffs: Optional[np.ndarray] = None

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Apply polynomial detrending at seam."""
        result = data.copy()

        if self.coeffs is not None and seam < len(data):
            # Reconstruct polynomial trend
            post_length = len(data) - seam
            t = np.arange(post_length)
            trend = np.polyval(self.coeffs, t)
            result[seam:] -= trend

        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Add polynomial trend back."""
        result = data.copy()

        if self.coeffs is not None and seam < len(data):
            post_length = len(data) - seam
            t = np.arange(post_length)
            trend = np.polyval(self.coeffs, t)
            result[seam:] += trend

        return result

    def num_params(self) -> int:
        """Parameters = degree + 1 polynomial coefficients."""
        return self.degree + 1

    def fit_params(self, data: np.ndarray, seam: int) -> None:
        """
        Fit polynomial to post-seam data.

        Args:
            data: Signal to analyze
            seam: Seam location
        """
        if seam >= len(data) - 1:
            self.coeffs = None
            return

        post_seam = data[seam:]
        t = np.arange(len(post_seam))

        # Fit polynomial (returns coefficients highest degree first)
        self.coeffs = np.polyfit(t, post_seam, self.degree)

    def __repr__(self) -> str:
        """String representation with degree."""
        return f"PolynomialDetrendAtom(degree={self.degree})"


class CompositeFlipAtom(FlipAtom):
    """
    Composite of multiple flip atoms applied sequentially.

    F_composite = F_n ∘ F_{n-1} ∘ ... ∘ F_1

    This allows combining multiple transformations, e.g.,
    sign flip + variance scaling.

    Warning:
        Composition may break ℤ₂ commutativity. Verify that
        [F_composite, S] = 0 for your specific atoms.

    Examples:
        >>> sign_flip = SignFlipAtom()
        >>> var_scale = VarianceScaleAtom()
        >>> composite = CompositeFlipAtom([sign_flip, var_scale])
        >>> signal = np.random.randn(100)
        >>> transformed = composite.apply(signal, seam=50)
    """

    def __init__(self, atoms: list[FlipAtom]):
        """
        Initialize with list of atoms.

        Args:
            atoms: List of FlipAtom instances to compose

        Raises:
            ValueError: If atoms list is empty
        """
        if not atoms:
            raise ValueError("Composite atom requires at least one atom")

        self.atoms = atoms

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Apply atoms in sequence."""
        result = data.copy()
        for atom in self.atoms:
            result = atom.apply(result, seam)
        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """Apply inverse atoms in reverse order."""
        result = data.copy()
        for atom in reversed(self.atoms):
            result = atom.inverse(result, seam)
        return result

    def num_params(self) -> int:
        """Sum of parameters from all atoms."""
        return sum(atom.num_params() for atom in self.atoms)

    def fit_params(self, data: np.ndarray, seam: int) -> None:
        """Fit parameters for all atoms."""
        current_data = data.copy()
        for atom in self.atoms:
            atom.fit_params(current_data, seam)
            current_data = atom.apply(current_data, seam)

    def __repr__(self) -> str:
        """String representation listing all atoms."""
        atom_names = [atom.__class__.__name__ for atom in self.atoms]
        return f"CompositeFlipAtom([{', '.join(atom_names)}])"
