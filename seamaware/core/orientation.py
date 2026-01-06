"""
Orientation tracking for navigation in quotient space ℂᴺ/ℤ₂.

This module implements the "anti-bit" framework for tracking position
in non-orientable quotient spaces, enabling proper signal reconstruction
and MDL cost accounting.
"""

import numpy as np
from typing import List, Optional


class OrientationTracker:
    """
    Track orientation state across seams in quotient space ℂᴺ/ℤ₂ ≅ ℝℙᴺ⁻¹.

    In the quotient space, each point x is identified with its antipode -x.
    We cannot globally distinguish them, but we can track **transitions**
    between sheets (original vs. flipped).

    The orientation state vector o ∈ {±1}ᴺ tracks:
        o(t) = +1  if position t is on the original sheet
        o(t) = -1  if position t is on the flipped sheet

    Each seam represents a crossing from one sheet to the other.

    Attributes:
        length: Signal length N
        orientations: State vector o(t) ∈ {±1}ᴺ
        seams: List of seam locations τᵢ

    Examples:
        >>> tracker = OrientationTracker(length=100)
        >>> tracker.flip_at(50)  # Seam at t=50
        >>> tracker.get_orientation(30)  # Before seam
        1
        >>> tracker.get_orientation(70)  # After seam
        -1
        >>> tracker.encoding_cost_bits()  # Cost to encode seam location
        6.643...  # log₂(100)
    """

    def __init__(self, length: int):
        """
        Initialize tracker for signal of given length.

        Args:
            length: Signal length N

        Raises:
            ValueError: If length <= 0
        """
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")

        self.length = length
        self.orientations = np.ones(length, dtype=np.int8)
        self.seams: List[int] = []

    def flip_at(self, seam: int) -> None:
        """
        Toggle orientation starting at seam location.

        This represents crossing from one sheet of the quotient space
        to the other via the antipodal map.

        Args:
            seam: Location τ where flip occurs (0 ≤ τ < N)

        Raises:
            ValueError: If seam is out of bounds
        """
        if seam < 0 or seam >= self.length:
            raise ValueError(f"Seam {seam} out of bounds [0, {self.length})")

        # Toggle orientation for all points at or after seam
        self.orientations[seam:] *= -1
        self.seams.append(seam)

    def get_orientation(self, t: int) -> int:
        """
        Return ±1 orientation at time index t.

        Args:
            t: Time index (0 ≤ t < N)

        Returns:
            +1 (original sheet) or -1 (flipped sheet)

        Raises:
            ValueError: If t is out of bounds
        """
        if t < 0 or t >= self.length:
            raise ValueError(f"Index {t} out of bounds [0, {self.length})")

        return int(self.orientations[t])

    def get_orientation_vector(self) -> np.ndarray:
        """
        Return full orientation state vector.

        Returns:
            Array of shape (N,) with values ±1
        """
        return self.orientations.copy()

    def encoding_cost_bits(self) -> float:
        """
        Cost in bits to encode seam locations.

        For k seams in a signal of length N, each seam location
        requires log₂(N) bits to encode.

        Returns:
            k·log₂(N) bits

        Notes:
            This is the "anti-bit" cost that must be offset by
            improved model fit for seams to be MDL-justified.
        """
        k = len(self.seams)
        if k == 0:
            return 0.0

        return k * np.log2(self.length)

    def num_seams(self) -> int:
        """
        Return number of seams.

        Returns:
            Number of flips applied
        """
        return len(self.seams)

    def get_seam_locations(self) -> List[int]:
        """
        Return list of seam locations.

        Returns:
            Sorted list of seam indices
        """
        return sorted(self.seams.copy())

    def reset(self) -> None:
        """Reset to all +1 orientations (original sheet)."""
        self.orientations.fill(1)
        self.seams.clear()

    def apply_to_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply orientation state to signal (element-wise product).

        This reconstructs the signal in the "canonical" orientation
        by multiplying each sample by its orientation state.

        Args:
            signal: Input signal (length N)

        Returns:
            Orientation-corrected signal

        Raises:
            ValueError: If signal length doesn't match tracker length

        Examples:
            >>> signal = np.ones(100)
            >>> tracker = OrientationTracker(100)
            >>> tracker.flip_at(50)
            >>> corrected = tracker.apply_to_signal(signal)
            >>> corrected[30]  # Before seam
            1.0
            >>> corrected[70]  # After seam
            -1.0
        """
        if len(signal) != self.length:
            raise ValueError(
                f"Signal length ({len(signal)}) != tracker length ({self.length})"
            )

        return signal * self.orientations

    def from_seam_list(self, seams: List[int]) -> None:
        """
        Reconstruct orientation state from list of seam locations.

        Args:
            seams: List of seam indices to apply

        Raises:
            ValueError: If any seam is out of bounds
        """
        self.reset()
        for seam in sorted(seams):
            self.flip_at(seam)

    def __repr__(self) -> str:
        """String representation showing seam count and locations."""
        k = self.num_seams()
        if k == 0:
            return f"OrientationTracker(length={self.length}, seams=[])"
        else:
            seam_str = ", ".join(str(s) for s in self.get_seam_locations())
            return f"OrientationTracker(length={self.length}, seams=[{seam_str}])"

    def __eq__(self, other: object) -> bool:
        """Check equality based on seam locations."""
        if not isinstance(other, OrientationTracker):
            return False
        return (
            self.length == other.length
            and self.get_seam_locations() == other.get_seam_locations()
        )


def compute_antisymmetric_energy(signal: np.ndarray) -> float:
    """
    Compute the antisymmetric energy fraction α₋.

    For a signal x, decompose via ℤ₂ projection operators:
        x = 𝐏₊x + 𝐏₋x

    where:
        𝐏₊ = (I + S)/2  (symmetric component)
        𝐏₋ = (I - S)/2  (antisymmetric component)
        S is the antipodal map (x → -x)

    The antisymmetric energy fraction is:
        α₋ = ‖𝐏₋x‖² / ‖x‖²

    Args:
        signal: Input signal (length N)

    Returns:
        α₋ ∈ [0, 1] measuring "non-orientability"

    Notes:
        High α₋ indicates the signal naturally inhabits the
        antisymmetric eigenspace of the ℤ₂ action, making
        seam-aware modeling more beneficial.

    Examples:
        >>> signal = np.array([1, -1, 1, -1])  # Fully antisymmetric
        >>> alpha = compute_antisymmetric_energy(signal)
        >>> alpha
        1.0
    """
    # Antipodal map: Sx = -x
    # For discrete signals, we approximate by reversing and negating
    signal_flipped = -signal[::-1]

    # Projection operators
    # 𝐏₊x = (x + Sx)/2
    # 𝐏₋x = (x - Sx)/2
    p_minus = (signal - signal_flipped) / 2

    # Energy fraction
    energy_total = np.sum(signal**2)
    energy_antisymmetric = np.sum(p_minus**2)

    if energy_total == 0:
        return 0.0

    alpha_minus = energy_antisymmetric / energy_total
    return float(np.clip(alpha_minus, 0.0, 1.0))
