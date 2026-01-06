"""
MASS: Manifold-Aware Seam Segmentation

The main framework for seam-aware time series modeling,
integrating detection, transformation, and MDL-based selection.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import warnings

from seamaware.core.seam_detection import detect_seams_roughness, compute_roughness
from seamaware.core.flip_atoms import (
    FlipAtom,
    SignFlipAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
    PolynomialDetrendAtom,
)
from seamaware.core.orientation import OrientationTracker
from seamaware.core.mdl import compute_mdl, delta_mdl
from seamaware.models.baselines import PolynomialBaseline, FourierBaseline


@dataclass
class MASSResult:
    """
    Results from MASS framework.

    Attributes:
        prediction: Final model prediction
        mdl_score: MDL in bits
        seam_locations: List of detected seam indices
        flip_atoms_used: List of flip atoms applied
        orientation_tracker: OrientationTracker instance
        baseline_mdl: MDL of baseline (no seams)
        improvement: ΔMDL (baseline - seam_aware)
        metadata: Additional information
    """

    prediction: np.ndarray
    mdl_score: float
    seam_locations: List[int]
    flip_atoms_used: List[FlipAtom]
    orientation_tracker: OrientationTracker
    baseline_mdl: float
    improvement: float
    metadata: dict


class MASSFramework:
    """
    Manifold-Aware Seam Segmentation framework.

    The MASS framework:
    1. Detects candidate seams via roughness analysis
    2. Tests flip atoms at each seam
    3. Selects seams that minimize MDL
    4. Returns best model with orientation tracking

    Args:
        baseline_model: Baseline model class (default: PolynomialBaseline)
        baseline_kwargs: Kwargs for baseline model
        flip_atoms: List of flip atoms to try (default: all standard atoms)
        detection_window: Window size for seam detection
        detection_threshold: Threshold sigma for seam detection
        max_seams: Maximum number of seams to consider
        greedy: Use greedy seam addition (faster)

    Examples:
        >>> import numpy as np
        >>> from seamaware import MASSFramework
        >>> # Generate signal with seam
        >>> t = np.linspace(0, 4*np.pi, 200)
        >>> signal = np.sin(t)
        >>> signal[100:] *= -1
        >>> # Apply MASS
        >>> mass = MASSFramework()
        >>> result = mass.fit_predict(signal)
        >>> len(result.seam_locations) > 0
        True
        >>> result.improvement > 0  # MDL improvement
        True
    """

    def __init__(
        self,
        baseline_model=PolynomialBaseline,
        baseline_kwargs: Optional[dict] = None,
        flip_atoms: Optional[List[FlipAtom]] = None,
        detection_window: int = 20,
        detection_threshold: float = 2.0,
        max_seams: int = 5,
        greedy: bool = True,
    ):
        """Initialize MASS framework."""
        self.baseline_model = baseline_model
        self.baseline_kwargs = baseline_kwargs or {"degree": 2}

        # Default flip atoms to try
        if flip_atoms is None:
            self.flip_atoms = [
                SignFlipAtom(),
                TimeReversalAtom(),
                VarianceScaleAtom(),
                PolynomialDetrendAtom(degree=1),
            ]
        else:
            self.flip_atoms = flip_atoms

        self.detection_window = detection_window
        self.detection_threshold = detection_threshold
        self.max_seams = max_seams
        self.greedy = greedy

    def fit_predict(self, data: np.ndarray) -> MASSResult:
        """
        Fit MASS model and generate predictions.

        Args:
            data: Input signal

        Returns:
            MASSResult with predictions and metadata

        Raises:
            ValueError: If data is too short
        """
        n = len(data)

        if n < 2 * self.detection_window:
            raise ValueError(
                f"Data length ({n}) must be at least 2*detection_window ({2*self.detection_window})"
            )

        # === Step 1: Baseline model (no seams) ===
        baseline = self.baseline_model(**self.baseline_kwargs)
        baseline_pred = baseline.fit_predict(data)
        baseline_mdl = compute_mdl(data, baseline_pred, baseline.num_params())

        # === Step 2: Detect candidate seams ===
        candidate_seams = detect_seams_roughness(
            data,
            window=self.detection_window,
            threshold_sigma=self.detection_threshold,
            max_seams=self.max_seams * 2,  # Detect more, then filter
        )

        if len(candidate_seams) == 0:
            # No seams detected → return baseline
            tracker = OrientationTracker(n)
            return MASSResult(
                prediction=baseline_pred,
                mdl_score=baseline_mdl,
                seam_locations=[],
                flip_atoms_used=[],
                orientation_tracker=tracker,
                baseline_mdl=baseline_mdl,
                improvement=0.0,
                metadata={"message": "No seams detected"},
            )

        # === Step 3: Greedy seam addition ===
        if self.greedy:
            return self._fit_greedy(data, baseline_mdl, candidate_seams)
        else:
            # Exhaustive search (slower, not implemented yet)
            warnings.warn("Exhaustive search not implemented, using greedy")
            return self._fit_greedy(data, baseline_mdl, candidate_seams)

    def _fit_greedy(
        self, data: np.ndarray, baseline_mdl: float, candidate_seams: List[int]
    ) -> MASSResult:
        """
        Greedy seam addition: add seams one at a time while MDL decreases.

        Args:
            data: Input signal
            baseline_mdl: Baseline MDL (no seams)
            candidate_seams: List of candidate seam locations

        Returns:
            MASSResult with best seams
        """
        n = len(data)
        tracker = OrientationTracker(n)

        best_mdl = baseline_mdl
        best_prediction = None
        best_seams = []
        best_atoms = []

        current_data = data.copy()
        current_mdl = baseline_mdl

        for iteration in range(self.max_seams):
            # Try adding each remaining candidate seam
            improvement_found = False
            best_candidate = None
            best_candidate_atom = None
            best_candidate_mdl = current_mdl
            best_candidate_data = None

            for seam in candidate_seams:
                if seam in best_seams:
                    continue  # Already used

                # Try each flip atom
                for atom in self.flip_atoms:
                    # Fit atom parameters
                    atom.fit_params(current_data, seam)

                    # Apply flip
                    flipped_data = atom.apply(current_data, seam)

                    # Fit baseline model to flipped data
                    baseline = self.baseline_model(**self.baseline_kwargs)
                    pred = baseline.fit_predict(flipped_data)

                    # Compute MDL
                    # Parameters: baseline + atom + seam locations
                    num_params = baseline.num_params() + atom.num_params()
                    mdl = compute_mdl(flipped_data, pred, num_params)

                    # Add seam encoding cost
                    seam_cost = (len(best_seams) + 1) * np.log2(n)
                    mdl += seam_cost

                    # Check if improvement
                    if mdl < best_candidate_mdl:
                        best_candidate_mdl = mdl
                        best_candidate = seam
                        best_candidate_atom = atom
                        best_candidate_data = flipped_data
                        improvement_found = True

            # Add best candidate if it improves MDL
            if improvement_found and best_candidate_mdl < current_mdl:
                best_seams.append(best_candidate)
                best_atoms.append(best_candidate_atom)
                tracker.flip_at(best_candidate)

                current_data = best_candidate_data
                current_mdl = best_candidate_mdl

                # Update best overall
                baseline = self.baseline_model(**self.baseline_kwargs)
                best_prediction = baseline.fit_predict(current_data)
                best_mdl = current_mdl
            else:
                # No improvement found → stop
                break

        # If no seams were added, use baseline
        if len(best_seams) == 0:
            baseline = self.baseline_model(**self.baseline_kwargs)
            best_prediction = baseline.fit_predict(data)
            best_mdl = baseline_mdl

        improvement = baseline_mdl - best_mdl

        return MASSResult(
            prediction=best_prediction,
            mdl_score=best_mdl,
            seam_locations=sorted(best_seams),
            flip_atoms_used=best_atoms,
            orientation_tracker=tracker,
            baseline_mdl=baseline_mdl,
            improvement=improvement,
            metadata={
                "num_candidates": len(candidate_seams),
                "num_seams_added": len(best_seams),
                "iterations": iteration + 1,
            },
        )

    def evaluate_seam(
        self, data: np.ndarray, seam: int, atom: FlipAtom
    ) -> dict:
        """
        Evaluate a single seam with given flip atom.

        Args:
            data: Input signal
            seam: Seam location
            atom: Flip atom to apply

        Returns:
            Dictionary with MDL scores and predictions
        """
        n = len(data)

        # Baseline (no seam)
        baseline = self.baseline_model(**self.baseline_kwargs)
        baseline_pred = baseline.fit_predict(data)
        baseline_mdl = compute_mdl(data, baseline_pred, baseline.num_params())

        # With seam
        atom.fit_params(data, seam)
        flipped = atom.apply(data, seam)

        baseline_seam = self.baseline_model(**self.baseline_kwargs)
        seam_pred = baseline_seam.fit_predict(flipped)

        num_params = baseline_seam.num_params() + atom.num_params()
        seam_mdl = compute_mdl(flipped, seam_pred, num_params)
        seam_mdl += np.log2(n)  # Seam encoding cost

        return {
            "baseline_mdl": baseline_mdl,
            "seam_mdl": seam_mdl,
            "delta_mdl": seam_mdl - baseline_mdl,
            "improvement": baseline_mdl - seam_mdl,
            "baseline_prediction": baseline_pred,
            "seam_prediction": seam_pred,
            "flipped_data": flipped,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MASSFramework(baseline={self.baseline_model.__name__}, "
            f"atoms={len(self.flip_atoms)}, max_seams={self.max_seams})"
        )
