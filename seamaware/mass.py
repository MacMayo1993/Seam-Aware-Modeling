"""
MASS Framework: Multi-scale Antipodal Seam Search

The main entry point for seam-aware modeling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .core.atoms import get_atom
from .core.detection import detect_seam
from .core.mdl import (
    LikelihoodType,
    MDLResult,
    compute_mdl,
    mdl_improvement,
)
from .core.validation import validate_signal
from .models.baselines import FourierBaseline


@dataclass
class MASSResult:
    """Complete result from MASS framework."""

    # MDL scores
    baseline_mdl: MDLResult
    seam_mdl: MDLResult

    # Detection
    seam_detected: bool
    seam_position: Optional[int]
    seam_confidence: float
    detection_method: str

    # Atom applied
    atom_used: Optional[str]

    # Predictions
    baseline_prediction: np.ndarray
    seam_prediction: np.ndarray
    corrected_signal: np.ndarray

    # Improvement metrics
    mdl_reduction: float  # bits saved
    mdl_reduction_percent: float
    compression_ratio: float

    # Metadata
    snr_estimate: float
    above_k_star: bool

    @property
    def effective(self) -> bool:
        """Whether seam-aware modeling improved MDL."""
        return self.mdl_reduction > 0


class MASSFramework:
    """
    Multi-scale Antipodal Seam Search Framework.

    Detects orientation discontinuities (seams) in time series and applies
    flip atoms to reduce MDL.

    Parameters
    ----------
    baseline : str
        Baseline model type: "fourier", "polynomial", "ar"
    baseline_params : dict
        Parameters for baseline model
    detection_method : str
        Seam detection method: "cusum", "roughness", "ensemble"
    atoms : list of str
        Flip atoms to try: "sign_flip", "time_reversal", etc.
    likelihood : str
        Likelihood model: "gaussian", "laplace", "cauchy"
    min_confidence : float
        Minimum detection confidence to apply seam correction

    Example
    -------
    >>> mass = MASSFramework()
    >>> result = mass.fit(signal)
    >>> print(f"MDL reduction: {result.mdl_reduction_percent:.1f}%")
    """

    def __init__(
        self,
        baseline: str = "fourier",
        baseline_params: Optional[Dict[str, Any]] = None,
        detection_method: str = "cusum",
        atoms: Optional[List[str]] = None,
        likelihood: str = "gaussian",
        min_confidence: float = 0.3,
    ):
        self.baseline_type = baseline
        self.baseline_params = baseline_params or {"K": 3}
        self.detection_method = detection_method
        self.atoms = atoms or ["sign_flip"]
        self.likelihood = LikelihoodType(likelihood)
        self.min_confidence = min_confidence

        # Validate atoms
        for atom_name in self.atoms:
            get_atom(atom_name)  # Raises if invalid

    def _get_baseline(self):
        """Instantiate baseline model."""
        if self.baseline_type == "fourier":
            return FourierBaseline(**self.baseline_params)
        else:
            raise ValueError(f"Unknown baseline: {self.baseline_type}")

    def _estimate_snr(self, signal: np.ndarray, prediction: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        signal_power = np.var(prediction)
        noise_power = np.var(signal - prediction)
        if noise_power < 1e-15:
            return float("inf")
        return np.sqrt(signal_power / noise_power)

    def fit(self, signal: np.ndarray) -> MASSResult:
        """
        Fit MASS framework to signal.

        Parameters
        ----------
        signal : np.ndarray
            1D time series

        Returns
        -------
        MASSResult
            Complete analysis result
        """
        # Validate input
        signal = validate_signal(signal, min_length=20)

        # Fit baseline model
        baseline = self._get_baseline()
        baseline_pred = baseline.fit_predict(signal)
        baseline_mdl = compute_mdl(
            signal, baseline_pred, baseline.num_params(), likelihood=self.likelihood
        )

        # Detect seam
        detection = detect_seam(signal, method=self.detection_method)

        # Check if detection is confident enough
        if detection.confidence < self.min_confidence:
            # No confident seam detected - return baseline
            return MASSResult(
                baseline_mdl=baseline_mdl,
                seam_mdl=baseline_mdl,
                seam_detected=False,
                seam_position=None,
                seam_confidence=detection.confidence,
                detection_method=detection.method,
                atom_used=None,
                baseline_prediction=baseline_pred,
                seam_prediction=baseline_pred,
                corrected_signal=signal,
                mdl_reduction=0.0,
                mdl_reduction_percent=0.0,
                compression_ratio=1.0,
                snr_estimate=self._estimate_snr(signal, baseline_pred),
                above_k_star=False,
            )

        # Try each atom at each candidate position and keep best
        best_mdl = baseline_mdl
        best_atom = None
        best_corrected = signal
        best_pred = baseline_pred
        best_position = None

        # Use a grid search over positions to be robust to detection errors
        # Try every 5th position (more robust than relying only on detection)
        n = len(signal)
        min_seg = 10
        candidate_positions = list(range(min_seg, n - min_seg, 5))

        for candidate_pos in candidate_positions:
            for atom_name in self.atoms:
                atom = get_atom(atom_name)

                try:
                    result = atom.apply(signal, candidate_pos)
                    corrected = result.transformed

                    # Fit baseline to corrected signal
                    pred = baseline.fit_predict(corrected)

                    # Compute MDL (+1 param for seam position)
                    mdl = compute_mdl(
                        corrected,
                        pred,
                        baseline.num_params() + 1,
                        likelihood=self.likelihood,
                    )

                    if mdl.total_bits < best_mdl.total_bits:
                        best_mdl = mdl
                        best_atom = atom_name
                        best_corrected = corrected
                        best_pred = pred
                        best_position = candidate_pos

                except Exception:
                    # Atom failed (e.g., boundary issues) - skip
                    continue

        # Compute improvement metrics
        improvement = mdl_improvement(baseline_mdl, best_mdl)
        k_star = 0.721
        snr = self._estimate_snr(signal, baseline_pred)

        return MASSResult(
            baseline_mdl=baseline_mdl,
            seam_mdl=best_mdl,
            seam_detected=best_atom is not None,
            seam_position=best_position if best_atom else None,
            seam_confidence=detection.confidence,
            detection_method=detection.method,
            atom_used=best_atom,
            baseline_prediction=baseline_pred,
            seam_prediction=best_pred,
            corrected_signal=best_corrected,
            mdl_reduction=improvement["absolute_reduction"],
            mdl_reduction_percent=improvement["relative_reduction"] * 100,
            compression_ratio=improvement["compression_ratio"],
            snr_estimate=snr,
            above_k_star=snr > k_star,
        )

    def fit_predict(self, signal: np.ndarray) -> MASSResult:
        """Alias for fit()."""
        return self.fit(signal)
