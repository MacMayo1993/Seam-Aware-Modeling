"""
Comprehensive edge case tests.

Run with: pytest tests/test_edge_cases.py -v
"""

import numpy as np
import pytest

from seamaware.core.atoms import SignFlipAtom, get_atom
from seamaware.core.detection import detect_seam_cusum
from seamaware.core.mdl import LikelihoodType, compute_mdl
from seamaware.core.validation import (
    ValidationError,
    validate_seam_position,
    validate_signal,
)


class TestValidation:
    """Test input validation."""

    def test_empty_signal(self):
        with pytest.raises(ValidationError, match="Length 0"):
            validate_signal(np.array([]))

    def test_scalar_signal(self):
        with pytest.raises(ValidationError, match="Scalar"):
            validate_signal(5.0)

    def test_short_signal(self):
        with pytest.raises(ValidationError, match="Length"):
            validate_signal(np.array([1.0]), min_length=2)

    def test_nan_signal(self):
        with pytest.raises(ValidationError, match="NaN"):
            validate_signal(np.array([1.0, np.nan, 2.0]))

    def test_inf_signal(self):
        with pytest.raises(ValidationError, match="Inf"):
            validate_signal(np.array([1.0, np.inf, 2.0]))

    def test_complex_signal_rejected(self):
        with pytest.raises(ValidationError, match="Complex"):
            validate_signal(np.array([1 + 2j, 3 + 4j]))

    def test_complex_signal_allowed(self):
        result = validate_signal(np.array([1 + 2j, 3 + 4j]), allow_complex=True)
        assert np.iscomplexobj(result)

    def test_2d_signal_flattened(self):
        result = validate_signal(np.array([[1, 2, 3]]))
        assert result.shape == (3,)

    def test_valid_signal(self):
        signal = validate_signal(np.array([1.0, 2.0, 3.0]))
        assert signal.dtype == np.float64
        assert len(signal) == 3


class TestSeamPosition:
    """Test seam position validation."""

    def test_seam_at_zero(self):
        with pytest.raises(ValidationError, match="too close to start"):
            validate_seam_position(0, 100, min_segment=1)

    def test_seam_at_end(self):
        with pytest.raises(ValidationError, match="too close to end"):
            validate_seam_position(99, 100, min_segment=1)

    def test_seam_valid(self):
        pos = validate_seam_position(50, 100, min_segment=10)
        assert pos == 50


class TestMDL:
    """Test MDL computation edge cases."""

    def test_perfect_fit(self):
        """Near-zero residuals shouldn't cause -inf."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = data.copy()  # Perfect predictions
        result = compute_mdl(data, pred, num_params=2)
        assert np.isfinite(result.total_bits)

    def test_constant_signal(self):
        """Constant signal has zero variance."""
        data = np.ones(100)
        pred = np.ones(100)
        result = compute_mdl(data, pred, num_params=1)
        assert np.isfinite(result.total_bits)

    def test_single_sample(self):
        """Single sample edge case."""
        # Should work but model_bits term needs n>1
        data = np.array([1.0, 2.0])  # Minimum 2
        pred = np.array([1.0, 2.0])
        result = compute_mdl(data, pred, num_params=1)
        assert result.num_samples == 2

    def test_laplace_likelihood(self):
        """Laplace likelihood for heavy tails."""
        data = np.random.standard_cauchy(100)  # Heavy tailed
        pred = np.zeros(100)
        result = compute_mdl(
            data, pred, num_params=0, likelihood=LikelihoodType.LAPLACE
        )
        assert result.likelihood_type == "laplace"

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mdl(np.array([1, 2, 3]), np.array([1, 2]), num_params=1)


class TestFlipAtoms:
    """Test flip atom involution properties."""

    @pytest.mark.parametrize(
        "atom_name", ["sign_flip", "time_reversal", "sign_time_reversal"]
    )
    def test_involution_property(self, atom_name):
        """True involutions satisfy F(F(x)) = x."""
        atom = get_atom(atom_name)
        assert atom.is_involution

        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        seam_pos = 50

        # Apply twice
        result1 = atom.apply(signal, seam_pos)
        result2 = atom.apply(result1.transformed, seam_pos)

        np.testing.assert_allclose(signal, result2.transformed, atol=1e-14)

    def test_variance_scale_not_involution(self):
        """Variance scaling is NOT an involution."""
        atom = get_atom("variance_scale")
        assert not atom.is_involution

    def test_seam_at_boundary(self):
        """Seam at index 0 should raise."""
        atom = SignFlipAtom()
        with pytest.raises(ValueError):
            atom.apply(np.array([1, 2, 3]), seam_position=-1)


class TestDetection:
    """Test seam detection edge cases."""

    def test_obvious_seam(self):
        """Clear sign flip should be detected."""
        signal = np.ones(100)
        signal[50:] = -1  # Clear flip at 50
        result = detect_seam_cusum(signal)
        # Should be within 5 samples of true position
        assert abs(result.position - 50) < 5

    def test_no_seam(self):
        """Constant signal has no real seam."""
        signal = np.ones(100) + 0.01 * np.random.randn(100)
        result = detect_seam_cusum(signal)
        # Confidence should be low
        assert result.confidence < 0.5

    def test_short_signal(self):
        """Short signal should raise."""
        with pytest.raises(ValueError, match="too short"):
            detect_seam_cusum(np.array([1, 2, 3]), min_segment_length=5)

    def test_multiple_candidates(self):
        """Signal with multiple jumps returns candidates."""
        signal = np.concatenate([np.zeros(30), np.ones(40), np.zeros(30)])
        result = detect_seam_cusum(signal)
        assert len(result.all_candidates) >= 1


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_with_seam(self):
        """Test complete MASSFramework with clear seam."""
        from seamaware import MASSFramework

        # Generate signal with seam
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        signal[100:] *= -1  # Clear seam at 100
        signal += 0.1 * np.random.randn(200)

        mass = MASSFramework()
        result = mass.fit(signal)

        # Should detect seam
        assert result.seam_detected
        # Position should be close to 100
        assert result.seam_position is not None
        assert abs(result.seam_position - 100) < 10

    def test_full_pipeline_without_seam(self):
        """Test MASSFramework with no seam."""
        from seamaware import MASSFramework

        # Pure sine wave
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t) + 0.1 * np.random.randn(200)

        mass = MASSFramework(min_confidence=0.5)
        result = mass.fit(signal)

        # May or may not detect seam, but should not crash
        assert isinstance(result.mdl_reduction, float)
        assert np.isfinite(result.mdl_reduction)


class TestEdgeCasesContinued:
    """Additional edge cases."""

    def test_very_short_signal(self):
        """Signal at minimum length."""
        from seamaware import MASSFramework

        signal = np.random.randn(20)  # Minimum length
        mass = MASSFramework()

        # Should handle without crashing
        result = mass.fit(signal)
        assert isinstance(result, object)

    def test_high_noise_signal(self):
        """Signal with very high noise."""
        from seamaware import MASSFramework

        signal = np.random.randn(200) * 100  # Very noisy
        mass = MASSFramework()

        result = mass.fit(signal)
        # Should not crash
        assert np.isfinite(result.baseline_mdl.total_bits)

    def test_zero_signal(self):
        """All-zero signal."""
        from seamaware import MASSFramework

        signal = np.zeros(200)
        mass = MASSFramework()

        result = mass.fit(signal)
        # Should handle constant signal
        assert np.isfinite(result.baseline_mdl.total_bits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
