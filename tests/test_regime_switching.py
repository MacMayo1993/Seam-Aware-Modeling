"""
Tests for regime switching and entropy reduction via seam-aware modeling.

This module validates that non-orientable geometry, antipodal symmetry, and
parity-aware transformations provide measurable improvements when modeling
signals with genuine regime changes.

Key questions answered:
1. Does seam-aware modeling reduce residual variance for sign-flipped data?
2. Does variance scaling homogenize heteroscedastic regimes?
3. Does polynomial detrending handle level/trend shifts?
4. How much improvement do we get in each case?
"""

import numpy as np
import pytest

try:
    from scipy.optimize import curve_fit

    HAS_SCIPY_OPTIMIZE = True
except ImportError:
    HAS_SCIPY_OPTIMIZE = False

from seamaware.core.flip_atoms import (
    CompositeFlipAtom,
    PolynomialDetrendAtom,
    SignFlipAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
)
from seamaware.core.mdl import compute_mdl, mdl_improvement


class PolynomialBaseline:
    """Simple polynomial baseline model for testing."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self._coeffs = None

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        t = np.arange(len(data))
        self._coeffs = np.polyfit(t, data, self.degree)
        return np.polyval(self._coeffs, t)

    def num_params(self) -> int:
        return self.degree + 1


def _sin_model(t, a, b, c, d):
    """Sinusoidal model for curve fitting."""
    return a * np.sin(b * t + c) + d


# =============================================================================
# Test Sign Flip (Parity / Antipodal Symmetry)
# =============================================================================


class TestSignFlipRegime:
    """Test improvements for sign-flip regime changes (antipodal geometry)."""

    @pytest.mark.skipif(not HAS_SCIPY_OPTIMIZE, reason="scipy.optimize not available")
    def test_sign_flip_reduces_residual_variance(self):
        """Sign flip correction should reduce residual variance.

        Key insight: When we flip a discontinuous signal to make it
        continuous, sinusoidal models fit much better.
        """
        np.random.seed(42)
        n = 200
        seam = 100

        # Create signal with sign flip at seam
        t = np.linspace(0, 4 * np.pi, n)
        signal = np.sin(t)
        signal[seam:] *= -1  # Flip sign after seam
        signal += 0.1 * np.random.randn(n)  # Add noise

        atom = SignFlipAtom()
        flipped = atom.apply(signal, seam)

        # Fit sinusoid to original (discontinuous)
        popt_orig, _ = curve_fit(_sin_model, t, signal, p0=[1, 1, 0, 0], maxfev=5000)
        pred_orig = _sin_model(t, *popt_orig)
        var_orig = np.var(signal - pred_orig)

        # Fit sinusoid to flipped (now continuous)
        popt_flip, _ = curve_fit(_sin_model, t, flipped, p0=[1, 1, 0, 0], maxfev=5000)
        pred_flip = _sin_model(t, *popt_flip)
        var_flip = np.var(flipped - pred_flip)

        # Flipped should have MUCH lower residual variance
        assert var_flip < var_orig * 0.5, (
            f"Flipped signal should fit better: var_orig={var_orig:.4f}, "
            f"var_flip={var_flip:.4f}"
        )

        # Print improvement
        improvement = (var_orig - var_flip) / var_orig * 100
        print(f"\n  Sign flip variance reduction: {improvement:.1f}%")

    def test_sign_flip_is_involution(self):
        """Sign flip should be its own inverse (F² = I)."""
        np.random.seed(42)
        signal = np.random.randn(100)
        seam = 50

        atom = SignFlipAtom()
        once = atom.apply(signal, seam)
        twice = atom.apply(once, seam)

        np.testing.assert_allclose(
            twice, signal, err_msg="Sign flip should be an involution"
        )


# =============================================================================
# Test Variance Shift (Heteroscedasticity)
# =============================================================================


class TestVarianceShiftRegime:
    """Test improvements for variance shift regime changes."""

    def test_variance_scaling_homogenizes(self):
        """Variance scaling should reduce variance ratio across seam."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Create signal with 9:1 variance ratio
        pre_seam = np.random.randn(seam)  # std=1
        post_seam = 3.0 * np.random.randn(n - seam)  # std=3 (var=9)
        signal = np.concatenate([pre_seam, post_seam])

        atom = VarianceScaleAtom()
        atom.fit_params(signal, seam)
        scaled = atom.apply(signal, seam)

        # Check variance ratio before/after
        var_ratio_before = np.var(signal[seam:]) / np.var(signal[:seam])
        var_ratio_after = np.var(scaled[seam:]) / np.var(scaled[:seam])

        assert var_ratio_after < var_ratio_before, (
            f"Variance should be more homogeneous: "
            f"before={var_ratio_before:.2f}, after={var_ratio_after:.2f}"
        )

        # Should be close to 1.0 after scaling
        assert (
            0.5 < var_ratio_after < 2.0
        ), f"Variance ratio should be near 1.0, got {var_ratio_after:.2f}"

        print(f"\n  Variance ratio: {var_ratio_before:.1f}x → {var_ratio_after:.2f}x")

    def test_variance_scaling_mdl_improvement(self):
        """Variance scaling should reduce MDL for heteroscedastic data."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Create heteroscedastic signal
        pre = np.random.randn(seam)
        post = 3.0 * np.random.randn(n - seam)
        signal = np.concatenate([pre, post])

        baseline = PolynomialBaseline(degree=1)
        atom = VarianceScaleAtom()

        # Baseline MDL
        pred_baseline = baseline.fit_predict(signal)
        mdl_baseline = compute_mdl(signal, pred_baseline, baseline.num_params())

        # Seam-aware MDL
        atom.fit_params(signal, seam)
        scaled = atom.apply(signal, seam)
        pred_seam = baseline.fit_predict(scaled)
        mdl_seam = compute_mdl(scaled, pred_seam, baseline.num_params() + 1)

        improvement = mdl_improvement(mdl_baseline, mdl_seam)

        assert improvement["effective"], f"Variance scaling should reduce MDL"

        print(f"\n  MDL reduction: {improvement['relative_reduction']*100:.1f}%")


# =============================================================================
# Test Mean/Level Shift
# =============================================================================


class TestMeanShiftRegime:
    """Test improvements for mean shift regime changes."""

    def test_level_shift_detrending(self):
        """Polynomial detrending should remove level discontinuity."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Create signal with level shift
        signal = np.zeros(n) + 0.3 * np.random.randn(n)
        signal[seam:] += 5.0  # Jump of 5 units

        atom = PolynomialDetrendAtom(degree=0)
        atom.fit_params(signal, seam)
        detrended = atom.apply(signal, seam)

        # Check mean difference before/after
        mean_diff_before = abs(np.mean(signal[seam:]) - np.mean(signal[:seam]))
        mean_diff_after = abs(np.mean(detrended[seam:]) - np.mean(detrended[:seam]))

        assert (
            mean_diff_after < mean_diff_before
        ), f"Detrending should reduce mean difference"

        # Should be much closer after detrending
        assert (
            mean_diff_after < 1.0
        ), f"Mean difference should be small after detrending, got {mean_diff_after:.2f}"

        print(f"\n  Level shift: {mean_diff_before:.1f} → {mean_diff_after:.2f}")

    def test_level_shift_mdl_improvement(self):
        """Level shift removal should reduce MDL."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Create signal with level shift
        signal = np.zeros(n) + 0.5 * np.random.randn(n)
        signal[seam:] += 5.0

        baseline = PolynomialBaseline(degree=1)
        atom = PolynomialDetrendAtom(degree=0)

        # Baseline MDL
        pred_baseline = baseline.fit_predict(signal)
        mdl_baseline = compute_mdl(signal, pred_baseline, baseline.num_params())

        # Seam-aware MDL
        atom.fit_params(signal, seam)
        detrended = atom.apply(signal, seam)
        pred_seam = baseline.fit_predict(detrended)
        mdl_seam = compute_mdl(detrended, pred_seam, baseline.num_params() + 1)

        improvement = mdl_improvement(mdl_baseline, mdl_seam)

        assert improvement["effective"], f"Level shift removal should reduce MDL"

        print(f"\n  MDL reduction: {improvement['relative_reduction']*100:.1f}%")


# =============================================================================
# Test Composite Transformations
# =============================================================================


class TestCompositeRegime:
    """Test composite flip atoms for complex regime changes."""

    def test_composite_transformation_invertible(self):
        """Composite transformations should be invertible."""
        np.random.seed(42)
        n = 200
        seam = 100

        signal = np.random.randn(n)
        signal[seam:] *= 2  # Variance change

        atom = CompositeFlipAtom([SignFlipAtom(), VarianceScaleAtom()])
        atom.fit_params(signal, seam)

        transformed = atom.apply(signal, seam)
        recovered = atom.inverse(transformed, seam)

        np.testing.assert_allclose(
            recovered,
            signal,
            rtol=1e-10,
            err_msg="Composite transformation should be invertible",
        )

    def test_composite_homogenizes_variance(self):
        """Composite with VarianceScaleAtom should homogenize variance."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Signal with both sign flip and variance change
        signal = np.sin(np.linspace(0, 4 * np.pi, n))
        signal[seam:] *= -1
        signal[:seam] += 0.3 * np.random.randn(seam)
        signal[seam:] += 1.0 * np.random.randn(n - seam)

        atom = CompositeFlipAtom([SignFlipAtom(), VarianceScaleAtom()])
        atom.fit_params(signal, seam)
        transformed = atom.apply(signal, seam)

        var_ratio_before = np.var(signal[seam:]) / np.var(signal[:seam])
        var_ratio_after = np.var(transformed[seam:]) / np.var(transformed[:seam])

        assert (
            var_ratio_after < var_ratio_before
        ), f"Composite should reduce variance ratio"


# =============================================================================
# Test No-Seam Baseline (Sanity Check)
# =============================================================================


class TestNoSeamBaseline:
    """Verify behavior when there's no actual regime change."""

    @pytest.mark.skipif(not HAS_SCIPY_OPTIMIZE, reason="scipy.optimize not available")
    def test_spurious_flip_increases_error(self):
        """Applying sign flip to continuous data should increase error."""
        np.random.seed(42)
        n = 200
        seam = 100

        # Create smooth signal with NO regime change
        t = np.linspace(0, 4 * np.pi, n)
        signal = np.sin(t) + 0.1 * np.random.randn(n)

        atom = SignFlipAtom()
        flipped = atom.apply(signal, seam)

        # Fit sinusoid to original
        popt_orig, _ = curve_fit(_sin_model, t, signal, p0=[1, 1, 0, 0], maxfev=5000)
        var_orig = np.var(signal - _sin_model(t, *popt_orig))

        # Fit sinusoid to flipped (which now has discontinuity)
        popt_flip, _ = curve_fit(_sin_model, t, flipped, p0=[1, 1, 0, 0], maxfev=5000)
        var_flip = np.var(flipped - _sin_model(t, *popt_flip))

        # Spurious flip should make fit WORSE
        assert var_flip > var_orig, (
            f"Spurious flip should increase error: "
            f"var_orig={var_orig:.4f}, var_flip={var_flip:.4f}"
        )


# =============================================================================
# Summary Test
# =============================================================================


class TestRegimeSummary:
    """Summary of all regime type improvements."""

    @pytest.mark.skipif(not HAS_SCIPY_OPTIMIZE, reason="scipy.optimize not available")
    def test_all_regimes_effective(self):
        """All transformations should be effective for their target regimes."""
        np.random.seed(42)
        n = 300
        seam = 150

        results = {}

        # 1. Sign flip - measure variance reduction
        t = np.linspace(0, 4 * np.pi, n)
        signal = np.sin(t)
        signal[seam:] *= -1
        signal += 0.1 * np.random.randn(n)

        atom = SignFlipAtom()
        flipped = atom.apply(signal, seam)

        popt_orig, _ = curve_fit(_sin_model, t, signal, p0=[1, 1, 0, 0], maxfev=5000)
        var_orig = np.var(signal - _sin_model(t, *popt_orig))

        popt_flip, _ = curve_fit(_sin_model, t, flipped, p0=[1, 1, 0, 0], maxfev=5000)
        var_flip = np.var(flipped - _sin_model(t, *popt_flip))

        results["Sign Flip (Parity)"] = (
            (var_orig - var_flip) / var_orig * 100,
            var_flip < var_orig,
        )

        # 2. Variance shift
        np.random.seed(43)
        signal = np.concatenate([np.random.randn(seam), 3 * np.random.randn(n - seam)])
        atom = VarianceScaleAtom()
        atom.fit_params(signal, seam)

        var_ratio_before = np.var(signal[seam:]) / np.var(signal[:seam])
        transformed = atom.apply(signal, seam)
        var_ratio_after = np.var(transformed[seam:]) / np.var(transformed[:seam])

        results["Variance Shift"] = (
            (var_ratio_before - var_ratio_after) / var_ratio_before * 100,
            var_ratio_after < var_ratio_before,
        )

        # 3. Level shift
        np.random.seed(44)
        signal = 0.3 * np.random.randn(n)
        signal[seam:] += 3.0
        atom = PolynomialDetrendAtom(degree=0)
        atom.fit_params(signal, seam)

        mean_diff_before = abs(np.mean(signal[seam:]) - np.mean(signal[:seam]))
        transformed = atom.apply(signal, seam)
        mean_diff_after = abs(np.mean(transformed[seam:]) - np.mean(transformed[:seam]))

        results["Level Shift"] = (
            (mean_diff_before - mean_diff_after) / mean_diff_before * 100,
            mean_diff_after < mean_diff_before,
        )

        # Print summary
        print("\n" + "=" * 65)
        print("  SEAM-AWARE MODELING: TRANSFORMATION EFFECTIVENESS")
        print("=" * 65)
        print(f"  {'Regime Type':<25} {'Improvement':>20} {'Effective':>10}")
        print("-" * 65)
        for regime, (improvement, effective) in results.items():
            status = "YES" if effective else "NO"
            print(f"  {regime:<25} {improvement:>+19.1f}% {status:>10}")
        print("=" * 65)

        # All should be effective
        for regime, (improvement, effective) in results.items():
            assert effective, f"{regime} should be effective"
            assert improvement > 50, f"{regime} should show >50% improvement"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
