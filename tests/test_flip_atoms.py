"""
Tests for flip atoms.
"""

import numpy as np
import pytest

from seamaware.core.flip_atoms import (
    CompositeFlipAtom,
    PolynomialDetrendAtom,
    SignFlipAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
)


def test_sign_flip_involution():
    """SignFlip is its own inverse (involution)."""
    atom = SignFlipAtom()
    signal = np.random.randn(100)
    seam = 50

    flipped = atom.apply(signal, seam)
    recovered = atom.inverse(flipped, seam)

    np.testing.assert_array_almost_equal(signal, recovered)


def test_sign_flip_correctness():
    """SignFlip negates values after seam."""
    atom = SignFlipAtom()
    signal = np.ones(100)
    seam = 50

    flipped = atom.apply(signal, seam)

    # Before seam: unchanged
    assert np.all(flipped[:seam] == 1.0)
    # After seam: negated
    assert np.all(flipped[seam:] == -1.0)


def test_time_reversal_involution():
    """TimeReversal is its own inverse."""
    atom = TimeReversalAtom()
    signal = np.arange(100)
    seam = 50

    reversed_sig = atom.apply(signal, seam)
    recovered = atom.inverse(reversed_sig, seam)

    np.testing.assert_array_almost_equal(signal, recovered)


def test_time_reversal_correctness():
    """TimeReversal reverses order after seam."""
    atom = TimeReversalAtom()
    signal = np.arange(100)
    seam = 50

    reversed_sig = atom.apply(signal, seam)

    # Before seam: unchanged
    np.testing.assert_array_equal(reversed_sig[:seam], signal[:seam])
    # After seam: reversed
    np.testing.assert_array_equal(reversed_sig[seam:], signal[seam:][::-1])


def test_variance_scale_inverse():
    """VarianceScale inverse recovers original."""
    atom = VarianceScaleAtom()
    signal = np.concatenate([np.ones(50), 2 * np.ones(50)])
    seam = 50

    atom.fit_params(signal, seam)
    scaled = atom.apply(signal, seam)
    recovered = atom.inverse(scaled, seam)

    np.testing.assert_array_almost_equal(signal, recovered)


def test_variance_scale_homogenization():
    """VarianceScale should homogenize variance across seam."""
    atom = VarianceScaleAtom()

    # Create signal with variance shift
    pre_seam = np.random.randn(100)
    post_seam = 3.0 * np.random.randn(100)
    signal = np.concatenate([pre_seam, post_seam])
    seam = 100

    # Fit and apply
    atom.fit_params(signal, seam)
    scaled = atom.apply(signal, seam)

    # Check that variances are more similar
    var_pre_original = np.var(signal[:seam])
    var_post_original = np.var(signal[seam:])
    ratio_original = max(var_pre_original, var_post_original) / min(
        var_pre_original, var_post_original
    )

    var_pre_scaled = np.var(scaled[:seam])
    var_post_scaled = np.var(scaled[seam:])
    ratio_scaled = max(var_pre_scaled, var_post_scaled) / min(
        var_pre_scaled, var_post_scaled
    )

    assert ratio_scaled < ratio_original  # Should be more homogeneous


def test_polynomial_detrend_inverse():
    """PolynomialDetrend inverse recovers original."""
    atom = PolynomialDetrendAtom(degree=1)
    signal = np.concatenate([np.zeros(50), np.linspace(0, 5, 50)])
    seam = 50

    atom.fit_params(signal, seam)
    detrended = atom.apply(signal, seam)
    recovered = atom.inverse(detrended, seam)

    np.testing.assert_array_almost_equal(signal, recovered, decimal=10)


def test_polynomial_detrend_mean_removal():
    """PolynomialDetrend should remove polynomial trend."""
    atom = PolynomialDetrendAtom(degree=1)

    # Create signal with linear trend after seam
    t = np.linspace(0, 1, 100)
    signal = np.concatenate([np.zeros(50), 5 * t[:50]])
    seam = 50

    atom.fit_params(signal, seam)
    detrended = atom.apply(signal, seam)

    # Post-seam should have lower mean after detrending
    mean_before = np.mean(signal[seam:])
    mean_after = np.mean(detrended[seam:])

    assert abs(mean_after) < abs(mean_before)


def test_composite_atom():
    """CompositeFlipAtom applies atoms sequentially."""
    sign_flip = SignFlipAtom()
    var_scale = VarianceScaleAtom()

    composite = CompositeFlipAtom([sign_flip, var_scale])

    signal = np.random.randn(100)
    seam = 50

    # Apply composite
    var_scale.fit_params(signal, seam)
    transformed = composite.apply(signal, seam)

    # Should be equivalent to applying sequentially
    intermediate = sign_flip.apply(signal, seam)
    var_scale.fit_params(intermediate, seam)
    expected = var_scale.apply(intermediate, seam)

    np.testing.assert_array_almost_equal(transformed, expected)


def test_composite_inverse():
    """CompositeFlipAtom inverse works correctly."""
    atoms = [SignFlipAtom(), TimeReversalAtom()]
    composite = CompositeFlipAtom(atoms)

    signal = np.arange(100)
    seam = 50

    transformed = composite.apply(signal, seam)
    recovered = composite.inverse(transformed, seam)

    np.testing.assert_array_almost_equal(signal, recovered)


def test_flip_atom_num_params():
    """Test parameter counting."""
    assert SignFlipAtom().num_params() == 0
    assert TimeReversalAtom().num_params() == 0
    assert VarianceScaleAtom().num_params() == 1
    assert PolynomialDetrendAtom(degree=2).num_params() == 3  # 3 coeffs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
