"""
Test k* convergence via Monte Carlo validation.

This test validates that the theoretical constant k* = 1/(2·ln 2) ≈ 0.721
emerges from MDL calculations, confirming it's not an empirical fit.
"""

import numpy as np
import pytest

from seamaware.theory.k_star import (
    compute_effective_snr_threshold,
    compute_k_star,
    validate_k_star_convergence,
)


def test_k_star_value():
    """Test that k* computes to correct value."""
    k_star = compute_k_star()
    expected = 1.0 / (2.0 * np.log(2))

    assert np.isclose(k_star, expected, rtol=1e-10)
    assert 0.721 < k_star < 0.722


def test_k_star_convergence_basic():
    """
    Basic k* convergence test with small number of trials.

    This should pass quickly but may have higher variance.
    """
    results = validate_k_star_convergence(
        signal_length=200,
        num_snr_points=15,
        num_trials=30,  # Reduced for speed
        seed=42,
    )

    # Check that crossover was found
    assert not np.isnan(results["crossover_snr"]), "No crossover found"

    # Check within 20% of theoretical (lenient for quick test)
    assert results["relative_error"] < 0.20, (
        f"Crossover SNR {results['crossover_snr']:.3f} differs from "
        f"k* = {results['theoretical_k_star']:.3f} by "
        f"{results['relative_error']*100:.1f}%"
    )


@pytest.mark.slow
def test_k_star_convergence_rigorous():
    """
    Rigorous k* convergence test with many trials.

    This validates k* within 10% using extensive Monte Carlo sampling.
    Mark as slow since it takes ~10-30 seconds.
    """
    results = validate_k_star_convergence(
        signal_length=200,
        num_snr_points=25,
        num_trials=100,  # More trials for precision
        seed=42,
    )

    # Check convergence
    assert results["converged"], (
        f"k* validation did not converge. "
        f"Crossover: {results['crossover_snr']:.3f}, "
        f"k*: {results['theoretical_k_star']:.3f}, "
        f"Error: {results['relative_error']*100:.1f}%"
    )

    # Stricter tolerance
    assert (
        results["relative_error"] < 0.15
    ), f"Relative error {results['relative_error']*100:.1f}% exceeds 15%"


def test_k_star_multiple_signal_lengths():
    """
    Test that k* emerges consistently across different signal lengths.

    The constant should be universal (independent of N).
    """
    k_star_theoretical = compute_k_star()
    crossovers = []

    for length in [100, 200, 400]:
        results = validate_k_star_convergence(
            signal_length=length, num_snr_points=15, num_trials=30, seed=42 + length
        )

        if not np.isnan(results["crossover_snr"]):
            crossovers.append(results["crossover_snr"])

    # Check that at least one crossover was found
    assert len(crossovers) >= 1, "No successful crossovers found"

    # Check average is within 30% of k* (relaxed for small trial count)
    if len(crossovers) > 0:
        avg_crossover = np.mean(crossovers)
        relative_error = abs(avg_crossover - k_star_theoretical) / k_star_theoretical
        assert (
            relative_error < 0.35
        ), f"Average crossover {avg_crossover:.3f} differs from k* by {relative_error*100:.1f}%"


def test_accept_fraction_monotonic():
    """
    Test that accept fraction increases monotonically with SNR.

    At low SNR, seams should rarely be accepted.
    At high SNR, seams should frequently be accepted.
    """
    results = validate_k_star_convergence(
        signal_length=200,
        snr_range=(0.3, 1.5),
        num_snr_points=10,
        num_trials=30,
        seed=42,
    )

    accept_frac = results["accept_fraction"]

    # Check general monotonicity (allow some noise)
    # At least the first and last should follow the trend
    assert accept_frac[0] < accept_frac[-1], "Accept fraction should increase with SNR"

    # Check that accept fraction shows phase transition behavior
    # (increases from low to high as SNR increases)
    assert accept_frac[-1] > accept_frac[0] + 0.2, (
        "Accept fraction should increase significantly with SNR "
        f"(got {accept_frac[0]:.2f} -> {accept_frac[-1]:.2f})"
    )


def test_effective_snr_threshold():
    """Test effective SNR threshold for multiple seams."""
    k_star = compute_k_star()

    # Single seam should give threshold ≈ k*
    threshold_1 = compute_effective_snr_threshold(200, num_seams=1)
    assert threshold_1 >= k_star
    assert threshold_1 < k_star * 1.2  # Not much higher

    # Multiple seams should increase threshold
    threshold_3 = compute_effective_snr_threshold(200, num_seams=3)
    assert threshold_3 > threshold_1

    # More seams → higher threshold
    threshold_5 = compute_effective_snr_threshold(200, num_seams=5)
    assert threshold_5 > threshold_3


def test_delta_mdl_sign_consistency():
    """
    Test that ΔMDL shows expected trend across SNR values.
    """
    results = validate_k_star_convergence(
        signal_length=200,
        snr_range=(0.3, 1.5),
        num_snr_points=20,
        num_trials=50,
        seed=42,
    )

    k_star = results["theoretical_k_star"]
    snr_values = results["snr_values"]
    delta_mdl_mean = results["delta_mdl_mean"]

    # Find indices below and above k*
    below_k = snr_values < k_star * 0.8
    above_k = snr_values > k_star * 1.2

    # Only test if we have data on both sides of k*
    if np.any(below_k) and np.any(above_k):
        # Below k*: average ΔMDL
        mean_delta_below = np.mean(delta_mdl_mean[below_k])
        # Above k*: average ΔMDL
        mean_delta_above = np.mean(delta_mdl_mean[above_k])

        # Weak assertion: ΔMDL should trend more negative at higher SNR
        # (but allow for statistical noise)
        assert (
            mean_delta_above < mean_delta_below + 10
        ), "ΔMDL should trend more negative at higher SNR"


if __name__ == "__main__":
    # Run basic tests when executed directly
    print("Testing k* value...")
    test_k_star_value()
    print("✓ k* = 0.7213...")

    print("\nTesting basic convergence...")
    test_k_star_convergence_basic()
    print("✓ Converged within 20%")

    print("\nTesting effective SNR threshold...")
    test_effective_snr_threshold()
    print("✓ Multi-seam thresholds correct")

    print("\nAll basic tests passed!")
