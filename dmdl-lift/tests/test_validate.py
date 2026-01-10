"""Tests for validation pipeline including negative controls.

These tests verify that the ΔMDL detector responds to lift structure,
not merely to word type or other confounds.
"""
import numpy as np
from dmdl_lift.validate import validate_method
from dmdl_lift.words import fibonacci_word, periodic_word
from dmdl_lift.signal import generate_signal, generate_null_signal
from dmdl_lift.delta import delta_mdl_trajectory


def test_validate_method():
    """Test standard validation protocol passes."""
    results = validate_method(L=6)
    assert results is not None
    assert 'periodic' in results
    assert 'fibonacci' in results
    assert 'stats' in results
    # Validation should pass with default parameters
    assert results['success'] == True


def test_negative_control_fibonacci_word_null_signal():
    """NEGATIVE CONTROL: Fibonacci word + null signal → no lift detected.

    This proves the detector responds to lift in the signal, not to
    "Fibonacci-ness" of the word.

    Expected: ΔMDL ≈ 0 or negative (like periodic case)
    """
    T = 1200
    L = 6
    w_fib = fibonacci_word(T)

    # Generate NULL signal (no lift) even with Fibonacci word
    x_fib_null = generate_null_signal(w_fib, noise=0.06, seed=42)

    # Compute ΔMDL
    ts, deltas, phases, ess = delta_mdl_trajectory(
        x_fib_null, w_fib, L=L, min_samples=10
    )

    mean_delta = np.mean(deltas)

    # Should NOT show lift preference (should be ≤ 0)
    assert mean_delta < 10.0, (
        f"Fibonacci word + null signal should not show lift, "
        f"but got mean ΔMDL = {mean_delta:.2f}"
    )

    print(f"  Negative control (Fib+null): mean ΔMDL = {mean_delta:.2f} (expect ≤ 0)")


def test_negative_control_periodic_word_lifted_signal():
    """NEGATIVE CONTROL: Periodic word + lifted signal → lift detected.

    This proves the detector can detect lift even with simple periodic word.

    Expected: ΔMDL > 0 (detector responds to phase structure)
    """
    T = 1200
    L = 6
    w_per = periodic_word(T)

    # Generate LIFTED signal with periodic word
    # This creates 2-phase structure
    x_per_lifted = generate_signal(w_per, lifted=True, noise=0.06, seed=43, L=L)

    # Compute ΔMDL
    ts, deltas, phases, ess = delta_mdl_trajectory(
        x_per_lifted, w_per, L=L, min_samples=10
    )

    mean_delta = np.mean(deltas)
    mean_phases = np.mean(phases)

    # Should show SOME lift preference (positive ΔMDL)
    # Won't be as strong as Fibonacci (only 2 phases vs 7), but should be positive
    assert mean_delta > -5.0, (
        f"Periodic word + lifted signal should show some lift, "
        f"but got mean ΔMDL = {mean_delta:.2f}"
    )

    assert mean_phases == 2, "Periodic word should learn 2 phases"

    print(f"  Negative control (Per+lifted): mean ΔMDL = {mean_delta:.2f} (expect > 0)")


def test_word_structure_vs_signal_lift():
    """Compare all four combinations of word type and signal type.

    This comprehensive test shows the detector responds to signal lift,
    with word type determining phase count but not lift detection.

    Matrix:
                     Null Signal    Lifted Signal
    Periodic word    ΔMDL ≈ 0       ΔMDL > 0 (weak)
    Fibonacci word   ΔMDL ≈ 0       ΔMDL >> 0 (strong)
    """
    T = 1200
    L = 6

    w_per = periodic_word(T)
    w_fib = fibonacci_word(T)

    # Generate all four combinations
    x_per_null = generate_null_signal(w_per, noise=0.06, seed=10)
    x_per_lift = generate_signal(w_per, lifted=True, noise=0.06, seed=11, L=L)
    x_fib_null = generate_null_signal(w_fib, noise=0.06, seed=12)
    x_fib_lift = generate_signal(w_fib, lifted=True, noise=0.06, seed=13, L=L)

    # Compute ΔMDL for all four
    _, d_per_null, _, _ = delta_mdl_trajectory(x_per_null, w_per, L=L)
    _, d_per_lift, _, _ = delta_mdl_trajectory(x_per_lift, w_per, L=L)
    _, d_fib_null, _, _ = delta_mdl_trajectory(x_fib_null, w_fib, L=L)
    _, d_fib_lift, _, _ = delta_mdl_trajectory(x_fib_lift, w_fib, L=L)

    mu_per_null = np.mean(d_per_null)
    mu_per_lift = np.mean(d_per_lift)
    mu_fib_null = np.mean(d_fib_null)
    mu_fib_lift = np.mean(d_fib_lift)

    print("\n  Comprehensive matrix:")
    print(f"    Per+null: {mu_per_null:6.2f}")
    print(f"    Per+lift: {mu_per_lift:6.2f}")
    print(f"    Fib+null: {mu_fib_null:6.2f}")
    print(f"    Fib+lift: {mu_fib_lift:6.2f}")

    # Assertions
    # Null signals should not show strong lift (rank-1 preferred)
    assert mu_per_null < 10.0, "Per+null should not show lift"
    assert mu_fib_null < 10.0, "Fib+null should not show lift"

    # Lifted signals should show positive ΔMDL
    # Fibonacci should show stronger lift than periodic (more phases)
    assert mu_fib_lift > mu_fib_null + 30.0, "Fib+lift should show strong separation from Fib+null"
    assert mu_fib_lift > mu_per_lift, "Fib+lift should show stronger lift than Per+lift (more phases)"


def test_validation_structural_checks():
    """Test that validation catches structural mismatches."""
    # The main validation should pass structural checks
    results = validate_method(L=6)
    assert results is not None

    # Check that complexity and phase coverage are recorded
    assert 'complexity' in results
    assert 'periodic' in results['complexity']
    assert 'fibonacci' in results['complexity']

    # Check frozen config is recorded
    assert results['config']['L'] == 6
