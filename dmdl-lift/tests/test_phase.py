"""Tests for phase computation and k-mer hashing."""
from dmdl_lift.phase import phase_state_coverage, expected_phases, kmer_id
from dmdl_lift.words import periodic_word, fibonacci_word


def test_phase_state_coverage_periodic_L8():
    """Test phase coverage for periodic word at L=8."""
    w_per = periodic_word(1200)
    n_per, _ = phase_state_coverage(w_per, L=8)
    assert n_per == 2  # Periodic always has 2 k-mers


def test_phase_state_coverage_fibonacci_L8():
    """Test phase coverage for Fibonacci word at L=8."""
    w_fib = fibonacci_word(1200)
    n_fib, _ = phase_state_coverage(w_fib, L=8)
    assert n_fib == 9  # Fibonacci has L+1 = 9 k-mers


def test_phase_state_coverage_frozen_L6():
    """Test phase coverage at frozen config L=6.

    This is the operational configuration, so it's critical to validate.
    """
    w_per = periodic_word(1200)
    w_fib = fibonacci_word(1200)

    n_per, _ = phase_state_coverage(w_per, L=6)
    n_fib, _ = phase_state_coverage(w_fib, L=6)

    assert n_per == 2, f"Expected 2 phases for periodic at L=6, got {n_per}"
    assert n_fib == 7, f"Expected 7 phases for Fibonacci at L=6, got {n_fib}"


def test_expected_phases():
    """Test expected_phases helper function."""
    assert expected_phases('periodic', 8) == 2
    assert expected_phases('fibonacci', 8) == 9
    assert expected_phases('fibonacci', 6) == 7


def test_kmer_id():
    """Test k-mer hashing function."""
    word = periodic_word(20)  # ABABAB...

    # At t=5 (L=3), k-mer is word[3:6] = word[t-L+1:t+1]
    # word[3:6] = "BAB" (positions 3,4,5 in ABABAB...)
    # Binary: A=0, B=1 → "BAB" = 101 = 5
    kid = kmer_id(word, t=5, L=3)
    assert kid == 5

    # Before L-1, should return None
    kid = kmer_id(word, t=1, L=3)
    assert kid is None
