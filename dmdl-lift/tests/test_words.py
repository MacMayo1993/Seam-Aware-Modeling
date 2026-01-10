"""Tests for word generation and complexity analysis."""
import numpy as np
from dmdl_lift.words import fibonacci_word, periodic_word, complexity_profile, kmer_coverage


def test_periodic_word():
    """Test periodic word generation."""
    w = periodic_word(10)
    assert ''.join(w) == "ABABABABAB"


def test_fibonacci_word():
    """Test Fibonacci word generation via substitution system."""
    # Check short prefix matches expected substitution
    w = fibonacci_word(5)
    assert ''.join(w) == "ABAAB"

    # Check substitution property on longer prefix
    w13 = fibonacci_word(13)
    expected = "ABAABABAABAAB"  # A→AB, B→A applied iteratively
    assert ''.join(w13) == expected


def test_complexity_profile_periodic():
    """Test k-mer complexity for periodic ABAB... word.

    For periodic ABAB..., expect exactly 2 distinct k-mers for even L.
    (For odd L, depends on alignment; we test even L here.)
    """
    w_per = periodic_word(100)
    comp_per = complexity_profile(w_per, Ls=(2, 4, 6, 8, 10, 12))

    # All even lengths should give exactly 2 distinct k-mers
    for L in [2, 4, 6, 8, 10, 12]:
        assert comp_per[L] == 2, f"Expected 2 distinct k-mers at L={L}, got {comp_per[L]}"


def test_complexity_profile_fibonacci():
    """Test k-mer complexity for Fibonacci word.

    Fibonacci word is Sturmian: exactly L+1 distinct k-mers of length L.
    """
    w_fib = fibonacci_word(1000)
    comp_fib = complexity_profile(w_fib, Ls=(2, 4, 6, 8, 10, 12))

    # Check Sturmian property: L+1 k-mers
    assert comp_fib[2] == 3
    assert comp_fib[4] == 5
    assert comp_fib[6] == 7
    assert comp_fib[8] == 9
    assert comp_fib[10] == 11
    assert comp_fib[12] == 13


def test_kmer_coverage():
    """Test kmer_coverage function directly."""
    w_per = periodic_word(50)
    w_fib = fibonacci_word(50)

    # Periodic at L=6 should have 2 k-mers
    assert kmer_coverage(w_per, L=6) == 2

    # Fibonacci at L=6 should have 7 k-mers (L+1)
    assert kmer_coverage(w_fib, L=6) == 7
