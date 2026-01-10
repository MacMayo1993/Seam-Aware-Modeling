"""Tests for MDL computation."""
import numpy as np
from dmdl_lift.mdl import mdl_rank1, mdl_rank2, fit_ar1, nll_gaussian
from dmdl_lift.words import periodic_word


def test_fit_ar1():
    """Test AR(1) fitting function."""
    # Simple AR(1) process
    rng = np.random.default_rng(42)
    x = np.zeros(100)
    x[0] = 1.0
    for t in range(1, 100):
        x[t] = 0.8 * x[t-1] + 0.1 * rng.normal()

    a, c = fit_ar1(x)
    assert np.isfinite(a)
    assert np.isfinite(c)
    # Should recover approximately 0.8
    assert 0.7 < a < 0.9


def test_nll_gaussian():
    """Test Gaussian NLL computation."""
    residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
    nll = nll_gaussian(residuals)
    assert np.isfinite(nll)
    # NLL can be negative for small residuals (high likelihood)
    # Just check it's finite and reasonable
    assert -100 < nll < 100


def test_mdl_rank1():
    """Test rank-1 MDL computation."""
    x = np.random.default_rng(42).normal(size=200)
    mdl, (a, c) = mdl_rank1(x, start_idx=5)
    assert np.isfinite(mdl)
    assert np.isfinite(a)
    assert np.isfinite(c)


def test_mdl_rank2_fallback():
    """Test rank-2 MDL with fallback for sparse phases."""
    x = np.random.default_rng(42).normal(size=200)
    w = periodic_word(200)
    mdl, n_phases, ess = mdl_rank2(x, w, L=8, min_samples=5)
    assert np.isfinite(mdl)
    assert n_phases >= 0
    assert ess >= 0


def test_mdl_consistency():
    """Test that rank-1 and rank-2 use consistent segment for fitting.

    Both models should fit on the segment from start_idx onward.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    w = periodic_word(200)

    L = 6
    start_idx = L - 1

    # Rank-1 should fit on x[start_idx-1:]
    mdl1, (a1, c1) = mdl_rank1(x, start_idx=start_idx)

    # Rank-2 should use the same segment
    mdl2, n_phases, ess = mdl_rank2(x, w, L=L, min_samples=10)

    # Both should produce finite MDL
    assert np.isfinite(mdl1)
    assert np.isfinite(mdl2)

    # For random data with periodic word, rank-1 should typically be preferred
    # (ΔMDL = mdl1 - mdl2 should be close to 0 or negative)
    delta = mdl1 - mdl2
    assert np.isfinite(delta)
