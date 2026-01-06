"""
Tests for MDL calculations.
"""

import pytest
import numpy as np
from seamaware.core.mdl import (
    compute_mdl,
    delta_mdl,
    compute_bic,
    compute_aic,
    residual_variance,
    effective_snr,
)


def test_mdl_perfect_fit():
    """MDL for perfect fit should be close to parameter cost only."""
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    prediction = data.copy()  # Perfect fit

    mdl = compute_mdl(data, prediction, num_params=2)

    # With perfect fit, NLL ≈ 0, so MDL ≈ parameter cost
    param_cost = (2 / 2) * np.log2(100)
    assert mdl > 0
    assert mdl < param_cost * 2  # Should be dominated by param cost


def test_mdl_monotonicity():
    """Better fit should give lower MDL (for same num_params)."""
    data = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Good fit
    pred_good = data + 0.1 * np.random.randn(100)
    mdl_good = compute_mdl(data, pred_good, num_params=2)

    # Bad fit
    pred_bad = data + 1.0 * np.random.randn(100)
    mdl_bad = compute_mdl(data, pred_bad, num_params=2)

    assert mdl_good < mdl_bad


def test_mdl_parameter_penalty():
    """More parameters should increase MDL (if fit doesn't improve)."""
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    prediction = data + 0.1 * np.random.randn(100)

    mdl_2 = compute_mdl(data, prediction, num_params=2)
    mdl_10 = compute_mdl(data, prediction, num_params=10)

    assert mdl_10 > mdl_2  # More params without better fit → higher MDL


def test_delta_mdl():
    """Test ΔMDL computation."""
    mdl_baseline = 1000.0
    mdl_seam = 850.0

    delta = delta_mdl(mdl_baseline, mdl_seam)

    assert delta < 0  # Seam improves model
    assert delta == -150.0


def test_residual_variance():
    """Test residual variance calculation."""
    data = np.array([1, 2, 3, 4, 5])
    prediction = np.array([1, 2, 3, 4, 5])  # Perfect

    var = residual_variance(data, prediction)
    assert var == 0.0

    # Imperfect fit
    prediction2 = np.array([1, 2, 3.5, 4, 5])
    var2 = residual_variance(data, prediction2)
    assert var2 > 0


def test_effective_snr():
    """Test effective SNR calculation."""
    data = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Baseline: noisy fit
    pred_baseline = data + 0.5 * np.random.randn(100)

    # Seam-aware: better fit
    pred_seam = data + 0.1 * np.random.randn(100)

    snr = effective_snr(data, pred_baseline, pred_seam)

    assert snr > 0  # Should show improvement
    # Exact value depends on random noise, just check it's positive


def test_mdl_bic_aic_consistency():
    """Test that MDL, BIC, AIC all penalize complexity."""
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    prediction = data + 0.1 * np.random.randn(100)

    mdl = compute_mdl(data, prediction, num_params=2)
    bic = compute_bic(data, prediction, num_params=2)
    aic = compute_aic(data, prediction, num_params=2)

    # All should be positive
    assert mdl > 0
    assert bic > 0
    assert aic > 0

    # AIC typically penalizes less than BIC/MDL
    assert aic < bic


def test_mdl_input_validation():
    """Test input validation for MDL."""
    data = np.array([1, 2, 3])
    prediction = np.array([1, 2, 3, 4])  # Wrong length

    with pytest.raises(ValueError, match="length"):
        compute_mdl(data, prediction, num_params=2)

    # NaN values
    data_nan = np.array([1, 2, np.nan])
    with pytest.raises(ValueError, match="NaN"):
        compute_mdl(data_nan, prediction[:3], num_params=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
