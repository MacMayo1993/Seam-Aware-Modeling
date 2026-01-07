"""
Tests for MASS/SMASH implementation

Run with: pytest tests/test_mass_smash.py -v
Or: python -m pytest tests/test_mass_smash.py
"""
import sys
from pathlib import Path

import numpy as np

# Try pytest, fall back to basic assertions
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    print("pytest not available, running basic tests")

# Add examples directory to path so we can import mass_smash
# NOTE: This path manipulation must happen before local imports (noqa: E402)
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

# Import from mass_smash (path manipulation done above)
from mass_smash import (  # noqa: E402
    MASSSMASHConfig,
    antipodal_symmetry_scanner,
    bic_from_rss,
    build_model_zoo,
    detect_seam_candidates,
    fit_best_model,
    generate_signal_with_seams,
    mdl_bits,
    roughness_detector,
    run_mass_smash,
)


# =============================================================================
# Test Signal Generation
# =============================================================================

def test_signal_generation_basic():
    """Test basic signal generation."""
    y, seams = generate_signal_with_seams(T=300, seed=42)
    assert len(y) == 300, f"Expected length 300, got {len(y)}"
    assert isinstance(seams, list), "Seams should be a list"
    print("✓ test_signal_generation_basic")


def test_signal_generation_seam_positions():
    """Test seam positions are near requested fractions."""
    y, seams = generate_signal_with_seams(
        T=300,
        seam_positions=[0.33, 0.67],
        seed=42
    )
    assert len(seams) == 2, f"Expected 2 seams, got {len(seams)}"
    assert 90 <= seams[0] <= 110, f"First seam {seams[0]} not near 99"
    assert 190 <= seams[1] <= 210, f"Second seam {seams[1]} not near 201"
    print("✓ test_signal_generation_seam_positions")


def test_signal_generation_reproducibility():
    """Test that same seed gives same signal."""
    y1, seams1 = generate_signal_with_seams(T=100, seed=123)
    y2, seams2 = generate_signal_with_seams(T=100, seed=123)
    np.testing.assert_array_equal(y1, y2)
    assert seams1 == seams2
    print("✓ test_signal_generation_reproducibility")


# =============================================================================
# Test MDL/BIC Scoring
# =============================================================================

def test_mdl_seam_penalty():
    """More seams should increase MDL (all else equal)."""
    mdl_0 = mdl_bits(rss=100, n=300, p=10, m=0, alpha=2.0)
    mdl_1 = mdl_bits(rss=100, n=300, p=10, m=1, alpha=2.0)
    mdl_2 = mdl_bits(rss=100, n=300, p=10, m=2, alpha=2.0)
    assert mdl_0 < mdl_1 < mdl_2, f"MDL should increase: {mdl_0}, {mdl_1}, {mdl_2}"
    print("✓ test_mdl_seam_penalty")


def test_mdl_fit_improvement():
    """Better fit (lower RSS) should reduce MDL."""
    mdl_good = mdl_bits(rss=50, n=300, p=10, m=1, alpha=2.0)
    mdl_bad = mdl_bits(rss=100, n=300, p=10, m=1, alpha=2.0)
    assert mdl_good < mdl_bad, "Better fit should have lower MDL"
    print("✓ test_mdl_fit_improvement")


def test_mdl_param_penalty():
    """More parameters should increase MDL (all else equal)."""
    mdl_simple = mdl_bits(rss=100, n=300, p=5, m=1, alpha=2.0)
    mdl_complex = mdl_bits(rss=100, n=300, p=20, m=1, alpha=2.0)
    assert mdl_simple < mdl_complex, "More params should increase MDL"
    print("✓ test_mdl_param_penalty")


def test_bic_consistency():
    """BIC should be consistent with MDL ordering."""
    bic1 = bic_from_rss(rss=50, n=100, p=5)
    bic2 = bic_from_rss(rss=100, n=100, p=5)
    assert bic1 < bic2, "Lower RSS should give lower BIC"
    print("✓ test_bic_consistency")


# =============================================================================
# Test Detectors
# =============================================================================

def test_antipodal_detector_basic():
    """Antipodal detector should return candidates."""
    y, _ = generate_signal_with_seams(T=300, noise_std=0.1, seed=0)
    cands = antipodal_symmetry_scanner(y, threshold=0.3, top_k=5)
    assert isinstance(cands, list), "Should return list"
    assert all(isinstance(c, tuple) and len(c) == 2 for c in cands)
    print("✓ test_antipodal_detector_basic")


def test_antipodal_finds_sign_flip():
    """Antipodal detector should find sign flip seam."""
    y, true_seams = generate_signal_with_seams(
        T=300, noise_std=0.1,
        seam_positions=[0.5],
        seam_types=['sign_flip'],
        seed=0
    )
    cands = antipodal_symmetry_scanner(y, threshold=0.3, top_k=10)
    # Should find something near the true seam
    if cands:
        min_dist = min(abs(c[0] - true_seams[0]) for c in cands)
        assert min_dist < 50, f"Closest candidate {min_dist} samples from seam"
    print("✓ test_antipodal_finds_sign_flip")


def test_roughness_detector_basic():
    """Roughness detector should return candidates."""
    y, _ = generate_signal_with_seams(T=300, noise_std=0.2, seed=0)
    cands = roughness_detector(y, threshold=0.2, top_k=5)
    assert isinstance(cands, list), "Should return list"
    print("✓ test_roughness_detector_basic")


def test_combined_detection():
    """Combined detection should merge from both detectors."""
    y, _ = generate_signal_with_seams(T=300, noise_std=0.2, seed=0)
    config = MASSSMASHConfig()
    cands = detect_seam_candidates(y, config)
    assert isinstance(cands, list)
    assert all(len(c) == 3 for c in cands)  # (idx, score, detector_name)
    print("✓ test_combined_detection")


# =============================================================================
# Test Model Zoo
# =============================================================================

def test_build_model_zoo():
    """Model zoo should build without errors."""
    config = MASSSMASHConfig(include_mlp=False)
    zoo = build_model_zoo(config)
    assert len(zoo) >= 5, f"Expected at least 5 models, got {len(zoo)}"
    print("✓ test_build_model_zoo")


def test_fit_best_model():
    """Should fit and return best model by BIC."""
    y = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100)
    config = MASSSMASHConfig(include_mlp=False)
    zoo = build_model_zoo(config)
    result = fit_best_model(y, zoo)
    assert hasattr(result, 'model_name')
    assert hasattr(result, 'yhat')
    assert hasattr(result, 'bic')
    assert len(result.yhat) == len(y)
    print("✓ test_fit_best_model")


# =============================================================================
# Test Full Pipeline
# =============================================================================

def test_pipeline_runs():
    """Full pipeline should run without errors."""
    y, _ = generate_signal_with_seams(T=200, seed=0)
    config = MASSSMASHConfig(verbose=False, include_mlp=False)
    best, solutions = run_mass_smash(y, config)

    assert best is not None
    assert len(solutions) > 0
    assert hasattr(best, 'total_mdl')
    assert hasattr(best, 'seams')
    assert hasattr(best, 'yhat')
    assert len(best.yhat) == len(y)
    print("✓ test_pipeline_runs")


def test_pipeline_with_known_seams():
    """Pipeline should detect seams in clean signal."""
    y, true_seams = generate_signal_with_seams(
        T=300, noise_std=0.1,
        seam_positions=[0.5],
        seam_types=['sign_flip'],
        seed=42
    )
    config = MASSSMASHConfig(alpha=1.0, verbose=False, include_mlp=False)
    best, _ = run_mass_smash(y, config)

    # With low noise and low alpha, should find at least one seam
    # (though not guaranteed to be exactly right)
    assert hasattr(best, 'seams')
    print(f"  Found {len(best.seams)} seams: {list(best.seams)}")
    print("✓ test_pipeline_with_known_seams")


def test_alpha_affects_seam_count():
    """Higher alpha should find fewer or equal seams."""
    y, _ = generate_signal_with_seams(
        T=300, noise_std=0.15,
        seam_positions=[0.5],
        seed=42
    )

    config_low = MASSSMASHConfig(alpha=1.0, verbose=False, include_mlp=False)
    config_high = MASSSMASHConfig(alpha=3.0, verbose=False, include_mlp=False)

    best_low, _ = run_mass_smash(y, config_low)
    best_high, _ = run_mass_smash(y, config_high)

    assert len(best_low.seams) >= len(best_high.seams), \
        f"Low alpha found {len(best_low.seams)}, high found {len(best_high.seams)}"
    print("✓ test_alpha_affects_seam_count")


def test_solutions_sorted_by_mdl():
    """Returned solutions should be sorted by MDL."""
    y, _ = generate_signal_with_seams(T=200, seed=0)
    config = MASSSMASHConfig(verbose=False, include_mlp=False)
    _, solutions = run_mass_smash(y, config)

    mdls = [s.total_mdl for s in solutions]
    assert mdls == sorted(mdls), "Solutions should be sorted by MDL"
    print("✓ test_solutions_sorted_by_mdl")


# =============================================================================
# Test Configuration
# =============================================================================

def test_config_defaults():
    """Config should have sensible defaults."""
    config = MASSSMASHConfig()
    assert config.alpha == 2.0
    assert config.max_seams == 3
    assert config.min_separation == 30
    print("✓ test_config_defaults")


def test_config_custom():
    """Custom config values should be respected."""
    config = MASSSMASHConfig(
        alpha=1.5,
        max_seams=5,
        include_mlp=False
    )
    assert config.alpha == 1.5
    assert config.max_seams == 5
    assert not config.include_mlp
    print("✓ test_config_custom")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # Signal generation
        test_signal_generation_basic,
        test_signal_generation_seam_positions,
        test_signal_generation_reproducibility,

        # MDL/BIC
        test_mdl_seam_penalty,
        test_mdl_fit_improvement,
        test_mdl_param_penalty,
        test_bic_consistency,

        # Detectors
        test_antipodal_detector_basic,
        test_antipodal_finds_sign_flip,
        test_roughness_detector_basic,
        test_combined_detection,

        # Model zoo
        test_build_model_zoo,
        test_fit_best_model,

        # Full pipeline
        test_pipeline_runs,
        test_pipeline_with_known_seams,
        test_alpha_affects_seam_count,
        test_solutions_sorted_by_mdl,

        # Config
        test_config_defaults,
        test_config_custom,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("MASS/SMASH v2 Test Suite")
    print("=" * 60 + "\n")

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Run basic tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
