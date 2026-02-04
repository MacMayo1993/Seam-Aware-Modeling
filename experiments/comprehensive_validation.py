"""
Comprehensive Experimental Validation for SeamAware Hypotheses

This script runs rigorous tests to validate or break the following hypotheses:
H1: MDL reduction on seam data (10-63% improvement)
H2: Universal phase boundary at k* ≈ 0.721
H3: Robust seam localization
H4: ℤ₂ involution property of flip atoms
H5: Universality of k* across signal lengths

Author: Experimental Validation Suite
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seamaware.core.flip_atoms import (
    CompositeFlipAtom,
    PolynomialDetrendAtom,
    SignFlipAtom,
    TimeReversalAtom,
    VarianceScaleAtom,
)
from seamaware.core.mdl import compute_mdl, delta_mdl, effective_snr
from seamaware.core.seam_detection import detect_seams_cusum, detect_seams_roughness
from seamaware.models.baselines import FourierBaseline, PolynomialBaseline
from seamaware.theory.k_star import compute_k_star, validate_k_star_convergence


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    hypothesis: str
    test_name: str
    passed: bool
    details: Dict[str, Any]
    execution_time: float


def run_hypothesis_1_mdl_reduction() -> List[ExperimentResult]:
    """
    H1: Seam-aware modeling achieves 10-63% MDL reduction on seam data
    """
    results = []

    # Test 1.1: Sign flip seam
    start = time.time()
    np.random.seed(42)

    mdl_improvements = []
    for trial in range(50):
        # Generate signal with sign flip
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        seam_loc = 100
        signal[seam_loc:] *= -1
        signal += np.random.normal(0, 0.1, len(signal))

        # Baseline: Fourier
        baseline = FourierBaseline(K=6)
        pred_baseline = baseline.fit_predict(signal)
        mdl_baseline = compute_mdl(signal, pred_baseline, baseline.num_params())

        # Seam-aware
        flip_atom = SignFlipAtom()
        flipped = flip_atom.apply(signal, seam_loc)
        pred_seam = baseline.fit_predict(flipped)
        mdl_seam_result = compute_mdl(flipped, pred_seam, baseline.num_params() + 1)
        mdl_seam = mdl_seam_result.total_bits + np.log2(200)  # Seam location cost

        improvement = (mdl_baseline.total_bits - mdl_seam) / mdl_baseline.total_bits * 100
        mdl_improvements.append(improvement)

    mean_improvement = np.mean(mdl_improvements)
    std_improvement = np.std(mdl_improvements)
    passed = 10 <= mean_improvement <= 70

    results.append(ExperimentResult(
        hypothesis="H1",
        test_name="MDL reduction on sign flip seam",
        passed=passed,
        details={
            "mean_improvement_percent": mean_improvement,
            "std_improvement_percent": std_improvement,
            "min_improvement": np.min(mdl_improvements),
            "max_improvement": np.max(mdl_improvements),
            "expected_range": "10-63%",
            "num_trials": 50
        },
        execution_time=time.time() - start
    ))

    # Test 1.2: Variance shift seam
    start = time.time()
    np.random.seed(43)

    mdl_improvements_var = []
    for trial in range(50):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        seam_loc = 100
        signal[:seam_loc] += np.random.normal(0, 0.1, seam_loc)
        signal[seam_loc:] += np.random.normal(0, 0.5, 200 - seam_loc)

        baseline = PolynomialBaseline(degree=3)
        pred_baseline = baseline.fit_predict(signal)
        mdl_baseline = compute_mdl(signal, pred_baseline, baseline.num_params())

        # Variance scaling
        var_atom = VarianceScaleAtom()
        scaled = var_atom.apply(signal, seam_loc)
        pred_seam = baseline.fit_predict(scaled)
        mdl_seam_result = compute_mdl(scaled, pred_seam, baseline.num_params() + 1)
        mdl_seam = mdl_seam_result.total_bits + np.log2(200)

        improvement = (mdl_baseline.total_bits - mdl_seam) / mdl_baseline.total_bits * 100
        mdl_improvements_var.append(improvement)

    mean_improvement_var = np.mean(mdl_improvements_var)
    passed_var = mean_improvement_var > 5  # More lenient for variance shifts

    results.append(ExperimentResult(
        hypothesis="H1",
        test_name="MDL reduction on variance shift seam",
        passed=passed_var,
        details={
            "mean_improvement_percent": mean_improvement_var,
            "std_improvement_percent": np.std(mdl_improvements_var),
            "num_trials": 50
        },
        execution_time=time.time() - start
    ))

    # Test 1.3: No seam baseline (should NOT improve)
    start = time.time()
    np.random.seed(44)

    spurious_changes = []
    for trial in range(30):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t) + np.random.normal(0, 0.1, 200)  # No seam

        baseline = FourierBaseline(K=6)
        pred_baseline = baseline.fit_predict(signal)
        mdl_baseline = compute_mdl(signal, pred_baseline, baseline.num_params())

        # Try spurious seam
        flip_atom = SignFlipAtom()
        flipped = flip_atom.apply(signal, 100)
        pred_seam = baseline.fit_predict(flipped)
        mdl_seam_result = compute_mdl(flipped, pred_seam, baseline.num_params() + 1)
        mdl_seam = mdl_seam_result.total_bits + np.log2(200)

        change = (mdl_baseline.total_bits - mdl_seam) / mdl_baseline.total_bits * 100
        spurious_changes.append(change)

    # Spurious seams should NOT consistently improve MDL
    fraction_improved = np.mean(np.array(spurious_changes) > 5)
    passed_spurious = fraction_improved < 0.2  # Less than 20% should show improvement

    results.append(ExperimentResult(
        hypothesis="H1",
        test_name="Spurious seam should not improve MDL",
        passed=passed_spurious,
        details={
            "mean_change_percent": np.mean(spurious_changes),
            "fraction_showing_improvement": fraction_improved,
            "expected": "< 20% should show > 5% improvement"
        },
        execution_time=time.time() - start
    ))

    return results


def run_hypothesis_2_k_star_validation() -> List[ExperimentResult]:
    """
    H2: Universal phase boundary at k* = 1/(2·ln 2) ≈ 0.721
    """
    results = []

    # Test 2.1: Basic k* value
    start = time.time()
    k_star_theoretical = compute_k_star()
    k_star_expected = 1 / (2 * np.log(2))
    passed_value = abs(k_star_theoretical - k_star_expected) < 1e-10

    results.append(ExperimentResult(
        hypothesis="H2",
        test_name="k* theoretical value",
        passed=passed_value,
        details={
            "computed_k_star": k_star_theoretical,
            "expected_k_star": k_star_expected,
            "difference": abs(k_star_theoretical - k_star_expected)
        },
        execution_time=time.time() - start
    ))

    # Test 2.2: Monte Carlo validation (50 trials)
    start = time.time()
    validation_50 = validate_k_star_convergence(
        signal_length=200,
        num_trials=50,
        num_snr_points=25,
        snr_range=(0.2, 1.5),
        seed=42
    )

    relative_error_50 = validation_50["relative_error"]
    passed_mc_50 = relative_error_50 < 0.20  # Within 20%

    results.append(ExperimentResult(
        hypothesis="H2",
        test_name="k* Monte Carlo validation (50 trials)",
        passed=passed_mc_50,
        details={
            "empirical_k_star": validation_50["crossover_snr"],
            "theoretical_k_star": validation_50["theoretical_k_star"],
            "relative_error": relative_error_50,
            "converged": validation_50["converged"],
            "num_trials": 50
        },
        execution_time=time.time() - start
    ))

    # Test 2.3: Monte Carlo validation (100 trials - more rigorous)
    start = time.time()
    validation_100 = validate_k_star_convergence(
        signal_length=200,
        num_trials=100,
        num_snr_points=30,
        snr_range=(0.2, 1.5),
        seed=43
    )

    relative_error_100 = validation_100["relative_error"]
    passed_mc_100 = relative_error_100 < 0.15  # Within 15%

    results.append(ExperimentResult(
        hypothesis="H2",
        test_name="k* Monte Carlo validation (100 trials)",
        passed=passed_mc_100,
        details={
            "empirical_k_star": validation_100["crossover_snr"],
            "theoretical_k_star": validation_100["theoretical_k_star"],
            "relative_error": relative_error_100,
            "accept_fraction_at_crossover": float(np.interp(
                validation_100["crossover_snr"],
                validation_100["snr_values"],
                validation_100["accept_fraction"]
            )),
            "num_trials": 100
        },
        execution_time=time.time() - start
    ))

    # Test 2.4: Phase transition behavior (accept fraction should be monotonic)
    start = time.time()
    snr_values = validation_100["snr_values"]
    accept_fractions = validation_100["accept_fraction"]

    # Check monotonicity with some tolerance
    diffs = np.diff(accept_fractions)
    monotonic_violations = np.sum(diffs < -0.15)  # Allow small decreases
    passed_monotonic = monotonic_violations <= 2  # At most 2 violations

    results.append(ExperimentResult(
        hypothesis="H2",
        test_name="Accept fraction monotonicity",
        passed=passed_monotonic,
        details={
            "monotonic_violations": int(monotonic_violations),
            "accept_fraction_range": [float(np.min(accept_fractions)), float(np.max(accept_fractions))],
            "expected": "Generally increasing with SNR"
        },
        execution_time=time.time() - start
    ))

    return results


def run_hypothesis_3_localization() -> List[ExperimentResult]:
    """
    H3: Seam localization within ±3 samples for SNR ≥ 6 dB
    """
    results = []

    # Test 3.1: High SNR localization accuracy
    start = time.time()
    np.random.seed(45)

    localization_errors = []
    true_seam = 100

    for trial in range(50):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t) * 2  # High amplitude
        signal[true_seam:] *= -1
        signal += np.random.normal(0, 0.2, 200)  # SNR ≈ 10

        detected = detect_seams_cusum(signal, threshold=5.0)

        if len(detected) > 0:
            best_detection = min(detected, key=lambda x: abs(x - true_seam))
            error = abs(best_detection - true_seam)
        else:
            error = 100  # Penalty for missing seam

        localization_errors.append(error)

    mean_error = np.mean(localization_errors)
    within_3_samples = np.mean(np.array(localization_errors) <= 3) * 100
    passed_high_snr = within_3_samples >= 70  # 70% should be within 3 samples

    results.append(ExperimentResult(
        hypothesis="H3",
        test_name="High SNR localization (SNR ≈ 10)",
        passed=passed_high_snr,
        details={
            "mean_localization_error": mean_error,
            "percent_within_3_samples": within_3_samples,
            "percent_within_5_samples": float(np.mean(np.array(localization_errors) <= 5) * 100),
            "num_trials": 50
        },
        execution_time=time.time() - start
    ))

    # Test 3.2: Low SNR localization (should be worse)
    start = time.time()
    np.random.seed(46)

    localization_errors_low = []
    for trial in range(50):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        signal[true_seam:] *= -1
        signal += np.random.normal(0, 1.0, 200)  # Low SNR ≈ 1

        detected = detect_seams_cusum(signal, threshold=5.0)

        if len(detected) > 0:
            best_detection = min(detected, key=lambda x: abs(x - true_seam))
            error = abs(best_detection - true_seam)
        else:
            error = 100

        localization_errors_low.append(error)

    mean_error_low = np.mean(localization_errors_low)
    # Low SNR should have worse localization
    passed_low_snr = mean_error_low > mean_error  # Should be worse than high SNR

    results.append(ExperimentResult(
        hypothesis="H3",
        test_name="Low SNR localization (SNR ≈ 1)",
        passed=passed_low_snr,
        details={
            "mean_localization_error": mean_error_low,
            "high_snr_error": mean_error,
            "expected": "Low SNR error > High SNR error"
        },
        execution_time=time.time() - start
    ))

    # Test 3.3: Detection rate vs SNR
    start = time.time()
    np.random.seed(47)

    snr_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
    detection_rates = []

    for snr in snr_levels:
        detected_count = 0
        for trial in range(30):
            t = np.linspace(0, 4 * np.pi, 200)
            signal = np.sin(t)
            signal[true_seam:] *= -1
            noise_std = 1.0 / snr
            signal += np.random.normal(0, noise_std, 200)

            detected = detect_seams_cusum(signal, threshold=5.0)
            if len(detected) > 0:
                best = min(detected, key=lambda x: abs(x - true_seam))
                if abs(best - true_seam) <= 10:
                    detected_count += 1

        detection_rates.append(detected_count / 30 * 100)

    # Detection rate should increase with SNR
    is_monotonic = all(detection_rates[i] <= detection_rates[i+1] + 10 for i in range(len(detection_rates)-1))
    passed_rate = detection_rates[-1] > 80  # High SNR should have >80% detection

    results.append(ExperimentResult(
        hypothesis="H3",
        test_name="Detection rate vs SNR",
        passed=passed_rate and is_monotonic,
        details={
            "snr_levels": snr_levels,
            "detection_rates_percent": detection_rates,
            "is_monotonic": is_monotonic
        },
        execution_time=time.time() - start
    ))

    return results


def run_hypothesis_4_involution() -> List[ExperimentResult]:
    """
    H4: Flip atoms satisfy F(F(x)) = x (involution property)
    """
    results = []

    # Test 4.1: Sign flip involution
    start = time.time()
    np.random.seed(48)

    max_errors = []
    for trial in range(20):
        signal = np.random.randn(200)
        seam = 100

        flip = SignFlipAtom()
        once = flip.apply(signal, seam)
        twice = flip.apply(once, seam)

        max_error = np.max(np.abs(twice - signal))
        max_errors.append(max_error)

    passed_sign = np.max(max_errors) < 1e-10

    results.append(ExperimentResult(
        hypothesis="H4",
        test_name="Sign flip involution",
        passed=passed_sign,
        details={
            "max_reconstruction_error": float(np.max(max_errors)),
            "expected": "< 1e-10"
        },
        execution_time=time.time() - start
    ))

    # Test 4.2: Time reversal involution
    start = time.time()

    max_errors_time = []
    for trial in range(20):
        signal = np.random.randn(200)
        seam = 100

        flip = TimeReversalAtom()
        once = flip.apply(signal, seam)
        twice = flip.apply(once, seam)

        max_error = np.max(np.abs(twice - signal))
        max_errors_time.append(max_error)

    passed_time = np.max(max_errors_time) < 1e-10

    results.append(ExperimentResult(
        hypothesis="H4",
        test_name="Time reversal involution",
        passed=passed_time,
        details={
            "max_reconstruction_error": float(np.max(max_errors_time)),
            "expected": "< 1e-10"
        },
        execution_time=time.time() - start
    ))

    # Test 4.3: Combined sign+time involution
    start = time.time()

    max_errors_combined = []
    for trial in range(20):
        signal = np.random.randn(200)
        seam = 100

        flip = CompositeFlipAtom([SignFlipAtom(), TimeReversalAtom()])
        once = flip.apply(signal, seam)
        twice = flip.apply(once, seam)

        max_error = np.max(np.abs(twice - signal))
        max_errors_combined.append(max_error)

    passed_combined = np.max(max_errors_combined) < 1e-10

    results.append(ExperimentResult(
        hypothesis="H4",
        test_name="Combined sign+time involution",
        passed=passed_combined,
        details={
            "max_reconstruction_error": float(np.max(max_errors_combined)),
            "expected": "< 1e-10"
        },
        execution_time=time.time() - start
    ))

    # Test 4.4: Variance scaling (NOT an involution, but invertible)
    start = time.time()

    signal = np.random.randn(200)
    signal[:100] *= 0.5
    signal[100:] *= 2.0  # Different variance
    seam = 100

    var_atom = VarianceScaleAtom()
    scaled = var_atom.apply(signal, seam)
    # Variance scaling is NOT an involution (applying twice doesn't restore original)
    twice = var_atom.apply(scaled, seam)

    # Check that it's NOT an involution
    is_not_involution = np.max(np.abs(twice - signal)) > 0.01

    results.append(ExperimentResult(
        hypothesis="H4",
        test_name="Variance scaling is NOT an involution",
        passed=is_not_involution,
        details={
            "reconstruction_error": float(np.max(np.abs(twice - signal))),
            "expected": "> 0.01 (confirming it's not an involution)"
        },
        execution_time=time.time() - start
    ))

    return results


def run_hypothesis_5_universality() -> List[ExperimentResult]:
    """
    H5: k* is universal across signal lengths
    """
    results = []

    # Test 5.1: k* across different signal lengths
    start = time.time()

    signal_lengths = [100, 200, 400]
    crossover_values = []

    for length in signal_lengths:
        validation = validate_k_star_convergence(
            signal_length=length,
            num_trials=40,
            num_snr_points=20,
            snr_range=(0.3, 1.3),
            seed=49 + length
        )
        crossover_values.append(validation["crossover_snr"])

    # Crossover values should be similar (within 30% of each other)
    cv_std = np.std(crossover_values)
    cv_mean = np.mean(crossover_values)
    cv_rel_std = cv_std / cv_mean if cv_mean > 0 else 0

    passed_universal = cv_rel_std < 0.30  # Within 30% relative std

    results.append(ExperimentResult(
        hypothesis="H5",
        test_name="k* universality across signal lengths",
        passed=passed_universal,
        details={
            "signal_lengths": signal_lengths,
            "crossover_values": [float(v) for v in crossover_values],
            "mean_crossover": float(cv_mean),
            "relative_std": float(cv_rel_std),
            "theoretical_k_star": float(compute_k_star())
        },
        execution_time=time.time() - start
    ))

    # Test 5.2: k* across different polynomial degrees
    start = time.time()

    poly_degrees = [1, 2, 3, 4]
    crossover_by_degree = []

    for degree in poly_degrees:
        np.random.seed(50 + degree)

        accept_counts = {snr: 0 for snr in np.linspace(0.3, 1.3, 15)}
        trials_per_snr = 30

        for snr in accept_counts.keys():
            for trial in range(trials_per_snr):
                t = np.linspace(0, 4 * np.pi, 200)
                signal = np.sin(t)
                signal[100:] *= -1
                noise = np.random.normal(0, 1.0 / snr, 200)
                noisy = signal + noise

                baseline = PolynomialBaseline(degree=degree)
                pred_base = baseline.fit_predict(noisy)
                mdl_base = compute_mdl(noisy, pred_base, baseline.num_params())

                flip = SignFlipAtom()
                flipped = flip.apply(noisy, 100)
                pred_flip = baseline.fit_predict(flipped)
                mdl_flip_result = compute_mdl(flipped, pred_flip, baseline.num_params() + 1)
                mdl_flip = mdl_flip_result.total_bits + np.log2(200)

                if mdl_flip < mdl_base.total_bits:
                    accept_counts[snr] += 1

        # Find crossover
        snr_list = list(accept_counts.keys())
        fractions = [accept_counts[s] / trials_per_snr for s in snr_list]

        # Linear interpolation
        for i in range(len(fractions) - 1):
            if fractions[i] <= 0.5 <= fractions[i+1]:
                crossover = snr_list[i] + (0.5 - fractions[i]) * (snr_list[i+1] - snr_list[i]) / (fractions[i+1] - fractions[i])
                crossover_by_degree.append(crossover)
                break
        else:
            crossover_by_degree.append(np.nan)

    valid_crossovers = [c for c in crossover_by_degree if not np.isnan(c)]
    if len(valid_crossovers) >= 2:
        degree_std = np.std(valid_crossovers)
        degree_mean = np.mean(valid_crossovers)
        passed_degree = degree_std / degree_mean < 0.30
    else:
        passed_degree = False

    results.append(ExperimentResult(
        hypothesis="H5",
        test_name="k* universality across polynomial degrees",
        passed=passed_degree,
        details={
            "polynomial_degrees": poly_degrees,
            "crossover_values": [float(v) if not np.isnan(v) else None for v in crossover_by_degree],
            "mean_crossover": float(np.nanmean(crossover_by_degree)),
            "std_crossover": float(np.nanstd(crossover_by_degree))
        },
        execution_time=time.time() - start
    ))

    return results


def run_stress_tests() -> List[ExperimentResult]:
    """
    Additional stress tests to try to break the hypotheses
    """
    results = []

    # Stress Test 1: Extreme noise levels
    start = time.time()
    np.random.seed(60)

    # At very low SNR, seam detection should fail gracefully
    extreme_noise_improvements = []
    for trial in range(20):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        signal[100:] *= -1
        signal += np.random.normal(0, 10.0, 200)  # SNR ≈ 0.1

        baseline = FourierBaseline(K=6)
        pred_base = baseline.fit_predict(signal)
        mdl_base = compute_mdl(signal, pred_base, baseline.num_params())

        flip = SignFlipAtom()
        flipped = flip.apply(signal, 100)
        pred_flip = baseline.fit_predict(flipped)
        mdl_flip_result = compute_mdl(flipped, pred_flip, baseline.num_params() + 1)
        mdl_flip = mdl_flip_result.total_bits + np.log2(200)

        improvement = (mdl_base.total_bits - mdl_flip) / mdl_base.total_bits * 100
        extreme_noise_improvements.append(improvement)

    # At extreme noise, should NOT consistently show improvement
    mean_improvement = np.mean(extreme_noise_improvements)
    passed_extreme = abs(mean_improvement) < 10  # Small or no improvement

    results.append(ExperimentResult(
        hypothesis="Stress",
        test_name="Extreme noise (SNR ≈ 0.1)",
        passed=passed_extreme,
        details={
            "mean_improvement_percent": float(mean_improvement),
            "expected": "Near zero improvement at extreme noise"
        },
        execution_time=time.time() - start
    ))

    # Stress Test 2: Very short signals
    start = time.time()
    np.random.seed(61)

    short_signal_results = []
    for length in [20, 30, 40, 50]:
        try:
            t = np.linspace(0, 2 * np.pi, length)
            signal = np.sin(t)
            signal[length//2:] *= -1
            signal += np.random.normal(0, 0.2, length)

            detected = detect_seams_cusum(signal, threshold=5.0)
            short_signal_results.append({"length": length, "detected": len(detected), "error": None})
        except Exception as e:
            short_signal_results.append({"length": length, "detected": -1, "error": str(e)})

    # Should handle gracefully (either detect or not, but no crashes)
    no_crashes = all(r["error"] is None for r in short_signal_results)

    results.append(ExperimentResult(
        hypothesis="Stress",
        test_name="Very short signals",
        passed=no_crashes,
        details={
            "results_by_length": short_signal_results
        },
        execution_time=time.time() - start
    ))

    # Stress Test 3: Multiple seams
    start = time.time()
    np.random.seed(62)

    t = np.linspace(0, 6 * np.pi, 300)
    signal = np.sin(t)
    # Three seams
    signal[75:150] *= -1
    signal[225:] *= -1
    signal += np.random.normal(0, 0.15, 300)

    detected = detect_seams_roughness(signal, window=15, threshold_sigma=2.0)

    # Should detect approximately 3 seams (or at least 2)
    passed_multi = len(detected) >= 2

    results.append(ExperimentResult(
        hypothesis="Stress",
        test_name="Multiple seams detection",
        passed=passed_multi,
        details={
            "true_seams": [75, 150, 225],
            "detected_seams": [int(d) for d in detected],
            "num_detected": len(detected)
        },
        execution_time=time.time() - start
    ))

    return results


def main():
    """Run all experiments and compile results"""
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTAL VALIDATION FOR SEAMAWARE")
    print("=" * 70)
    print()

    all_results = []

    # Run all hypothesis tests
    print("Running H1: MDL Reduction Tests...")
    all_results.extend(run_hypothesis_1_mdl_reduction())

    print("Running H2: k* Validation Tests...")
    all_results.extend(run_hypothesis_2_k_star_validation())

    print("Running H3: Localization Tests...")
    all_results.extend(run_hypothesis_3_localization())

    print("Running H4: Involution Property Tests...")
    all_results.extend(run_hypothesis_4_involution())

    print("Running H5: Universality Tests...")
    all_results.extend(run_hypothesis_5_universality())

    print("Running Stress Tests...")
    all_results.extend(run_stress_tests())

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Group by hypothesis
    hypotheses = {}
    for result in all_results:
        h = result.hypothesis
        if h not in hypotheses:
            hypotheses[h] = []
        hypotheses[h].append(result)

    total_passed = 0
    total_tests = 0

    for h, tests in sorted(hypotheses.items()):
        passed = sum(1 for t in tests if t.passed)
        total = len(tests)
        total_passed += passed
        total_tests += total

        status = "✓ VALIDATED" if passed == total else "⚠ PARTIAL" if passed > 0 else "✗ BROKEN"
        print(f"\n{h}: {status} ({passed}/{total} tests passed)")
        print("-" * 50)

        for test in tests:
            status_symbol = "✓" if test.passed else "✗"
            print(f"  {status_symbol} {test.test_name}")
            print(f"    Time: {test.execution_time:.2f}s")
            for key, value in test.details.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                    print(f"    {key}: [{', '.join(f'{v:.3f}' for v in value)}]")
                else:
                    print(f"    {key}: {value}")

    print()
    print("=" * 70)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print("=" * 70)

    # Save detailed results to JSON
    output_path = Path(__file__).parent / "validation_results.json"
    json_results = [
        {
            "hypothesis": r.hypothesis,
            "test_name": r.test_name,
            "passed": r.passed,
            "details": r.details,
            "execution_time": r.execution_time
        }
        for r in all_results
    ]

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
