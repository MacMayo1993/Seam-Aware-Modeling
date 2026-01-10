"""Hardened validation pipeline for ΔMDL lift detection.

This validates that:
1. Periodic (rank-1 ground truth): ΔMDL ≈ 0 (no lift detected)
2. Fibonacci (rank-2 ground truth): ΔMDL >> 0 (lift detected)

The validation is "hardened" by:
- Enforcing ground truth in signal generation
- Using conservative thresholds
- Ensuring sufficient ESS per phase
"""
import numpy as np

from .words import complexity_profile, periodic_word, fibonacci_word, kmer_coverage
from .phase import phase_state_coverage, expected_phases
from .signal import generate_signal, generate_null_signal
from .delta import delta_mdl_trajectory


def validate_method(L=6):
    """Run hardened Phase 0 validation.

    Parameters
    ----------
    L : int
        K-mer length (default 6 gives 7 Fibonacci phases)

    Returns
    -------
    dict or None
        Results dictionary if validation completes, None if structural checks fail

    Notes
    -----
    Validation protocol:
    1. Check k-mer complexity (structural)
    2. Check phase state coverage
    3. Generate signals with enforced ground truth
    4. Compute ΔMDL trajectories
    5. Validate pass criteria
    """
    print("="*60)
    print("PHASE 0: HARDENED VALIDATION")
    print(f"Configuration: L={L}")
    print("="*60)

    T = 1200

    w_per = periodic_word(T)
    w_fib = fibonacci_word(T)

    # 1. K-mer complexity (structural validation)
    print("\n1. K-MER COMPLEXITY (structural):")
    print("-" * 40)

    comp_per = complexity_profile(w_per)
    comp_fib = complexity_profile(w_fib)

    print("Periodic:")
    for k, v in comp_per.items():
        print(f"  L={k}: {v} distinct")

    print("\nFibonacci:")
    for k, v in comp_fib.items():
        print(f"  L={k}: {v} distinct (expect ~{k+1})")

    ratio_at_L = kmer_coverage(w_fib, L) / max(kmer_coverage(w_per, L), 1)
    print(f"\nRatio at L={L}: {ratio_at_L:.1f}x")

    # 2. Phase state coverage
    print("\n2. PHASE STATE COVERAGE:")
    print("-" * 40)

    n_states_per, _ = phase_state_coverage(w_per, L=L)
    n_states_fib, _ = phase_state_coverage(w_fib, L=L)

    exp_per = expected_phases('periodic', L)
    exp_fib = expected_phases('fibonacci', L)

    print(f"Periodic:  {n_states_per} states (expected {exp_per})")
    print(f"Fibonacci: {n_states_fib} states (expected {exp_fib})")
    print(f"Ratio: {n_states_fib/max(n_states_per,1):.1f}x")

    per_reasonable = n_states_per == 2
    fib_reasonable = n_states_fib == L + 1

    print(f"→ Periodic matches expected: {per_reasonable}")
    print(f"→ Fibonacci matches expected: {fib_reasonable}")

    if not (per_reasonable and fib_reasonable):
        print("\n⚠️  Phase coverage mismatch - check word generation")
        return None

    # 3. Compute ΔMDL trajectories
    print("\n3. COMPUTING ΔMDL TRAJECTORIES...")
    print("-" * 40)

    # Key: Periodic uses NULL signal (no lift), Fibonacci uses lifted signal
    x_per = generate_null_signal(w_per, noise=0.06, seed=1)
    x_fib = generate_signal(w_fib, lifted=True, noise=0.06, seed=2, L=L)

    print("Ground truth:")
    print("  Periodic:  generate_null_signal (rank-1 dynamics)")
    print("  Fibonacci: generate_signal(lifted=True) (rank-2 dynamics)")

    ts_p, d_p, ph_p, ess_p = delta_mdl_trajectory(x_per, w_per, L=L, min_samples=10)
    ts_f, d_f, ph_f, ess_f = delta_mdl_trajectory(x_fib, w_fib, L=L, min_samples=10)

    per_mean = np.mean(d_p)
    per_std = np.std(d_p)
    fib_mean = np.mean(d_f)
    fib_std = np.std(d_f)

    per_phases = np.mean(ph_p)
    fib_phases = np.mean(ph_f)

    per_ess = np.mean(ess_p)
    fib_ess = np.mean(ess_f)

    frac_per_pos = np.mean(d_p > 0)
    frac_fib_pos = np.mean(d_f > 0)

    print(f"\nPeriodic:")
    print(f"  ΔMDL: {per_mean:.2f} ± {per_std:.2f}")
    print(f"  Learned phases: {per_phases:.1f}")
    print(f"  ESS/phase: {per_ess:.1f}")
    print(f"  Fraction ΔMDL>0: {frac_per_pos:.2%}")

    print(f"\nFibonacci:")
    print(f"  ΔMDL: {fib_mean:.2f} ± {fib_std:.2f}")
    print(f"  Learned phases: {fib_phases:.1f}")
    print(f"  ESS/phase: {fib_ess:.1f}")
    print(f"  Fraction ΔMDL>0: {frac_fib_pos:.2%}")

    # 4. Validation criteria
    print("\n4. VALIDATION CRITERIA:")
    print("-" * 40)

    # Periodic: should show no lift preference (ΔMDL ≤ 0)
    # With pure AR(1), rank-2 MDL penalty should make ΔMDL negative
    per_negative = per_mean < 0  # Rank-1 should be preferred
    per_phases_ok = per_phases == 2  # Should learn exactly 2 phases
    per_ess_ok = per_ess > 50  # Good ESS for 2 phases in 200-sample window

    # Fibonacci: should show strong lift preference (ΔMDL >> 0)
    fib_positive = fib_mean > 50.0  # Clear positive signal
    fib_phases_rich = fib_phases >= L  # Should learn ~L+1 phases
    fib_ess_ok = fib_ess > 15  # Reasonable ESS given more phases
    clean_sep = fib_mean - per_mean > 40.0  # Clear separation

    print("Periodic (rank-1 ground truth):")
    print(f"  ✓ μ < 0:             {per_negative} ({per_mean:.1f})")
    print(f"  ✓ phases == 2:       {per_phases_ok} ({per_phases:.1f})")
    print(f"  ✓ ESS/phase > 50:    {per_ess_ok} ({per_ess:.1f})")

    print("\nFibonacci (rank-2 ground truth):")
    print(f"  ✓ μ > 50:            {fib_positive} ({fib_mean:.1f})")
    print(f"  ✓ phases >= {L}:       {fib_phases_rich} ({fib_phases:.1f})")
    print(f"  ✓ ESS/phase > 15:    {fib_ess_ok} ({fib_ess:.1f})")
    print(f"  ✓ separation > 40:   {clean_sep} ({fib_mean - per_mean:.1f})")

    per_pass = per_negative and per_phases_ok and per_ess_ok
    fib_pass = fib_positive and fib_phases_rich and fib_ess_ok and clean_sep
    success = per_pass and fib_pass

    print("\n" + "="*60)
    print(f"METHOD VALIDATION: {'✓ PASS' if success else '✗ FAIL'}")
    print("="*60)

    if success:
        print("\n→ READY FOR REAL DATA")
        print("→ No further tuning allowed")
        print(f"→ Frozen config: L={L}")
    else:
        print("\n→ Debug required:")
        if not per_pass:
            print("  • Periodic not behaving as rank-1")
        if not fib_pass:
            print("  • Fibonacci not showing lift preference")

    return {
        'periodic': {'ts': ts_p, 'delta': d_p, 'phases': ph_p,
                     'ess': ess_p, 'signal': x_per, 'word': w_per},
        'fibonacci': {'ts': ts_f, 'delta': d_f, 'phases': ph_f,
                      'ess': ess_f, 'signal': x_fib, 'word': w_fib},
        'stats': {
            'per_mean': per_mean, 'per_std': per_std,
            'fib_mean': fib_mean, 'fib_std': fib_std,
            'per_phases': per_phases, 'fib_phases': fib_phases,
            'per_ess': per_ess, 'fib_ess': fib_ess
        },
        'complexity': {'periodic': comp_per, 'fibonacci': comp_fib},
        'config': {'L': L},
        'success': success
    }
