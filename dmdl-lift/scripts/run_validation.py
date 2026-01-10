"""Standalone validation runner script.

This script runs the Phase 0 validation and prints a frozen config summary.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from dmdl_lift.validate import validate_method
from dmdl_lift.plot import plot_results

print("HARDENED PHASE 0: Reviewer-Proof Edition")
print("Enforced ground truth + conservative fallback + clean cutoff\n")

results = validate_method(L=6)

if results is not None:
    fig = plot_results(results)
    fig.savefig('phase0_hardened_validation.png', dpi=150, bbox_inches='tight')
    print("\n→ Figure saved: phase0_hardened_validation.png")

    if results['success']:
        print("\n" + "="*60)
        print("FROZEN CONFIGURATION FOR REAL DATA:")
        print("="*60)
        print(f"L = {results['config']['L']}")
        print("\nAllowed preprocessing (only):")
        print("  • z-score normalization")
        print("  • linear detrend (if visually obvious)")
        print("  • 3-point moving average (apply to both)")
        print("\nForbidden:")
        print("  • ANY parameter changes")
        print("  • Per-dataset tuning")
        print("  • Model family changes")
        print("="*60)

    plt.close(fig)
