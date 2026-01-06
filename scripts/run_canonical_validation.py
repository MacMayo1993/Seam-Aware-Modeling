#!/usr/bin/env python3
"""
Canonical validation script for k* convergence.

This script runs the rigorous Monte Carlo validation with 100 trials
and saves results for publication. All parameters are fixed for reproducibility.

Usage:
    python scripts/run_canonical_validation.py [--trials 100] [--seed 42]

Outputs:
    results/k_star_validation.csv - Raw data
    results/k_star_validation.png - Publication plot
    results/validation_summary.txt - Statistical summary
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seamaware.theory.k_star import (
    compute_k_star,
    plot_k_star_validation,
    validate_k_star_convergence,
)


def main():
    parser = argparse.ArgumentParser(description="Run canonical k* validation")
    parser.add_argument(
        "--trials", type=int, default=100, help="Monte Carlo trials per SNR (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--signal-length", type=int, default=200, help="Signal length (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Canonical k* Validation - Seam-Aware Modeling Framework")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Signal length: {args.signal_length}")
    print(f"  Trials per SNR: {args.trials}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nTheoretical k* = {compute_k_star():.10f}")
    print("\nRunning Monte Carlo validation (this may take 1-2 minutes)...")

    # Run validation
    results = validate_k_star_convergence(
        signal_length=args.signal_length,
        num_snr_points=25,  # High resolution
        num_trials=args.trials,
        seed=args.seed,
    )

    # Extract results
    snr_values = results["snr_values"]
    delta_mdl_mean = results["delta_mdl_mean"]
    delta_mdl_std = results["delta_mdl_std"]
    accept_fraction = results["accept_fraction"]
    crossover_snr = results["crossover_snr"]
    k_star_theoretical = results["theoretical_k_star"]
    relative_error = results["relative_error"]
    converged = results["converged"]

    # Print summary
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Crossover SNR: {crossover_snr:.4f}")
    print(f"Theoretical k*: {k_star_theoretical:.4f}")
    print(f"Relative error: {relative_error*100:.2f}%")
    print(f"Converged (<15%): {converged}")

    if converged:
        print("\n✅ VALIDATION SUCCESSFUL - k* emerges from MDL theory")
    else:
        print(
            f"\n⚠️  Convergence not achieved (error {relative_error*100:.1f}% > 15%)"
        )
        print("   Consider increasing --trials or adjusting SNR range")

    # Save CSV
    csv_path = args.output_dir / "k_star_validation.csv"
    df = pd.DataFrame(
        {
            "snr": snr_values,
            "delta_mdl_mean": delta_mdl_mean,
            "delta_mdl_std": delta_mdl_std,
            "accept_fraction": accept_fraction,
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Data saved to: {csv_path}")

    # Save summary
    summary_path = args.output_dir / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("K* Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Signal length: {args.signal_length}\n")
        f.write(f"Trials per SNR: {args.trials}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"SNR points: 25\n\n")
        f.write("Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Crossover SNR: {crossover_snr:.6f}\n")
        f.write(f"Theoretical k*: {k_star_theoretical:.6f}\n")
        f.write(f"Relative error: {relative_error*100:.3f}%\n")
        f.write(f"Converged (<15%): {converged}\n\n")
        f.write("Statistical Summary:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean accept fraction below k*: {np.mean(accept_fraction[snr_values < k_star_theoretical]):.3f}\n")
        f.write(f"Mean accept fraction above k*: {np.mean(accept_fraction[snr_values > k_star_theoretical]):.3f}\n")
        f.write(f"Mean ΔMDL below k*: {np.mean(delta_mdl_mean[snr_values < k_star_theoretical]):.2f} bits\n")
        f.write(f"Mean ΔMDL above k*: {np.mean(delta_mdl_mean[snr_values > k_star_theoretical]):.2f} bits\n")

    print(f"📄 Summary saved to: {summary_path}")

    # Create publication plot
    plot_path = args.output_dir / "k_star_validation.png"
    plot_k_star_validation(results, save_path=str(plot_path))
    print(f"📈 Plot saved to: {plot_path}")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)

    # Return exit code based on convergence
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
