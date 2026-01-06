"""Interactive CLI demo for SeamAware framework."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from seamaware.core import SignFlipAtom, detect_seams_roughness
from seamaware.models.baselines import FourierBaseline


def main():
    """Run the SeamAware demonstration."""
    print("=" * 70)
    print("SeamAware Framework - Interactive Demo")
    print("=" * 70)
    print()
    print("Demonstrating seam detection in non-orientable time series...")
    print()

    # Generate synthetic signal with hidden sign flip
    print("1. Generating synthetic signal with hidden orientation seam...")
    N = 200
    t = np.linspace(0, 4 * np.pi, N)
    freq = 2.0

    # Create signal with sign flip at midpoint
    signal = np.sin(freq * t)
    seam_idx = 100  # Midpoint for clarity
    signal[seam_idx:] *= -1  # Hidden orientation flip

    # Add noise (SNR ≈ 12.5)
    np.random.seed(42)
    noise = 0.08 * np.random.randn(N)
    noisy_signal = signal + noise

    print(f"   ✓ Signal length: {N} points")
    print(f"   ✓ True seam location: t={seam_idx}")
    print(f"   ✓ SNR: {1.0 / 0.08:.1f} (> k* ≈ 0.721)")
    print()

    # Baseline: Fourier without seam awareness
    print("2. Computing baseline (Fourier, no seam awareness)...")
    fourier_baseline = FourierBaseline(K=10)
    recon_baseline = fourier_baseline.fit_predict(noisy_signal)
    residuals_baseline = noisy_signal - recon_baseline
    mdl_baseline = (fourier_baseline.num_params() * 32 +
                   len(noisy_signal) * np.log2(np.var(residuals_baseline) + 1e-10))
    print(f"   ✓ Baseline MDL: {mdl_baseline:.2f} bits")
    print()

    # SeamAware: Detect and model seam
    print("3. Applying SeamAware detection...")
    flip_atom = SignFlipAtom()

    # Detect seam by finding the largest absolute jump in the signal
    # For sign flips, this is the most reliable method
    first_diff = np.abs(np.diff(noisy_signal))
    detected_seam = int(np.argmax(first_diff) + 1)

    print(f"   ✓ Detected seam at: t={detected_seam}")
    print(f"   ✓ True seam at: t={seam_idx}")
    print(f"   ✓ Detection error: {abs(detected_seam - seam_idx)} points")
    print()

    # Apply flip atom
    print("4. Applying orientation correction (flip atom)...")
    corrected_signal = flip_atom.apply(noisy_signal, detected_seam)

    # Recompute MDL with seam-aware model
    fourier_seamaware = FourierBaseline(K=10)
    recon_seamaware = fourier_seamaware.fit_predict(corrected_signal)
    residuals_seamaware = corrected_signal - recon_seamaware
    mdl_seamaware = (fourier_seamaware.num_params() * 32 +
                   32 +  # Cost of encoding seam location
                   len(corrected_signal) * np.log2(np.var(residuals_seamaware) + 1e-10))

    print(f"   ✓ SeamAware MDL: {mdl_seamaware:.2f} bits")
    print()

    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    mdl_reduction = mdl_baseline - mdl_seamaware
    pct_improvement = 100 * mdl_reduction / mdl_baseline

    print(f"Baseline MDL:    {mdl_baseline:.2f} bits")
    print(f"SeamAware MDL:   {mdl_seamaware:.2f} bits")
    print(f"MDL Reduction:   {mdl_reduction:.2f} bits ({pct_improvement:.1f}% improvement)")
    print()

    if pct_improvement > 10:
        print("✓ Significant MDL reduction achieved!")
        print("  → Seam-aware modeling successfully exploited non-orientable structure")
    else:
        print("⚠ Modest improvement (expected for this SNR regime)")

    print()
    print("=" * 70)
    print("Next Steps:")
    print("  - Run examples/quick_start.ipynb for interactive visualizations")
    print("  - See README.md for theoretical background")
    print("  - Explore scripts/ for advanced usage")
    print("=" * 70)


if __name__ == "__main__":
    main()
