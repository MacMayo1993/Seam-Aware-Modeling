#!/usr/bin/env python3
"""
MASS/SMASH v2 CLI demo — thin shim over seamaware.pipeline.

All algorithm code lives in seamaware/pipeline.py.
This file provides the command-line entry point only.
"""

from seamaware.pipeline import (  # noqa: F401  (re-export for backward compat)
    EPS,
    TRANSFORMS,
    ARModel,
    BaseModel,
    BatchResults,
    FitResult,
    FourierModel,
    HAS_MLP,
    MASSSMASHConfig,
    MLPModel,
    MeanModel,
    PolynomialModel,
    Solution,
    antipodal_symmetry_scanner,
    antipodal_symmetry_scanner_vector,
    apply_reflect_invert,
    apply_sign_flip,
    beam_search_configurations,
    bic_from_rss,
    build_model_zoo,
    detect_seam_candidates,
    enumerate_configurations,
    fit_best_model,
    gaussian_nll_bits,
    generate_signal_with_seams,
    get_segments,
    mdl_bits,
    piecewise_fit,
    plot_solution,
    roughness_detector,
    run_batch_evaluation,
    run_mass_smash,
    safe_log2,
    score_solution,
    set_seed,
    zscore,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MASS/SMASH v2 demo and batch evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise std (default 0.1; higher values reduce recall)")
    parser.add_argument("--batch", action="store_true",
                        help="Run 50-trial batch evaluation after the demo")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the demo plot")
    parser.add_argument("--vector-demo", action="store_true",
                        help="Run synthetic 3-component sign-flip demo for vector scanner")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("MASS/SMASH v2: Multi-Seam Modeling with Model Zoo")
    print("=" * 70)
    print()
    print("Pipeline:")
    print("  1. Seam proposal: antipodal + roughness detectors with NMS")
    print("  2. Configuration search: bounded subset enumeration")
    print("  3. Model zoo: Fourier / Poly / AR competition per segment")
    print("  4. MDL selection: global objective with explicit seam penalty")
    print()

    # Demo with 2 seams.
    # MLP is excluded: a global MLP can fit noisy sign-flip signals without
    # seams, causing the no-seam solution to win at high noise levels.
    print("=" * 70)
    print("DEMO: Signal with 2 sign-flip seams")
    print("=" * 70)

    y, true_seams = generate_signal_with_seams(
        T=300,
        noise_std=args.noise,
        seam_positions=[0.33, 0.67],
        seam_types=["sign_flip", "sign_flip"],
        seed=args.seed,
    )
    print(f"True seam positions: {true_seams}  (noise_std={args.noise})")

    config = MASSSMASHConfig(
        max_seams=3,
        alpha=2.0,
        verbose=True,
        include_mlp=False,  # MLP excluded from demo (see comment above)
    )

    best, all_solutions = run_mass_smash(y, config, true_seams)

    # Show top 5 solutions
    print("\nTop 5 solutions by MDL:")
    for i, sol in enumerate(all_solutions[:5]):
        print(
            f"  {i+1}. seams={list(sol.seams)}, transform={sol.transform}, "
            f"MDL={sol.total_mdl:.2f}, MSE={sol.total_mse:.6f}"
        )

    detected = list(best.seams)
    print(f"\nResult: detected={detected}, true={true_seams}")
    if detected:
        errors = [min(abs(t - d) for d in detected) for t in true_seams]
        print(f"Localization errors: {errors} samples")
    else:
        print("No seams detected (signal may require lower noise or different config).")

    if not args.no_plot:
        candidates = detect_seam_candidates(y, config)
        plot_solution(y, best, candidates, true_seams, save_path="mass_smash_demo.png")
        print("Demo plot saved to mass_smash_demo.png")

    if args.batch:
        print()
        print("=" * 70)
        print("BATCH EVALUATION: 50 signals, 1 seam each")
        print("=" * 70)

        results = run_batch_evaluation(
            n_runs=50,
            T=300,
            noise_std=args.noise,
            n_true_seams=1,
            config=MASSSMASHConfig(alpha=2.0, include_mlp=False),
        )

        print()
        print(results.summary())

    if args.vector_demo:
        import numpy as np
        print()
        print("=" * 70)
        print("VECTOR DEMO: 3-component sign-flip at τ=100")
        print("=" * 70)

        T_v = 200
        tau_v = 100
        t_v = np.linspace(0, 4 * np.pi, T_v)
        Bx = np.sin(t_v)
        By = np.cos(t_v)
        Bz = np.sin(2 * t_v)
        Bx[tau_v:] *= -1
        By[tau_v:] *= -1
        Bz[tau_v:] *= -1
        B_demo = np.column_stack([Bx, By, Bz])

        vec_candidates = antipodal_symmetry_scanner_vector(
            B_demo, window_size=40, threshold=0.3, top_k=5, min_separation=20
        )
        print(f"True seam: τ={tau_v}")
        print(f"Detected candidates: {vec_candidates}")
        if any(abs(idx - tau_v) < 15 for idx, _ in vec_candidates):
            print("Detection: PASS (seam found within 15 samples of truth)")
        else:
            print(f"Detection: nearest candidate at "
                  f"{min(abs(idx - tau_v) for idx, _ in vec_candidates)} samples from truth")

        # Also show detect_seam_candidates routing
        vec_config = MASSSMASHConfig(vector_mode=True, verbose=False)
        all_cands = detect_seam_candidates(B_demo, vec_config)
        print(f"detect_seam_candidates (vector_mode=True): {all_cands}")

    print()
    print("Done.")
