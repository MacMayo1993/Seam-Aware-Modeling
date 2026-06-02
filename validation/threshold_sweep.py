"""
Threshold Sweep and Parameter Sensitivity Module
=================================================
Sweeps the MDL gain threshold and key pipeline parameters to characterize
precision/recall trade-offs for the MASS/SMASH seam detector.

Usage:
    python validation/threshold_sweep.py          # full run
    python validation/threshold_sweep.py --quick  # fast CI run (fewer trials)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from seamaware.pipeline import (
    generate_signal_with_seams,
    run_mass_smash,
    MASSSMASHConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOLERANCE = 15  # ±15 samples for a match


def make_benchmark(seed: int, noise_std: float = 0.3):
    """Return (y, true_seams) for a reproducible synthetic trial."""
    y, true_seams = generate_signal_with_seams(
        T=300,
        noise_std=noise_std,
        seam_positions=[0.33, 0.67],
        seam_types=["sign_flip", "sign_flip"],
        seed=seed,
    )
    return y, true_seams


def precision_recall(detected, true_seams, tol=TOLERANCE):
    """Compute precision and recall with ±tol-sample tolerance matching."""
    if len(detected) == 0:
        prec = 1.0 if len(true_seams) == 0 else 0.0
        rec = 1.0 if len(true_seams) == 0 else 0.0
        return prec, rec
    if len(true_seams) == 0:
        return 0.0, 1.0

    matched_true = set()
    tp = 0
    for d in detected:
        for i, t in enumerate(true_seams):
            if abs(d - t) <= tol and i not in matched_true:
                tp += 1
                matched_true.add(i)
                break

    prec = tp / len(detected)
    rec = tp / len(true_seams)
    return prec, rec


def get_mdl_gain(y, config):
    """
    Run run_mass_smash and return (detected_seams, mdl_gain).

    mdl_gain = no_seam_mdl - best_mdl when best has seams, else 0.
    detected_seams is the list of seam indices from the best solution.
    """
    best, all_solutions = run_mass_smash(y, config)

    # Find the no-seam baseline
    no_seam_sol = next(
        (s for s in all_solutions if s.n_seams == 0), None
    )

    if no_seam_sol is None or best.n_seams == 0:
        return list(best.seams), 0.0

    gain = no_seam_sol.total_mdl - best.total_mdl
    return list(best.seams), max(gain, 0.0)


# ---------------------------------------------------------------------------
# 1. MDL Gain Threshold Sweep
# ---------------------------------------------------------------------------

GAIN_THRESHOLDS = [2, 5, 10, 20, 40, 80]


def run_gain_threshold_sweep(n_trials: int = 20):
    """
    For each gain threshold G, run n_trials and collect precision/recall.

    Returns dict: G -> (mean_precision, mean_recall)
    Also returns raw per-trial data: list of (gain, true_seams, detected_seams).
    """
    print(f"\n=== MDL Gain Threshold Sweep ({n_trials} trials) ===")

    # Collect raw data for all trials (using default config)
    raw = []  # list of (mdl_gain, true_seams, detected_seams)
    config = MASSSMASHConfig(verbose=False)

    for seed in range(n_trials):
        y, true_seams = make_benchmark(seed)
        detected, gain = get_mdl_gain(y, config)
        raw.append((gain, true_seams, detected))
        print(f"  Trial {seed:3d}: gain={gain:.1f}  detected={detected}  true={true_seams}")

    results = {}
    for G in GAIN_THRESHOLDS:
        precs, recs = [], []
        for gain, true_seams, detected in raw:
            # Apply threshold: only keep detection if gain >= G
            if gain >= G:
                det = detected
            else:
                det = []
            p, r = precision_recall(det, true_seams)
            precs.append(p)
            recs.append(r)
        results[G] = (float(np.mean(precs)), float(np.mean(recs)))
        print(f"  G={G:4d}: P={results[G][0]:.3f}  R={results[G][1]:.3f}")

    return results


def plot_gain_sweep(results: dict, out_path: str):
    """Three-panel figure: P-R curve, Precision vs G, Recall vs G."""
    Gs = sorted(results.keys())
    precs = [results[G][0] for G in Gs]
    recs = [results[G][1] for G in Gs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("MDL Gain Threshold Sweep", fontsize=13, fontweight="bold")

    # Panel 1: P-R curve
    ax = axes[0]
    ax.plot(recs, precs, "o-", color="steelblue", linewidth=2, markersize=8)
    for G, p, r in zip(Gs, precs, recs):
        ax.annotate(f"G={G}", (r, p), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color="steelblue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: Precision vs G
    ax = axes[1]
    ax.plot(Gs, precs, "s-", color="darkorange", linewidth=2, markersize=8)
    ax.set_xlabel("Gain Threshold G (bits)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Gain Threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: Recall vs G
    ax = axes[2]
    ax.plot(Gs, recs, "^-", color="forestgreen", linewidth=2, markersize=8)
    ax.set_xlabel("Gain Threshold G (bits)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs Gain Threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. Parameter Sensitivity
# ---------------------------------------------------------------------------

PARAM_SWEEPS = {
    "antipodal_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "roughness_threshold": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "antipodal_window":    [20, 30, 40, 60, 80],
    "alpha":               [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    "min_separation":      [10, 20, 30, 50, 80],
}


def run_param_sensitivity(n_trials: int = 10):
    """
    Sweep each parameter independently while holding others at default.

    Returns dict: param_name -> list of (value, mean_P, mean_R, mean_F1)
    """
    print(f"\n=== Parameter Sensitivity ({n_trials} trials per point) ===")

    sensitivity = {}

    for param, values in PARAM_SWEEPS.items():
        print(f"\n  Parameter: {param}")
        rows = []
        for val in values:
            kwargs = {param: val, "verbose": False}
            config = MASSSMASHConfig(**kwargs)

            precs, recs, f1s = [], [], []
            for seed in range(n_trials):
                y, true_seams = make_benchmark(seed)
                best, all_solutions = run_mass_smash(y, config)
                detected = list(best.seams)
                p, r = precision_recall(detected, true_seams)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                precs.append(p)
                recs.append(r)
                f1s.append(f1)

            mp = float(np.mean(precs))
            mr = float(np.mean(recs))
            mf = float(np.mean(f1s))
            rows.append((val, mp, mr, mf))
            print(f"    {param}={val}: P={mp:.3f}  R={mr:.3f}  F1={mf:.3f}")

        sensitivity[param] = rows

    return sensitivity


def plot_param_sensitivity(sensitivity: dict, out_path: str):
    """Five-panel figure, one subplot per parameter."""
    params = list(PARAM_SWEEPS.keys())
    n = len(params)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Parameter Sensitivity", fontsize=13, fontweight="bold")

    colors = {"P": "steelblue", "R": "forestgreen", "F1": "darkorange"}
    markers = {"P": "o", "R": "^", "F1": "s"}

    for ax, param in zip(axes, params):
        rows = sensitivity[param]
        vals = [r[0] for r in rows]
        ps   = [r[1] for r in rows]
        rs   = [r[2] for r in rows]
        fs   = [r[3] for r in rows]

        for label, data in [("P", ps), ("R", rs), ("F1", fs)]:
            ax.plot(vals, data, marker=markers[label], color=colors[label],
                    linewidth=1.8, markersize=6, label=label)

        ax.set_xlabel(param, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title(param, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(gain_results: dict, sensitivity: dict):
    """Print a concise summary to stdout."""
    print("\n" + "=" * 60)
    print("SUMMARY: MDL Gain Threshold Sweep")
    print("=" * 60)
    print(f"{'G (bits)':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 46)
    for G in sorted(gain_results.keys()):
        p, r = gain_results[G]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{G:>10}  {p:>10.3f}  {r:>10.3f}  {f1:>10.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY: Parameter Sensitivity (best F1 per parameter)")
    print("=" * 60)
    print(f"{'Parameter':>25}  {'Best Value':>12}  {'Best F1':>10}")
    print("-" * 52)
    for param, rows in sensitivity.items():
        best = max(rows, key=lambda r: r[3])
        print(f"{param:>25}  {best[0]:>12}  {best[3]:>10.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Threshold sweep and parameter sensitivity for MASS/SMASH"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 5 trials (instead of 20/10) for fast CI runs"
    )
    args = parser.parse_args()

    n_gain = 5 if args.quick else 20
    n_sens = 5 if args.quick else 10

    # Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Gain threshold sweep
    gain_results = run_gain_threshold_sweep(n_trials=n_gain)
    plot_gain_sweep(gain_results, os.path.join(out_dir, "threshold_sweep.png"))

    # 2. Parameter sensitivity
    sensitivity = run_param_sensitivity(n_trials=n_sens)
    plot_param_sensitivity(sensitivity, os.path.join(out_dir, "param_sensitivity.png"))

    # Summary table
    print_summary_table(gain_results, sensitivity)


if __name__ == "__main__":
    main()
