"""
PELT (Pruned Exact Linear Time) baseline comparator for the SNR sweep.

Uses ruptures.Pelt with an L2 (Gaussian mean-change) model and a
BIC-derived penalty.  The penalty is noise-adaptive:

    pen = log(N) * sigma_noise^2

where sigma_noise is estimated from the median absolute deviation of
first-differences in a noise-dominated tail of the signal.

Run this AFTER snr_sweep.py (reads snr_sweep_results.json for the
MASS/SMASH / PVI / Baseline numbers, appends PELT results, and
produces an updated summary).

Requires: pip install ruptures
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.dirname(__file__))

from snr_sweep import generate, evaluate, TURB_FRACS, CADENCE_S, TOLERANCE_S

try:
    import ruptures as rpt
except ImportError:
    raise ImportError("Install ruptures:  pip install ruptures")


MIN_SEP_S = 300.0   # minimum separation between detections (seconds)


def pelt_detect(signal, times, pen_scale=1.0):
    """
    Run PELT on `signal` and return detected change-point indices.

    pen_scale multiplies the BIC penalty (>1 → fewer detections,
    higher precision; <1 → more detections, higher recall).
    """
    N  = len(signal)
    dt = float(np.median(np.diff(times)))
    min_sep = int(MIN_SEP_S / dt)

    # Noise estimate: MAD of first-differences / sqrt(2) (robust to jumps)
    diffs = np.diff(signal)
    sigma = np.median(np.abs(diffs - np.median(diffs))) / (np.sqrt(2) * 0.6745)
    sigma = max(sigma, 1e-3)

    # BIC penalty for Gaussian mean-change: log(N) * sigma^2
    pen = pen_scale * np.log(N) * (sigma ** 2)

    # jump=10 evaluates every 10th sample as a potential breakpoint;
    # 30s resolution is fine given our 150s tolerance window.
    algo = rpt.Pelt(model='l2', min_size=min_sep, jump=10)
    algo.fit(signal.reshape(-1, 1))
    bkpts = algo.predict(pen=pen)

    # bkpts includes N at the end; exclude it
    cps = np.array([b for b in bkpts if b < N], dtype=int)

    # Minimum-separation post-filter (greedy, ordered by distance to segment mean jump)
    if len(cps) == 0:
        return cps

    # Compute mean-change magnitude for each breakpoint, keep largest-gap-first
    jumps = []
    for cp in cps:
        lo = signal[max(0, cp - min_sep // 2) : cp]
        hi = signal[cp : min(N, cp + min_sep // 2)]
        jump = abs(np.mean(hi) - np.mean(lo)) if len(lo) > 0 and len(hi) > 0 else 0.0
        jumps.append((cp, jump))
    jumps.sort(key=lambda x: -x[1])

    kept = []
    for cp, _ in jumps:
        if all(abs(cp - k) >= min_sep for k in kept):
            kept.append(int(cp))
    return np.array(sorted(kept), dtype=int)


def run(pen_scale=1.0):
    os.makedirs('outputs', exist_ok=True)

    # Load existing results (if any) to append PELT column
    prev_path = 'outputs/snr_sweep_results.json'
    prev = {}
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)

    results = {}
    tol = int(TOLERANCE_S / CADENCE_S)

    print(f"\nPELT SNR sweep  (pen_scale={pen_scale})")
    print(f"{'Turb':>6}  {'SNR':>6}  {'P':>7}  {'R':>7}  {'F1':>7}  {'n_det':>6}")
    print('-' * 55)

    for turb_frac in TURB_FRACS:
        snr = 1.0 / turb_frac
        times, Bz, B_vec, B_mag, true_locs = generate(turb_frac, seed=42)
        pelt_peaks = pelt_detect(Bz, times, pen_scale=pen_scale)
        r = evaluate(pelt_peaks, true_locs, tol)
        key = f'{turb_frac:.2f}'
        results[key] = {
            'turb_frac': float(turb_frac),
            'snr':       float(snr),
            'pelt':      r,
            'n_det':     int(len(pelt_peaks)),
            'pen_scale': float(pen_scale),
        }
        print(f"  {turb_frac:.0%}   {snr:>4.0f}:1  "
              f"{r['precision']:.3f}  {r['recall']:.3f}  "
              f"{r['f1']:.3f}  {len(pelt_peaks):>6}")

    out_path = 'outputs/pelt_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump({'pen_scale': pen_scale, 'results': results}, f, indent=2)
    print(f"\nSaved {out_path}")

    _plot(results, prev, pen_scale)
    return results


def run_pen_grid():
    """Run PELT at multiple penalty scales and report the best F1 at each level."""
    tol = int(TOLERANCE_S / CADENCE_S)
    os.makedirs('outputs', exist_ok=True)

    pen_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    grid = {}   # grid[turb_key][pen_scale] = {P, R, F1, n_det}

    for pen_scale in pen_scales:
        print(f"\n  pen_scale = {pen_scale}")
        for turb_frac in TURB_FRACS:
            times, Bz, _, _, true_locs = generate(turb_frac, seed=42)
            peaks = pelt_detect(Bz, times, pen_scale=pen_scale)
            r = evaluate(peaks, true_locs, tol)
            key = f'{turb_frac:.2f}'
            if key not in grid:
                grid[key] = {}
            grid[key][pen_scale] = r

    # Pick the pen_scale that maximises mean F1 across turbulence levels
    best_pen = max(pen_scales,
                   key=lambda p: np.mean([grid[k][p]['f1'] for k in grid]))
    print(f"\nBest pen_scale (max mean F1): {best_pen}")
    for key in sorted(grid.keys()):
        best = grid[key][best_pen]
        print(f"  turb={key}  P={best['precision']:.3f}  "
              f"R={best['recall']:.3f}  F1={best['f1']:.3f}  "
              f"n={best['n_det']}")

    with open('outputs/pelt_grid_results.json', 'w') as f:
        json.dump({'pen_scales': pen_scales, 'grid': grid,
                   'best_pen_scale': best_pen}, f, indent=2)
    print("Saved outputs/pelt_grid_results.json")
    return grid, best_pen


def _plot(pelt_results, prev_results, pen_scale):
    turb_fracs = sorted(float(k) for k in pelt_results)
    x_labels   = [f'{f:.0%}' for f in turb_fracs]

    pelt_p  = [pelt_results[f'{f:.2f}']['pelt']['precision'] for f in turb_fracs]
    pelt_r  = [pelt_results[f'{f:.2f}']['pelt']['recall']    for f in turb_fracs]
    pelt_f  = [pelt_results[f'{f:.2f}']['pelt']['f1']        for f in turb_fracs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (vals, title) in zip(axes, [
            ([(pelt_p, pelt_r, pelt_f)], 'PELT'),  # placeholder structure
    ]):
        pass  # will build below

    axes[0].set_ylabel('Precision')
    axes[1].set_ylabel('Recall')
    axes[2].set_ylabel('F1')

    metrics = [
        ('precision', pelt_p, 'Precision'),
        ('recall',    pelt_r, 'Recall'),
        ('f1',        pelt_f, 'F1'),
    ]

    # Load MS / PVI / Baseline from snr_sweep_results.json if available
    base_data = {}
    if prev_results:
        for key, v in prev_results.items():
            base_data[key] = v

    method_styles = {
        'mass_smash': ('navy',          'D-',  'MASS/SMASH'),
        'pvi':        ('coral',         's--', 'Standard PVI'),
        'baseline':   ('mediumseagreen','^:',  'Baseline'),
    }

    for ax, (metric_key, pelt_vals, ylabel) in zip(axes, metrics):
        # Plot existing methods from snr_sweep_results if available
        for mkey, (color, ls, label) in method_styles.items():
            if base_data:
                try:
                    vals = [base_data[f'{f:.2f}'][mkey][metric_key]
                            for f in turb_fracs]
                    ax.plot(x_labels, vals, ls, color=color, label=label, lw=2, ms=7)
                except (KeyError, TypeError):
                    pass

        ax.plot(x_labels, pelt_vals, 'P-', color='darkviolet', lw=2, ms=8,
                label=f'PELT (pen×{pen_scale})')
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel('Turbulence amplitude (× B₀)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelrotation=30)

    fig.suptitle(f'SNR Sweep with PELT comparator (pen_scale={pen_scale})', fontsize=11)
    plt.tight_layout()
    path = f'outputs/fig_pelt_sweep.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pen-scale', type=float, default=1.0,
                        help='Penalty multiplier for BIC (default 1.0)')
    parser.add_argument('--grid', action='store_true',
                        help='Run penalty grid search')
    args = parser.parse_args()

    if args.grid:
        run_pen_grid()
    else:
        run(pen_scale=args.pen_scale)
