"""
Multi-seed SNR sweep: run the full turbulence sweep over N_SEEDS random seeds
and report mean ± std for precision, recall, and F1.

Seeds: SEEDS = [42, 43, 44, 45, 46]
Turbulence levels: TURB_FRACS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
Methods: MASS/SMASH, Standard PVI, Baseline

Runtime estimate: 5 seeds × 7 levels × ~2 min (MASS/SMASH) ≈ 70 min.
PVI and baseline add ~10 min total.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.dirname(__file__))

from mass_smash import MASSSMASHConfig, run_mass_smash
from catalog import compute_pvi, get_pvi_events
from baseline import baseline_detector
from snr_sweep import (generate, evaluate,
                       TURB_FRACS, CADENCE_S, TOLERANCE_S, B0)

SEEDS   = [42, 43, 44, 45, 46]
N_SEEDS = len(SEEDS)


# ── MASS/SMASH chunked runner ─────────────────────────────────────────────────

def _run_ms(signal, times, gain_threshold=20.0):
    dt = float(np.median(np.diff(times)))
    W  = int(600 / dt)
    S  = int(300 / dt)
    min_sep_g = int(300 / dt)
    N = len(signal)

    cfg = MASSSMASHConfig(
        top_k_candidates=3,
        min_separation=int(60 / dt),
        antipodal_threshold=0.35,
        roughness_threshold=0.20,
        max_seams=1,
        alpha=2.0,
        verbose=False,
        include_mlp=False,
    )

    hit_count = np.zeros(N, dtype=np.int32)
    mdl_gain  = np.zeros(N, dtype=np.float64)

    for start in range(0, N - W, S):
        chunk = signal[start : start + W]
        best, all_sols = run_mass_smash(chunk, cfg)
        if best.n_seams > 0:
            no_seam = next((s for s in all_sols if s.n_seams == 0), None)
            gain = (no_seam.total_mdl - best.total_mdl) if no_seam else 1.0
            for local_idx in best.seams:
                g = start + local_idx
                if 0 <= g < N:
                    hit_count[g] += 1
                    mdl_gain[g]  += max(0.0, gain)

    mask  = (hit_count >= 1) & (mdl_gain > gain_threshold)
    cands = np.where(mask)[0]
    if len(cands) == 0:
        return np.array([], dtype=int)

    scores = mdl_gain[cands]
    order  = np.argsort(scores)[::-1]
    kept   = []
    for pos in cands[order]:
        if all(abs(pos - k) >= min_sep_g for k in kept):
            kept.append(int(pos))
    return np.array(sorted(kept), dtype=int)


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('outputs', exist_ok=True)

    # raw_results[seed][turb_key][method] = {precision, recall, f1, n_det}
    raw_results = {}

    t0 = time.time()
    for seed in SEEDS:
        raw_results[seed] = {}
        print(f"\n{'='*60}")
        print(f"SEED {seed}  ({SEEDS.index(seed)+1}/{N_SEEDS})")
        print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
        print('='*60)

        for turb_frac in TURB_FRACS:
            snr = 1.0 / turb_frac
            times, Bz, B_vec, B_mag, true_locs = generate(turb_frac, seed=seed)
            tol = int(TOLERANCE_S / CADENCE_S)

            pvi      = compute_pvi(B_vec)
            _, pvi_peaks = get_pvi_events(pvi, times,
                                          threshold=3.0, min_separation_s=300)
            bl_peaks, _ = baseline_detector(B_mag, times, threshold_sigma=2.5)
            ms_peaks    = _run_ms(Bz, times)

            ms_r  = evaluate(ms_peaks,  true_locs, tol)
            pvi_r = evaluate(pvi_peaks, true_locs, tol)
            bl_r  = evaluate(bl_peaks,  true_locs, tol)

            key = f'{turb_frac:.2f}'
            raw_results[seed][key] = {
                'turb_frac':  float(turb_frac),
                'snr':        float(snr),
                'n_true':     int(len(true_locs)),
                'mass_smash': ms_r,
                'pvi':        pvi_r,
                'baseline':   bl_r,
            }

            print(f"  turb={turb_frac:.0%}  n_true={len(true_locs)}"
                  f"  MS:{ms_r['f1']:.3f}  PVI:{pvi_r['f1']:.3f}"
                  f"  BL:{bl_r['f1']:.3f}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    print(f"\n\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
    print("\nAggregating across seeds ...")

    stats = {}   # stats[turb_key][method][metric] = {'mean': x, 'std': y}
    for turb_frac in TURB_FRACS:
        key = f'{turb_frac:.2f}'
        stats[key] = {'turb_frac': float(turb_frac), 'snr': 1.0/turb_frac}
        for method in ('mass_smash', 'pvi', 'baseline'):
            stats[key][method] = {}
            for metric in ('precision', 'recall', 'f1', 'n_det'):
                vals = [raw_results[s][key][method][metric] for s in SEEDS]
                stats[key][method][metric] = {
                    'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals, ddof=1)),
                    'min':  float(np.min(vals)),
                    'max':  float(np.max(vals)),
                    'all':  vals,
                }

    out = {'seeds': SEEDS, 'raw': raw_results, 'stats': stats}
    with open('outputs/multi_seed_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved outputs/multi_seed_results.json")

    _print_table(stats)
    _plot(stats)
    return out


def _print_table(stats):
    print("\n\nSUMMARY: mean ± std across 5 seeds")
    print(f"{'Turb':>6}  {'SNR':>6}  "
          f"{'MS P':>10}  {'MS R':>10}  {'MS F1':>10}  "
          f"{'PVI P':>10}  {'PVI R':>10}  {'PVI F1':>10}")
    print('-' * 86)
    for key in sorted(stats.keys()):
        v = stats[key]
        ms  = v['mass_smash']
        pvi = v['pvi']
        def fmt(m):
            return f"{m['mean']:.3f}±{m['std']:.3f}"
        print(f"  {v['turb_frac']:.0%}  {v['snr']:>5.1f}:1  "
              f"  {fmt(ms['precision'])}  {fmt(ms['recall'])}  {fmt(ms['f1'])}  "
              f"  {fmt(pvi['precision'])}  {fmt(pvi['recall'])}  {fmt(pvi['f1'])}")


def _plot(stats):
    turb_fracs = sorted(stats[k]['turb_frac'] for k in stats)
    x_labels   = [f'{f:.0%}' for f in turb_fracs]
    snr_vals   = [1.0/f for f in turb_fracs]

    styles = {
        'mass_smash': ('navy',         'D-',  'MASS/SMASH'),
        'pvi':        ('coral',        's--', 'Standard PVI'),
        'baseline':   ('mediumseagreen','^:',  'Baseline'),
    }
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        for mkey, (color, ls, label) in styles.items():
            means = [stats[f'{f:.2f}'][mkey][metric]['mean'] for f in turb_fracs]
            stds  = [stats[f'{f:.2f}'][mkey][metric]['std']  for f in turb_fracs]
            means = np.array(means)
            stds  = np.array(stds)
            x = np.arange(len(turb_fracs))
            ax.plot(x_labels, means, ls, color=color, label=label, lw=2, ms=7)
            ax.fill_between(x_labels,
                            np.clip(means - stds, 0, 1),
                            np.clip(means + stds, 0, 1),
                            color=color, alpha=0.15)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel('Turbulence amplitude (× B₀)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelrotation=30)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(turb_fracs)))
        ax2.set_xticklabels([f'{s:.0f}:1' for s in snr_vals], fontsize=7)
        ax2.set_xlabel('SNR', fontsize=8)

    n = len(SEEDS)
    fig.suptitle(f'SNR Sweep: mean ± std across {n} seeds  '
                 f'(7-day synthetic, ≈28 crossings/seed)',
                 fontsize=11)
    plt.tight_layout()
    path = 'outputs/fig_multi_seed_sweep.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    run()
