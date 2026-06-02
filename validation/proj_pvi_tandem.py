"""
Tandem comparison: MASS/SMASH vs standard PVI vs projective PVI.

Two synthetic benchmarks:

  1. Polarity-reversal benchmark
     True events are HCS-like sign flips: B → −B.
     MASS/SMASH and PVI results are loaded from snr_sweep_results.json.
     Only projective PVI is computed fresh.

  2. Pure-rotation benchmark (new)
     True events are genuine 90° B-direction rotations (no sign flip).
     All three methods are evaluated.
     Only run at TURB_FRACS_FAST (3 representative levels) to manage runtime.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from seamaware.pipeline import MASSSMASHConfig, run_mass_smash
from catalog import compute_pvi, compute_projective_pvi, get_pvi_events
from snr_sweep import (generate, evaluate, _turb,
                       TURB_FRACS, CADENCE_S, TOLERANCE_S, N_DAYS, B0, SEED)

TURB_FRACS_FAST = [0.05, 0.15, 0.30, 0.50]   # representative levels for rotation bench


# ── Shared runners ────────────────────────────────────────────────────────────

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


def _run_pvi(B_vec, times, threshold=3.0, projective=False):
    fn   = compute_projective_pvi if projective else compute_pvi
    pvi  = fn(B_vec)
    _, peaks = get_pvi_events(pvi, times, threshold=threshold,
                              min_separation_s=300)
    return peaks


# ── Rotation-event benchmark generator ───────────────────────────────────────

def generate_rotation(turb_frac, n_days=N_DAYS, seed=SEED):
    """
    Synthetic signal with pure 90° field-direction rotations (no polarity flip).

    At each crossing B smoothly rotates between d1 = [0, 0, 1] (z-aligned) and
    d2 = [1, 0, 0] (x-aligned).  Projective distance at each crossing:
    d_RP²(d1, d2) = arccos(|d1·d2|) = π/2  (maximum).

    Standard PVI also fires (|ΔB| large) but MASS/SMASH antipodal path does NOT
    fire (no sign flip); roughness path may fire if the Bz step is large enough.
    """
    rng = np.random.default_rng(seed)
    N   = int(n_days * 86400 / CADENCE_S)
    times = np.arange(N, dtype=float) * CADENCE_S

    n_sheets = int(n_days * 24 / 6)
    min_sep  = int(3600 / CADENCE_S)
    sheet_locs = []
    attempts = 0
    while len(sheet_locs) < n_sheets and attempts < 100_000:
        attempts += 1
        loc = rng.integers(min_sep, N - min_sep)
        if all(abs(loc - s) > min_sep for s in sheet_locs):
            sheet_locs.append(int(loc))
    sheet_locs = sorted(sheet_locs)

    d1 = np.array([0.0, 0.0, 1.0])   # z-aligned sector
    d2 = np.array([1.0, 0.0, 0.0])   # x-aligned sector (90°, no sign flip)

    dirs = np.zeros((N, 3))
    current_d = d1.copy()
    next_d    = d2.copy()
    prev_e    = 0

    for loc in sheet_locs:
        w    = rng.integers(5, 21)
        half = w // 2
        s    = max(0, loc - half)
        e    = min(N, loc + half)

        dirs[prev_e:s] = current_d

        x     = np.linspace(-3.0, 3.0, e - s)
        alpha = (np.tanh(x) + 1.0) / 2.0
        for j, a in zip(range(s, e), alpha):
            v = (1.0 - a) * current_d + a * next_d
            norm = np.linalg.norm(v)
            dirs[j] = v / (norm + 1e-12)

        prev_e    = e
        current_d, next_d = next_d, current_d

    dirs[prev_e:] = current_d

    turb_amp = turb_frac * B0
    Bx = B0 * dirs[:, 0] + _turb(N, rng, turb_amp)
    By = B0 * dirs[:, 1] + _turb(N, rng, turb_amp)
    Bz = B0 * dirs[:, 2] + _turb(N, rng, turb_amp)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_vec = np.stack([Bx, By, Bz], axis=1)

    return times, Bz, B_vec, B_mag, np.array(sheet_locs)


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('outputs', exist_ok=True)
    results = {'polarity': {}, 'rotation': {}}

    # ── 1. Polarity-reversal benchmark ────────────────────────────────────────
    # Load existing MASS/SMASH and PVI results; only compute projective PVI fresh.
    print("\n" + "="*60)
    print("BENCHMARK 1: Polarity-reversal events (B → −B)")
    print("  (MASS/SMASH + PVI loaded from snr_sweep_results.json)")
    print("="*60)

    with open('outputs/snr_sweep_results.json') as f:
        snr_data = json.load(f)

    for turb_frac in TURB_FRACS:
        key = f'{turb_frac:.2f}'
        snr  = 1.0 / turb_frac

        times, Bz, B_vec, B_mag, true_locs = generate(turb_frac)
        tol  = int(TOLERANCE_S / CADENCE_S)

        ppvi_peaks = _run_pvi(B_vec, times, projective=True)
        ppvi_r     = evaluate(ppvi_peaks, true_locs, tol)

        ms_r  = snr_data[key]['mass_smash']
        pvi_r = snr_data[key]['pvi']

        results['polarity'][key] = {
            'turb_frac': float(turb_frac),
            'snr':       float(snr),
            'n_true':    int(len(true_locs)),
            'mass_smash': ms_r,
            'pvi':        pvi_r,
            'proj_pvi':   ppvi_r,
        }

        print(f"\nturb={turb_frac:.0%}  SNR≈{snr:.1f}:1  (n_true={len(true_locs)})")
        print(f"  MASS/SMASH    P={ms_r['precision']:.3f}  R={ms_r['recall']:.3f}"
              f"  F1={ms_r['f1']:.3f}  (det={ms_r['n_det']})")
        print(f"  Std PVI       P={pvi_r['precision']:.3f}  R={pvi_r['recall']:.3f}"
              f"  F1={pvi_r['f1']:.3f}  (det={pvi_r['n_det']})")
        print(f"  Proj PVI      P={ppvi_r['precision']:.3f}  R={ppvi_r['recall']:.3f}"
              f"  F1={ppvi_r['f1']:.3f}  (det={ppvi_r['n_det']})")

    # ── 2. Pure-rotation benchmark (representative levels) ────────────────────
    print("\n" + "="*60)
    print("BENCHMARK 2: Pure-rotation events (90°, no sign flip)")
    print(f"  Turbulence levels: {[f'{f:.0%}' for f in TURB_FRACS_FAST]}")
    print("="*60)

    for turb_frac in TURB_FRACS_FAST:
        snr = 1.0 / turb_frac
        times, Bz, B_vec, B_mag, true_locs = generate_rotation(turb_frac)
        tol = int(TOLERANCE_S / CADENCE_S)

        print(f"\nturb={turb_frac:.0%}  SNR≈{snr:.1f}:1  (n_true={len(true_locs)})")
        print("  Running MASS/SMASH ...")
        ms_peaks   = _run_ms(Bz, times)
        print("  Running PVI methods ...")
        pvi_peaks  = _run_pvi(B_vec, times, projective=False)
        ppvi_peaks = _run_pvi(B_vec, times, projective=True)

        ms_r   = evaluate(ms_peaks,   true_locs, tol)
        pvi_r  = evaluate(pvi_peaks,  true_locs, tol)
        ppvi_r = evaluate(ppvi_peaks, true_locs, tol)

        key = f'{turb_frac:.2f}'
        results['rotation'][key] = {
            'turb_frac': float(turb_frac),
            'snr':       float(snr),
            'n_true':    int(len(true_locs)),
            'mass_smash': ms_r,
            'pvi':        pvi_r,
            'proj_pvi':   ppvi_r,
        }

        print(f"  MASS/SMASH    P={ms_r['precision']:.3f}  R={ms_r['recall']:.3f}"
              f"  F1={ms_r['f1']:.3f}  (det={ms_r['n_det']})")
        print(f"  Std PVI       P={pvi_r['precision']:.3f}  R={pvi_r['recall']:.3f}"
              f"  F1={pvi_r['f1']:.3f}  (det={pvi_r['n_det']})")
        print(f"  Proj PVI      P={ppvi_r['precision']:.3f}  R={ppvi_r['recall']:.3f}"
              f"  F1={ppvi_r['f1']:.3f}  (det={ppvi_r['n_det']})")

    out_json = 'outputs/proj_pvi_tandem_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    _plot(results)
    return results


def _plot(results):
    turb_fracs = sorted(results['polarity'][k]['turb_frac']
                        for k in results['polarity'])
    x_labels = [f'{f:.0%}' for f in turb_fracs]
    snr_vals  = [1.0 / f for f in turb_fracs]

    rot_fracs  = sorted(results['rotation'][k]['turb_frac']
                        for k in results['rotation'])
    xr_labels  = [f'{f:.0%}' for f in rot_fracs]
    snr_r_vals = [1.0 / f for f in rot_fracs]

    styles = {
        'mass_smash': ('navy',      'D-',  'MASS/SMASH'),
        'pvi':        ('coral',     's--', 'Standard PVI'),
        'proj_pvi':   ('seagreen',  'o:',  'Projective PVI'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    bench_meta = [
        ('polarity', turb_fracs, x_labels, snr_vals,
         'Polarity-reversal events (B→−B)'),
        ('rotation', rot_fracs,  xr_labels, snr_r_vals,
         'Pure-rotation events (90°, no sign flip)'),
    ]

    for row, (bench, fracs, xl, sv, title) in enumerate(bench_meta):
        for col, metric in enumerate(['precision', 'recall', 'f1']):
            ax = axes[row, col]
            for key, (color, ls, label) in styles.items():
                vals = []
                for f in fracs:
                    k = f'{f:.2f}'
                    if k in results[bench]:
                        vals.append(results[bench][k][key][metric])
                    else:
                        vals.append(float('nan'))
                ax.plot(xl[:len(vals)], vals, ls, color=color,
                        label=label, lw=2, ms=7)
            ax.set_ylim(-0.02, 1.08)
            ax.set_xlabel('Turbulence (× B₀)', fontsize=9)
            ax.set_ylabel(metric.capitalize(), fontsize=9)
            ax.set_title(f'{metric.capitalize()} — {title}', fontsize=8.5)
            if col == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', labelrotation=30)

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(range(len(fracs)))
            ax2.set_xticklabels([f'{s:.0f}:1' for s in sv], fontsize=6)
            ax2.set_xlabel('SNR', fontsize=7)

    fig.suptitle('Projective PVI vs Standard PVI vs MASS/SMASH\n'
                 '7-day synthetic, seed=42',
                 fontsize=12)
    plt.tight_layout()
    path = 'outputs/fig_proj_pvi_tandem.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    run()
