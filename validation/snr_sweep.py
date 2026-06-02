"""
SNR sweep: detection performance vs turbulence amplitude.

Runs MASS/SMASH, PVI, and baseline across turbulence amplitudes
[5%, 10%, 15%, 20%, 30%, 40%, 50%] × B0 on 7-day synthetic signals
(~28 injected crossings each).

Expected:
  - At low turbulence (high SNR): all methods do well
  - As turbulence increases: gradient methods (PVI, baseline) accumulate
    false positives from turbulent spikes; MASS/SMASH precision degrades
    more slowly because the antipodal pre-filter rejects non-antipodal noise
  - At very high turbulence: MASS/SMASH recall drops (MDL gain falls below
    threshold) while precision holds — a different failure mode
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.dirname(__file__))

from mass_smash import MASSSMASHConfig, run_mass_smash
from catalog import compute_pvi, get_pvi_events
from baseline import baseline_detector

# ── Configuration ──────────────────────────────────────────────────────────────
TURB_FRACS  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_DAYS      = 7
CADENCE_S   = 3.0
B0          = 6.0
TOLERANCE_S = 150.0
SEED        = 42


# ── Signal generator ───────────────────────────────────────────────────────────

def _turb(N, rng, amp, idx=-5/3):
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1.0
    power = np.abs(freqs) ** idx
    power[0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, len(power))
    coeffs = np.sqrt(power) * np.exp(1j * phases)
    sig = np.fft.irfft(coeffs, n=N)
    return sig / (np.std(sig) + 1e-12) * amp


def generate(turb_frac, n_days=N_DAYS, seed=SEED):
    """
    Generate synthetic solar wind at a given turbulence fraction.
    Returns (times, Bz, B_vec, B_mag, injected_locs).
    Uses the same RNG sequence as the main benchmark so the injected
    crossing locations are identical across turbulence levels.
    """
    rng = np.random.default_rng(seed)
    N   = int(n_days * 86400 / CADENCE_S)
    times = np.arange(N, dtype=float) * CADENCE_S

    base_dir = np.array([0.20, 0.20, 0.96])
    base_dir /= np.linalg.norm(base_dir)

    n_sheets  = int(n_days * 24 / 6)
    min_sep   = int(3600 / CADENCE_S)
    sheet_locs = []
    attempts = 0
    while len(sheet_locs) < n_sheets and attempts < 100_000:
        attempts += 1
        loc = rng.integers(min_sep, N - min_sep)
        if all(abs(loc - s) > min_sep for s in sheet_locs):
            sheet_locs.append(int(loc))
    sheet_locs = sorted(sheet_locs)

    polarity = np.ones(N, dtype=float)
    current_pol = 1.0
    for loc in sheet_locs:
        current_pol *= -1.0
        polarity[loc:] = current_pol

    pol_smooth = polarity.copy()
    for loc in sheet_locs:
        w = rng.integers(5, 21)
        s, e = max(0, loc - w // 2), min(N, loc + w // 2)
        x = np.linspace(-3, 3, e - s)
        pol_smooth[s:e] = np.tanh(x) * np.sign(polarity[min(loc + 1, N - 1)])

    turb_amp = turb_frac * B0
    Bx = pol_smooth * B0 * base_dir[0] + _turb(N, rng, turb_amp)
    By = pol_smooth * B0 * base_dir[1] + _turb(N, rng, turb_amp)
    Bz = pol_smooth * B0 * base_dir[2] + _turb(N, rng, turb_amp)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_vec = np.stack([Bx, By, Bz], axis=1)

    return times, Bz, B_vec, B_mag, np.array(sheet_locs)


# ── MASS/SMASH chunked inference (no file I/O) ──────────────────────────────────

def run_ms(signal, times, gain_threshold=20.0):
    dt = float(np.median(np.diff(times)))
    window_samples = int(600 / dt)
    step_samples   = int(300 / dt)
    min_sep_global = int(300 / dt)
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

    starts = list(range(0, N - window_samples, step_samples))
    for i, start in enumerate(starts):
        chunk = signal[start : start + window_samples]
        best, all_sols = run_mass_smash(chunk, cfg)
        if best.n_seams > 0:
            no_seam = next((s for s in all_sols if s.n_seams == 0), None)
            gain = (no_seam.total_mdl - best.total_mdl) if no_seam else 1.0
            for local_idx in best.seams:
                g = start + local_idx
                if 0 <= g < N:
                    hit_count[g] += 1
                    mdl_gain[g]  += max(0.0, gain)
        if (i + 1) % 300 == 0:
            print(f"        chunk {i+1}/{len(starts)}")

    mask  = (hit_count >= 1) & (mdl_gain > gain_threshold)
    cands = np.where(mask)[0]
    if len(cands) == 0:
        return np.array([], dtype=int)

    scores = mdl_gain[cands]
    order  = np.argsort(scores)[::-1]
    kept   = []
    for pos in cands[order]:
        if all(abs(pos - k) >= min_sep_global for k in kept):
            kept.append(int(pos))
    return np.array(sorted(kept), dtype=int)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(peaks, true_locs, tol):
    peaks = np.array(peaks, dtype=int)
    tl    = np.array(true_locs, dtype=int)
    matched = set()
    tp = 0
    for d in peaks:
        for i, c in enumerate(tl):
            if abs(d - int(c)) <= tol and i not in matched:
                tp += 1; matched.add(i); break
    fp = len(peaks) - tp
    fn = len(tl) - len(matched)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'n_det': len(peaks)}


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('outputs', exist_ok=True)
    all_results = {}

    for turb_frac in TURB_FRACS:
        snr = 1.0 / turb_frac
        print(f"\n{'='*55}")
        print(f"turb={turb_frac:.0%} × B0   (SNR ≈ {snr:.1f}:1)")

        times, Bz, B_vec, B_mag, true_locs = generate(turb_frac)
        tol = int(TOLERANCE_S / CADENCE_S)
        dt  = CADENCE_S

        print("  PVI ...")
        pvi = compute_pvi(B_vec)
        _, pvi_peaks = get_pvi_events(pvi, times, threshold=3.0, min_separation_s=300)

        print("  Baseline ...")
        bl_peaks, _ = baseline_detector(B_mag, times, threshold_sigma=2.5)

        print("  MASS/SMASH ...")
        ms_peaks = run_ms(Bz, times, gain_threshold=20.0)

        ms_r  = evaluate(ms_peaks,  true_locs, tol)
        pvi_r = evaluate(pvi_peaks, true_locs, tol)
        bl_r  = evaluate(bl_peaks,  true_locs, tol)

        all_results[f'{turb_frac:.2f}'] = {
            'turb_frac':  float(turb_frac),
            'snr':        float(snr),
            'n_true':     int(len(true_locs)),
            'mass_smash': ms_r,
            'pvi':        pvi_r,
            'baseline':   bl_r,
        }

        print(f"  n_true={len(true_locs)}")
        print(f"  MASS/SMASH  P={ms_r['precision']:.3f}  R={ms_r['recall']:.3f}  "
              f"F1={ms_r['f1']:.3f}  (det={ms_r['n_det']})")
        print(f"  PVI         P={pvi_r['precision']:.3f}  R={pvi_r['recall']:.3f}  "
              f"F1={pvi_r['f1']:.3f}  (det={pvi_r['n_det']})")
        print(f"  Baseline    P={bl_r['precision']:.3f}  R={bl_r['recall']:.3f}  "
              f"F1={bl_r['f1']:.3f}  (det={bl_r['n_det']})")

    with open('outputs/snr_sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved outputs/snr_sweep_results.json")

    _plot(all_results)
    return all_results


def _plot(all_results):
    turb_fracs = sorted(v['turb_frac'] for v in all_results.values())
    snr_vals   = [1.0/f for f in turb_fracs]
    x_labels   = [f'{f:.0%}' for f in turb_fracs]

    styles = {
        'mass_smash': ('steelblue',       'o-',  'MASS/SMASH'),
        'pvi':        ('coral',           's--', 'PVI'),
        'baseline':   ('mediumseagreen',  '^:', 'Baseline'),
    }
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (metric, title) in zip(axes, metrics):
        for key, (color, ls, label) in styles.items():
            vals = [all_results[f'{f:.2f}'][key][metric] for f in turb_fracs]
            ax.plot(x_labels, vals, ls, color=color, label=label, lw=2, ms=7)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel('Turbulence amplitude (× B₀)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelrotation=30)

    # Secondary x-axis: SNR
    for ax in axes:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(turb_fracs)))
        ax2.set_xticklabels([f'{s:.0f}:1' for s in snr_vals], fontsize=7)
        ax2.set_xlabel('SNR', fontsize=8)

    fig.suptitle(f'Detection Performance vs SNR  ({N_DAYS}-day synthetic, '
                 f'{int(N_DAYS*24/6)} true crossings, tolerance={TOLERANCE_S:.0f}s)',
                 fontsize=11)
    plt.tight_layout()
    path = 'outputs/fig_snr_sweep.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    run()
