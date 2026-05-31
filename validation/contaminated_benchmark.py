"""
Contaminated benchmark: precision under non-antipodal confounding events.

Adds two types of confounders to the synthetic solar wind signal:
  Type A — Turbulent variance bursts: large |ΔBz|, no polarity change
  Type B — Partial (60°) rotations: real gradient, not a sign flip

Hypothesis:
  PVI and the baseline fire on confounders because they only measure
  gradient magnitude. MASS/SMASH's antipodal pre-filter scores
  corr(Bz[τ-w:τ], -Bz[τ:τ+w]) ≈ 0 for these events → rejects them.

Runs on a fresh 10-day synthetic signal (not the existing 30-day benchmark)
so we don't overwrite clean benchmark outputs.
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
N_DAYS          = 10
CADENCE_S       = 3.0
B0              = 6.0       # nT background
TURB_FRAC       = 0.30      # same as paper benchmark
N_CONFOUNDERS   = 20        # confounders of each type (A and B)
BURST_AMP       = 4.0       # nT; large enough that PVI > 3.0 fires
BURST_WIDTH_S   = 60        # seconds — comparable to transition width
TOLERANCE_S     = 150.0     # detection matching tolerance
MIN_SEP_S       = 600.0     # minimum separation between confounders and true crossings


# ── Synthetic data generation ──────────────────────────────────────────────────

def _turb(N, rng, amp, idx=-5/3):
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1.0
    power = np.abs(freqs) ** idx
    power[0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, len(power))
    coeffs = np.sqrt(power) * np.exp(1j * phases)
    sig = np.fft.irfft(coeffs, n=N)
    return sig / (np.std(sig) + 1e-12) * amp


def generate_clean_signal(n_days=N_DAYS, turb_frac=TURB_FRAC, seed=42):
    """Generate synthetic signal identical in structure to the main benchmark."""
    rng = np.random.default_rng(seed)
    N = int(n_days * 86400 / CADENCE_S)
    times = np.arange(N, dtype=float) * CADENCE_S

    base_dir = np.array([0.20, 0.20, 0.96])
    base_dir /= np.linalg.norm(base_dir)

    n_sheets = int(n_days * 24 / 6)
    min_sep = int(3600 / CADENCE_S)
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


# ── Confounder placement ────────────────────────────────────────────────────────

def place_confounders(N, true_locs, n_each, rng, min_sep=None):
    """Pick positions well-separated from true crossings and each other."""
    if min_sep is None:
        min_sep = int(MIN_SEP_S / CADENCE_S)
    margin = min_sep
    occupied = list(true_locs)
    locs = []
    attempts = 0
    while len(locs) < n_each and attempts < 500_000:
        attempts += 1
        loc = int(rng.integers(margin, N - margin))
        if all(abs(loc - o) > min_sep for o in occupied):
            locs.append(loc)
            occupied.append(loc)
    if len(locs) < n_each:
        print(f"  Warning: only placed {len(locs)}/{n_each} confounders")
    return sorted(locs)


# ── Confounder injection ────────────────────────────────────────────────────────

def inject_variance_bursts(Bz, B_vec, B_mag, locs, rng):
    """
    Type A: Turbulent variance burst at each loc.
    Large local |ΔBz| with no net polarity change — pure noise spike.
    """
    Bz_out  = Bz.copy()
    Bv_out  = B_vec.copy()
    N = len(Bz)
    width = int(BURST_WIDTH_S / CADENCE_S)
    for loc in locs:
        s = max(0, loc - width // 2)
        e = min(N, loc + width // 2)
        env = np.hanning(e - s)
        for c in range(3):
            noise = rng.normal(0, BURST_AMP, e - s) * env
            Bv_out[s:e, c] += noise
        Bz_out[s:e] = Bv_out[s:e, 2]
    B_mag_out = np.sqrt(np.sum(Bv_out**2, axis=1))
    return Bz_out, Bv_out, B_mag_out


def inject_bz_dips(Bz, B_vec, B_mag, locs):
    """
    Type B: Gaussian dip in Bz magnitude — large gradient, no sign flip.

    Bz drops to ~10% of its current value at each loc and returns,
    creating two gradient events (entry and exit of dip) that PVI detects
    but that have zero antipodal structure (Bz stays positive throughout).
    """
    Bz_out = Bz.copy()
    N = len(Bz)
    sigma = int(BURST_WIDTH_S / CADENCE_S / 4)  # ~15 samples
    for loc in locs:
        s = max(0, loc - 3 * sigma)
        e = min(N, loc + 3 * sigma)
        t = np.arange(e - s) - (e - s) / 2
        # Gaussian notch: scale Bz down by 90% at the center
        notch = 1.0 - 0.90 * np.exp(-t**2 / (2 * sigma**2))
        Bz_out[s:e] = Bz[s:e] * notch
    Bv_out = B_vec.copy()
    Bv_out[:, 2] = Bz_out
    B_mag_out = np.sqrt(np.sum(Bv_out**2, axis=1))
    return Bz_out, Bv_out, B_mag_out


# ── MASS/SMASH chunked inference (standalone, no file I/O) ─────────────────────

def run_ms_chunked(signal, times, gain_threshold=20.0):
    """Run MASS/SMASH chunked inference without touching outputs/ files."""
    dt = float(np.median(np.diff(times)))
    window_samples = int(600 / dt)
    step_samples   = int(300 / dt)
    min_sep_global = int(300 / dt)

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

    N = len(signal)
    hit_count = np.zeros(N, dtype=np.int32)
    mdl_gain  = np.zeros(N, dtype=np.float64)

    starts = range(0, N - window_samples, step_samples)
    n_chunks = len(range(0, N - window_samples, step_samples))
    print(f"    {n_chunks} chunks ...")

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
        if (i + 1) % 500 == 0:
            print(f"      chunk {i+1}/{n_chunks}")

    # NMS with gain threshold
    mask = (hit_count >= 1) & (mdl_gain > gain_threshold)
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

def evaluate(peaks, true_locs, conf_locs, tol):
    """
    Returns a dict with:
      tp, fp, fn, precision, recall, f1  — vs true crossings
      conf_hits  — number of confounder locs within tol of a detection
      conf_rate  — conf_hits / n_confounders
    """
    peaks = np.array(peaks, dtype=int)
    tl    = np.array(true_locs, dtype=int)
    cl    = np.array(conf_locs, dtype=int)

    matched_t = set()
    tp = 0
    for d in peaks:
        for i, c in enumerate(tl):
            if abs(d - int(c)) <= tol and i not in matched_t:
                tp += 1
                matched_t.add(i)
                break

    fp = len(peaks) - tp
    fn = len(tl) - len(matched_t)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    matched_c = set()
    conf_hits = 0
    for d in peaks:
        for i, c in enumerate(cl):
            if abs(d - int(c)) <= tol and i not in matched_c:
                # Don't double-count if it's also a TP
                if not any(abs(d - int(tc)) <= tol for tc in tl):
                    conf_hits += 1
                    matched_c.add(i)
                break

    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'n_det': int(len(peaks)),
        'precision': float(prec),
        'recall':    float(rec),
        'f1':        float(f1),
        'conf_hits': int(conf_hits),
        'conf_rate': float(conf_hits / max(len(cl), 1)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('outputs', exist_ok=True)
    rng_conf = np.random.default_rng(7)

    print("Generating clean 10-day synthetic signal ...")
    times, Bz, B_vec, B_mag, true_locs = generate_clean_signal()
    N   = len(times)
    dt  = CADENCE_S
    tol = int(TOLERANCE_S / dt)

    print(f"True crossings: {len(true_locs)}")

    # Place confounders well away from true crossings
    locs_A = place_confounders(N, true_locs, N_CONFOUNDERS, rng_conf)
    locs_B = place_confounders(N, np.concatenate([true_locs, locs_A]),
                               N_CONFOUNDERS, rng_conf)
    all_conf = sorted(locs_A + locs_B)
    print(f"Confounders:  {len(locs_A)} variance bursts (A) + {len(locs_B)} Bz dips (B)")

    # ── Inject confounders ──────────────────────────────────────────────────
    Bz_A, Bv_A, Bm_A = inject_variance_bursts(Bz, B_vec, B_mag, locs_A, rng_conf)
    # Apply Bz dips on top of burst-injected signal
    Bz_c, Bv_c, Bm_c = inject_bz_dips(Bz_A, Bv_A, Bm_A, locs_B)

    print("\n── Clean signal (no confounders) ──")
    all_results = {}

    for label, bz, bv, bm in [
        ('clean',         Bz,   B_vec, B_mag),
        ('contaminated',  Bz_c, Bv_c,  Bm_c),
    ]:
        print(f"\n=== {label.upper()} ===")

        print("  PVI ...")
        pvi = compute_pvi(bv)
        _, pvi_peaks = get_pvi_events(pvi, times, threshold=3.0, min_separation_s=300)

        print("  Baseline ...")
        bl_peaks, _ = baseline_detector(bm, times, threshold_sigma=2.5)

        print("  MASS/SMASH ...")
        ms_peaks = run_ms_chunked(bz, times, gain_threshold=20.0)

        conf_locs_for_eval = all_conf if label == 'contaminated' else []

        for name, peaks in [('MASS/SMASH', ms_peaks),
                             ('PVI',        pvi_peaks),
                             ('Baseline',   bl_peaks)]:
            r = evaluate(peaks, true_locs, conf_locs_for_eval, tol)
            all_results[f'{label}_{name}'] = r
            print(f"  {name:12s}: P={r['precision']:.3f}  R={r['recall']:.3f}  "
                  f"F1={r['f1']:.3f}  det={r['n_det']}", end='')
            if label == 'contaminated':
                print(f"  conf_hits={r['conf_hits']}/{len(all_conf)}  "
                      f"({r['conf_rate']:.0%})")
            else:
                print()

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        'n_true_crossings':   int(len(true_locs)),
        'n_confounders_A':    int(len(locs_A)),
        'n_confounders_B':    int(len(locs_B)),
        'n_confounders_total': int(len(all_conf)),
        'burst_amplitude_nT': float(BURST_AMP),
        'bz_dip_depth_pct': 90.0,
        'results': all_results,
    }
    with open('outputs/contaminated_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved outputs/contaminated_results.json")

    _plot(all_results, out)


def _plot(results, meta):
    """Summary figure: precision and confounder hit rate, clean vs contaminated."""
    detectors = ['MASS/SMASH', 'PVI', 'Baseline']
    colors = {'MASS/SMASH': 'steelblue', 'PVI': 'coral', 'Baseline': 'mediumseagreen'}
    x = np.arange(len(detectors))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: precision clean vs contaminated
    ax = axes[0]
    for i, det in enumerate(detectors):
        p_clean = results.get(f'clean_{det}',        {}).get('precision', 0)
        p_cont  = results.get(f'contaminated_{det}', {}).get('precision', 0)
        ax.bar(i - w/2, p_clean, w, color=colors[det], alpha=0.9,  label=det if i==0 else '')
        ax.bar(i + w/2, p_cont,  w, color=colors[det], alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(detectors, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Precision')
    ax.set_title('Precision\n(dark=clean  light=contaminated)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[d], label=d) for d in detectors],
              fontsize=8, loc='lower right')

    # Panel 2: confounder hit rate (contaminated only)
    ax = axes[1]
    for i, det in enumerate(detectors):
        r = results.get(f'contaminated_{det}', {})
        ax.bar(i, r.get('conf_rate', 0), color=colors[det], alpha=0.85)
        ax.text(i, r.get('conf_rate', 0) + 0.02,
                f"{r.get('conf_hits', 0)}/{meta['n_confounders_total']}",
                ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(detectors, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Confounder hit rate')
    ax.set_title(f"Confounder false alarm rate\n"
                 f"({meta['n_confounders_total']} non-antipodal events)")

    # Panel 3: F1 clean vs contaminated
    ax = axes[2]
    for i, det in enumerate(detectors):
        f_clean = results.get(f'clean_{det}',        {}).get('f1', 0)
        f_cont  = results.get(f'contaminated_{det}', {}).get('f1', 0)
        ax.bar(i - w/2, f_clean, w, color=colors[det], alpha=0.9)
        ax.bar(i + w/2, f_cont,  w, color=colors[det], alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(detectors, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel('F1')
    ax.set_title('F1 Score\n(dark=clean  light=contaminated)')

    fig.suptitle('Contaminated Benchmark: Non-Antipodal Confounders\n'
                 f"(Type A: variance bursts  |  Type B: Bz magnitude dips)",
                 fontsize=11)
    plt.tight_layout()
    path = 'outputs/fig_contaminated_benchmark.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    run()
