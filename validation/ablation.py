"""
Ablation study: isolate the contribution of each MASS/SMASH component.

Four variants run on the same SNR sweep (7 turbulence levels, 28 events, seed=42):

  anti_only  -- Antipodal pre-filter candidates, accepted directly (NO MDL gate)
  rough_mdl  -- Roughness candidates → MDL gate (NO antipodal pre-filter)
  anti_mdl   -- Antipodal candidates → MDL gate (NO roughness)
  full       -- Both candidates → MDL gate  (full MASS/SMASH)

Comparisons answer three questions:
  anti_only vs anti_mdl   : what does MDL add on top of the pre-filter?
  rough_mdl vs full       : what does the antipodal pre-filter add when MDL is present?
  rough_mdl vs anti_mdl   : which candidate generator is better for MDL?
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.dirname(__file__))

from mass_smash import (
    MASSSMASHConfig, run_mass_smash,
    antipodal_symmetry_scanner, roughness_detector,
)
from snr_sweep import generate, evaluate, TURB_FRACS, CADENCE_S, TOLERANCE_S

# ── shared chunking params ──────────────────────────────────────────────────
DT          = CADENCE_S
WINDOW_S    = 600.0
STEP_S      = 300.0
MIN_SEP_S   = 300.0
GAIN_THR    = 20.0
ANT_THR     = 0.35
ROUGH_THR   = 0.20
ANT_WIN     = 20    # half-window samples


def _nms(peaks_scores, min_sep):
    """Greedy NMS: keep highest-score peak, suppress within min_sep."""
    if not peaks_scores:
        return []
    peaks_scores = sorted(peaks_scores, key=lambda x: x[1], reverse=True)
    kept = []
    for pos, sc in peaks_scores:
        if all(abs(pos - k) >= min_sep for k, _ in kept):
            kept.append((pos, sc))
    return sorted(kept, key=lambda x: x[0])


# ── Variant 1: antipodal-only (no MDL) ──────────────────────────────────────
def run_anti_only(signal, times):
    """Accept antipodal candidates directly; no MDL scoring."""
    dt = float(np.median(np.diff(times)))
    W  = int(WINDOW_S / dt)
    S  = int(STEP_S   / dt)
    min_sep_g = int(MIN_SEP_S / dt)
    N  = len(signal)

    # Score accumulation
    anti_score = np.zeros(N)

    for start in range(0, N - W, S):
        chunk = signal[start : start + W]
        cands = antipodal_symmetry_scanner(
            chunk,
            window_size=ANT_WIN * 2,
            threshold=ANT_THR,
            top_k=6,
            min_separation=int(60 / dt),
        )
        for local_idx, sc in cands:
            g = start + local_idx
            if 0 <= g < N:
                anti_score[g] += sc

    # Accept positions with any antipodal support
    mask  = anti_score > 0
    cands = [(int(p), float(anti_score[p])) for p in np.where(mask)[0]]
    kept  = _nms(cands, min_sep_g)
    return np.array([p for p, _ in kept], dtype=int)


# ── Variant 2: roughness → MDL (no antipodal pre-filter) ───────────────────
def run_rough_mdl(signal, times):
    """Use only roughness candidates, then gate with MDL."""
    dt = float(np.median(np.diff(times)))
    W  = int(WINDOW_S / dt)
    S  = int(STEP_S   / dt)
    min_sep_g = int(MIN_SEP_S / dt)
    N  = len(signal)

    cfg = MASSSMASHConfig(
        top_k_candidates=3,
        min_separation=int(60 / dt),
        antipodal_threshold=999.0,   # effectively disable antipodal (threshold > 1)
        roughness_threshold=ROUGH_THR,
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

    mask  = (hit_count >= 1) & (mdl_gain > GAIN_THR)
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


# ── Variant 3: antipodal → MDL (no roughness) ───────────────────────────────
def run_anti_mdl(signal, times):
    """Antipodal candidates only, then MDL gate."""
    dt = float(np.median(np.diff(times)))
    W  = int(WINDOW_S / dt)
    S  = int(STEP_S   / dt)
    min_sep_g = int(MIN_SEP_S / dt)
    N  = len(signal)

    cfg = MASSSMASHConfig(
        top_k_candidates=3,
        min_separation=int(60 / dt),
        antipodal_threshold=ANT_THR,
        roughness_threshold=999.0,   # effectively disable roughness
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

    mask  = (hit_count >= 1) & (mdl_gain > GAIN_THR)
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


# ── Variant 4: full MASS/SMASH ───────────────────────────────────────────────
def run_full(signal, times):
    """Full MASS/SMASH: both candidate generators + MDL gate."""
    dt = float(np.median(np.diff(times)))
    W  = int(WINDOW_S / dt)
    S  = int(STEP_S   / dt)
    min_sep_g = int(MIN_SEP_S / dt)
    N  = len(signal)

    cfg = MASSSMASHConfig(
        top_k_candidates=3,
        min_separation=int(60 / dt),
        antipodal_threshold=ANT_THR,
        roughness_threshold=ROUGH_THR,
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

    mask  = (hit_count >= 1) & (mdl_gain > GAIN_THR)
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
VARIANTS = {
    'anti_only': ('Antipodal-only\n(no MDL)',     run_anti_only, 'orchid',      'o-'),
    'rough_mdl': ('Roughness+MDL\n(no antipodal)','run_rough_mdl','coral',      's--'),
    'anti_mdl':  ('Antipodal+MDL\n(no roughness)','run_anti_mdl', 'steelblue',  '^:'),
    'full':      ('Full MASS/SMASH',               'run_full',     'navy',       'D-'),
}
RUNNER_MAP = {
    'anti_only': run_anti_only,
    'rough_mdl': run_rough_mdl,
    'anti_mdl':  run_anti_mdl,
    'full':      run_full,
}


def run():
    os.makedirs('outputs', exist_ok=True)
    results = {}

    for turb_frac in TURB_FRACS:
        snr = 1.0 / turb_frac
        print(f"\n{'='*55}")
        print(f"turb={turb_frac:.0%}  SNR≈{snr:.1f}:1")

        times, Bz, B_vec, B_mag, true_locs = generate(turb_frac)
        tol = int(TOLERANCE_S / CADENCE_S)

        key = f'{turb_frac:.2f}'
        results[key] = {'turb_frac': float(turb_frac), 'snr': float(snr),
                        'n_true': int(len(true_locs))}

        for vname, (label, _, color, ls) in VARIANTS.items():
            runner = RUNNER_MAP[vname]
            peaks  = runner(Bz, times)
            r      = evaluate(peaks, true_locs, tol)
            results[key][vname] = r
            print(f"  {vname:12s}  P={r['precision']:.3f}  R={r['recall']:.3f}"
                  f"  F1={r['f1']:.3f}  (det={r['n_det']})")

    with open('outputs/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved outputs/ablation_results.json")

    _plot(results)
    return results


def _plot(results):
    turb_fracs = sorted(v['turb_frac'] for v in results.values())
    x_labels   = [f'{f:.0%}' for f in turb_fracs]
    snr_vals   = [1.0 / f for f in turb_fracs]

    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (metric, title) in zip(axes, metrics):
        for vname, (label, _, color, ls) in VARIANTS.items():
            vals = [results[f'{f:.2f}'][vname][metric] for f in turb_fracs]
            ax.plot(x_labels, vals, ls, color=color,
                    label=label.replace('\n', ' '), lw=2, ms=7)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel('Turbulence amplitude (×B₀)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower left' if metric == 'precision' else 'best')
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelrotation=30)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(turb_fracs)))
        ax2.set_xticklabels([f'{s:.0f}:1' for s in snr_vals], fontsize=7)
        ax2.set_xlabel('SNR', fontsize=8)

    fig.suptitle('Ablation: precision/recall by component  '
                 '(7-day synthetic, 28 crossings, seed=42)',
                 fontsize=11)
    plt.tight_layout()
    path = 'outputs/fig_ablation.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    run()
