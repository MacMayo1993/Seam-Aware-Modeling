"""
Prediction 1 Test: ℤ₂ SEAM Framework
======================================
Tests whether MASS/SMASH MDL-selected current sheets show a statistically
significant excess of ~π (180°) magnetic field rotation angles compared
to generic PVI-detected sheets.

Physics motivation:
    The ℤ₂ bundle map φ(f)(p,q) = exp(iπ(1-f(p,q))/2) maps orientation
    defects to π-flux in the effective EM connection. If topological
    protection is real, seams should preferentially show clean sign flips
    (ω ≈ 180°) rather than arbitrary rotations. Standard MHD turbulence
    predicts no such quantization.

Three possible outcomes:
    SIGNAL:    π-excess ratio > 1.3x, surrogate p < 0.05 → topological imprint
    NULL:      KS p > 0.05 → no distributional difference, framework needs revision
    AMBIGUOUS: Distributions differ but not at π → something different, not predicted
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp

# Window sizes to test — kinetic-scale sheets at 10s, MHD-scale at 120s
WINDOW_SIZES_S = [10, 30, 60, 120]
PI_WINDOW_DEG = 20   # degrees within 180° counted as π-excess
EPS = 1e-12


def compute_rotation_angles(B_vec, peaks, window_s=60, dt=3.0):
    """
    Compute magnetic field rotation angle across each detected crossing.

    For each peak index p:
        B_before = mean(B_vec[p-w : p])
        B_after  = mean(B_vec[p : p+w])
        ω = arccos(B̂_before · B̂_after)   in degrees

    Args:
        B_vec:    (N, 3) array of magnetic field vectors
        peaks:    1D array of sample indices for detected crossings
        window_s: averaging window on each side in seconds
        dt:       data cadence in seconds

    Returns:
        angles: 1D array of rotation angles in degrees (NaN where window fails)
    """
    w = max(int(window_s / dt), 1)
    N = len(B_vec)
    angles = []

    for p in peaks:
        p = int(p)
        if p < w or p > N - w:
            angles.append(np.nan)
            continue

        B_before = np.mean(B_vec[p - w : p], axis=0)
        B_after  = np.mean(B_vec[p     : p + w], axis=0)

        norm_before = np.linalg.norm(B_before)
        norm_after  = np.linalg.norm(B_after)

        if norm_before < EPS or norm_after < EPS:
            angles.append(np.nan)
            continue

        cos_angle = np.dot(B_before, B_after) / (norm_before * norm_after)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))

    return np.array(angles)


def run_single_window_test(pvi_angles, ms_angles):
    """
    Run statistical tests comparing two rotation angle distributions.

    Tests:
        KS test:  are the full distributions different?
        χ² test:  is the π-fraction (angles > 160°) different?

    Returns dict of test statistics, or None if insufficient data.
    """
    pvi_clean = pvi_angles[~np.isnan(pvi_angles)]
    ms_clean  = ms_angles[~np.isnan(ms_angles)]

    if len(pvi_clean) < 5 or len(ms_clean) < 5:
        return None

    ks_stat, ks_p = ks_2samp(pvi_clean, ms_clean)

    threshold = 180 - PI_WINDOW_DEG
    pvi_pi_frac = np.mean(pvi_clean > threshold)
    ms_pi_frac  = np.mean(ms_clean  > threshold)

    pvi_pi_count = int(np.sum(pvi_clean > threshold))
    ms_pi_count  = int(np.sum(ms_clean  > threshold))

    contingency = [
        [ms_pi_count,  max(len(ms_clean)  - ms_pi_count, 0)],
        [pvi_pi_count, max(len(pvi_clean) - pvi_pi_count, 0)],
    ]
    try:
        chi2, chi2_p, _, _ = chi2_contingency(contingency)
    except ValueError:
        chi2, chi2_p = np.nan, 1.0

    pi_excess_ratio = ms_pi_frac / (pvi_pi_frac + EPS)

    return {
        'n_pvi':            len(pvi_clean),
        'n_ms':             len(ms_clean),
        'ks_stat':          float(ks_stat),
        'ks_p':             float(ks_p),
        'pvi_pi_fraction':  float(pvi_pi_frac),
        'ms_pi_fraction':   float(ms_pi_frac),
        'pi_excess_ratio':  float(pi_excess_ratio),
        'chi2_p':           float(chi2_p),
    }


def phase_randomized_surrogate(B_vec, peaks, n_surrogates=200, dt=3.0, window_s=60):
    """
    Phase-randomized surrogate null model.

    Randomizes Fourier phases of B_vec while preserving the power spectrum,
    then recomputes rotation angles at the same peak locations. Any π-excess
    surviving this test cannot be explained by spectral properties alone.

    Args:
        B_vec:        (N, 3) magnetic field
        peaks:        sample indices to evaluate (ms_peaks)
        n_surrogates: number of surrogate realizations
        dt:           cadence in seconds
        window_s:     averaging window in seconds

    Returns:
        surrogate_pi_fracs: array of π-fractions from surrogate realizations
    """
    N = len(B_vec)
    surrogate_pi_fracs = []
    threshold = 180 - PI_WINDOW_DEG

    rng = np.random.default_rng(0)

    for _ in range(n_surrogates):
        B_surr = np.zeros_like(B_vec)
        for comp in range(3):
            fft_coeffs = np.fft.rfft(B_vec[:, comp])
            random_phases = rng.uniform(0, 2 * np.pi, len(fft_coeffs))
            fft_surr = np.abs(fft_coeffs) * np.exp(1j * random_phases)
            B_surr[:, comp] = np.fft.irfft(fft_surr, n=N)

        angles = compute_rotation_angles(B_surr, peaks, window_s=window_s, dt=dt)
        angles = angles[~np.isnan(angles)]
        surrogate_pi_fracs.append(
            float(np.mean(angles > threshold)) if len(angles) > 0 else 0.0
        )

    return np.array(surrogate_pi_fracs)


def run_prediction1_test(B_vec, pvi_peaks, ms_peaks, dt=3.0, save_dir='outputs/'):
    """
    Full Prediction 1 test across all window sizes with surrogate null model.

    Saves:
        outputs/fig3_rotation_angles.png  — 2x2 subplot, one per window size
        outputs/prediction1_results.json  — all numerical results

    Returns:
        all_results: dict keyed by window size string
        verdict:     'SIGNAL' | 'NULL' | 'AMBIGUOUS'
    """
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    signal_windows = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    bins = np.linspace(0, 180, 37)  # 5° bins

    for i, window_s in enumerate(WINDOW_SIZES_S):
        ax = axes[i]
        print(f"  Window {window_s}s: computing angles ...")

        pvi_angles = compute_rotation_angles(B_vec, pvi_peaks, window_s=window_s, dt=dt)
        ms_angles  = compute_rotation_angles(B_vec, ms_peaks,  window_s=window_s, dt=dt)

        stats = run_single_window_test(pvi_angles, ms_angles)
        if stats is None:
            all_results[f'window_{window_s}s'] = {'error': 'insufficient data'}
            ax.set_title(f'Window={window_s}s | insufficient data')
            continue

        print(f"    Running {100} surrogates ...")
        surrogate_fracs = phase_randomized_surrogate(
            B_vec, ms_peaks, n_surrogates=100, dt=dt, window_s=window_s
        )
        surrogate_p = float(np.mean(surrogate_fracs >= stats['ms_pi_fraction']))
        stats['surrogate_p'] = surrogate_p

        is_signal = (
            stats['pi_excess_ratio'] > 1.3
            and stats['chi2_p'] < 0.05
            and surrogate_p < 0.05
        )
        stats['signal'] = is_signal
        if is_signal:
            signal_windows.append(window_s)

        all_results[f'window_{window_s}s'] = stats

        pvi_clean = pvi_angles[~np.isnan(pvi_angles)]
        ms_clean  = ms_angles[~np.isnan(ms_angles)]

        ax.hist(pvi_clean, bins=bins, density=True, alpha=0.5,
                color='coral',     label=f'PVI (n={len(pvi_clean)})')
        ax.hist(ms_clean,  bins=bins, density=True, alpha=0.5,
                color='steelblue', label=f'MASS/SMASH (n={len(ms_clean)})')
        ax.axvline(180 - PI_WINDOW_DEG, color='black', ls='--', lw=1, alpha=0.4)

        ax.set_title(
            f'Window={window_s}s | π-ratio={stats["pi_excess_ratio"]:.2f}x'
            f' | surr-p={surrogate_p:.3f}',
            fontsize=9,
        )
        ax.set_xlabel('Rotation angle ω (°)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

        if is_signal:
            ax.text(0.02, 0.95, '★ SIGNAL', transform=ax.transAxes,
                    color='darkgreen', fontsize=10, fontweight='bold', va='top')

        print(f"    π-ratio={stats['pi_excess_ratio']:.2f}x  "
              f"χ²p={stats['chi2_p']:.4f}  surr-p={surrogate_p:.4f}"
              f"  {'★ SIGNAL' if is_signal else ''}")

    fig.suptitle(
        'Current Sheet Rotation Angles: MASS/SMASH vs PVI\n'
        'Prediction 1 Test — ℤ₂ SEAM Framework',
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'fig3_rotation_angles.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")

    # Overall verdict
    if len(signal_windows) >= 2:
        verdict = 'SIGNAL'
    elif len(signal_windows) == 1:
        verdict = 'AMBIGUOUS'
    else:
        any_ks_sig = any(
            r.get('ks_p', 1.0) < 0.05
            for r in all_results.values()
            if isinstance(r, dict) and 'ks_p' in r
        )
        verdict = 'AMBIGUOUS' if any_ks_sig else 'NULL'

    all_results['verdict'] = verdict
    all_results['signal_windows_s'] = signal_windows

    json_path = os.path.join(save_dir, 'prediction1_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")

    return all_results, verdict


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'outputs'))

    B_full = np.load('outputs/wind_mfi_B.npy')
    B_vec  = B_full[:, :3]
    times  = np.load('outputs/wind_mfi_times.npy')
    pvi_peaks = np.load('outputs/catalog_peaks.npy')
    ms_peaks  = np.load('outputs/ms_peaks.npy').astype(int)

    dt = float(np.median(np.diff(times)))

    print(f"B: {B_vec.shape}  PVI peaks: {len(pvi_peaks)}  MS peaks: {len(ms_peaks)}")
    results, verdict = run_prediction1_test(B_vec, pvi_peaks, ms_peaks, dt=dt)

    print(f"\n=== PREDICTION 1: Rotation Angle Test ===")
    print(f"Verdict: {verdict}")
    print(f"Signal windows: {results.get('signal_windows_s', [])}")
    for key in [f'window_{w}s' for w in WINDOW_SIZES_S]:
        r = results.get(key, {})
        if 'pi_excess_ratio' in r:
            print(f"  {key}: π-ratio={r['pi_excess_ratio']:.2f}x  "
                  f"χ²p={r['chi2_p']:.4f}  surr-p={r['surrogate_p']:.4f}"
                  f"  {'★' if r.get('signal') else ''}")
