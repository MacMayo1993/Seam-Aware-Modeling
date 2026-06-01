"""
Run MASS/SMASH + PVI + baseline on real Wind/MFI data.

Usage:
    # With a locally downloaded CDF:
    python run_real_data.py --cdf /path/to/wi_h0_mfi_20010301_v05.cdf

    # Without a CDF (falls back to synthetic):
    python run_real_data.py

Download instructions for real data:
    1. Go to https://cdaweb.gsfc.nasa.gov/cgi-bin/eval2.cgi
    2. Dataset: WI_H0_MFI
    3. Time range: 2001-03-01 to 2001-03-31
    4. Variable: BGSE  (magnetic field vector in GSE, 3-sec cadence)
    5. Output format: CDF  (or try "Download All Files" for the full month)
    6. Save the CDF file(s) and pass the path via --cdf

    Alternatively, install cdflib and PySPEDAS:
        pip install cdflib pyspedas
    Then just run without --cdf; PySPEDAS will download automatically.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.dirname(__file__))

from fetch_data import fetch_wind_mfi
from catalog import compute_pvi, get_pvi_events
from baseline import baseline_detector


def run_ms_chunked(signal, times, gain_threshold=20.0, verbose=True):
    from mass_smash import MASSSMASHConfig, run_mass_smash
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
        if verbose and (i + 1) % 500 == 0:
            print(f"  MASS/SMASH chunk {i+1}/{len(starts)}")

    mask  = (hit_count >= 1) & (mdl_gain > gain_threshold)
    cands = np.where(mask)[0]
    if len(cands) == 0:
        return np.array([], dtype=int), hit_count, mdl_gain

    scores = mdl_gain[cands]
    order  = np.argsort(scores)[::-1]
    kept   = []
    for pos in cands[order]:
        if all(abs(pos - k) >= min_sep_global for k in kept):
            kept.append(int(pos))

    return np.array(sorted(kept), dtype=int), hit_count, mdl_gain


def plot_overview(times, Bz, B_mag, ms_peaks, pvi_peaks, bl_peaks, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    t_days = (times - times[0]) / 86400.0

    axes[0].plot(t_days, Bz, lw=0.4, color='steelblue', label='Bz')
    for p in ms_peaks:
        axes[0].axvline(t_days[p], color='red', lw=0.7, alpha=0.8)
    axes[0].set_ylabel('Bz (nT)')
    axes[0].set_title('Bz with MASS/SMASH detections (red)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_days, Bz, lw=0.4, color='steelblue', alpha=0.5)
    for p in pvi_peaks:
        axes[1].axvline(t_days[p], color='coral', lw=0.7, alpha=0.8)
    axes[1].set_ylabel('Bz (nT)')
    axes[1].set_title('Bz with PVI detections (coral)')

    axes[2].plot(t_days, B_mag, lw=0.4, color='mediumseagreen', label='|B|')
    for p in bl_peaks:
        axes[2].axvline(t_days[p], color='purple', lw=0.7, alpha=0.6)
    axes[2].set_ylabel('|B| (nT)')
    axes[2].set_xlabel('Days from start')
    axes[2].set_title('|B| with Baseline detections (purple)')

    plt.suptitle(
        f'Wind/MFI detections — MASS/SMASH: {len(ms_peaks)}, '
        f'PVI: {len(pvi_peaks)}, Baseline: {len(bl_peaks)}',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved overview plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cdf', default=None,
                        help='Path to local Wind/MFI CDF file')
    parser.add_argument('--gain-threshold', type=float, default=20.0,
                        help='MDL gain threshold for MASS/SMASH (default 20.0)')
    parser.add_argument('--pvi-threshold', type=float, default=3.0,
                        help='PVI detection threshold (default 3.0)')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)

    print("Loading Wind/MFI data ...")
    times, Bx, By, Bz, B_mag = fetch_wind_mfi(cdf_path=args.cdf)
    B_vec = np.stack([Bx, By, Bz], axis=1)
    n_days = (times[-1] - times[0]) / 86400.0
    dt = float(np.median(np.diff(times)))
    print(f"  {len(times)} samples, {n_days:.1f} days, dt={dt:.1f}s")
    print(f"  |B| mean={np.mean(B_mag):.2f} nT, std={np.std(B_mag):.2f} nT")
    print(f"  Bz  range=[{np.min(Bz):.1f}, {np.max(Bz):.1f}] nT")

    print("\nRunning PVI ...")
    pvi = compute_pvi(B_vec)
    pvi_events, pvi_peaks = get_pvi_events(pvi, times,
                                            threshold=args.pvi_threshold,
                                            min_separation_s=300)
    print(f"  PVI: {len(pvi_peaks)} detections (threshold={args.pvi_threshold})")

    print("\nRunning Baseline ...")
    bl_peaks, _ = baseline_detector(B_mag, times, threshold_sigma=2.5)
    print(f"  Baseline: {len(bl_peaks)} detections")

    print("\nRunning MASS/SMASH (this may take a few minutes) ...")
    ms_peaks, hit_count, mdl_gain = run_ms_chunked(Bz, times,
                                                    gain_threshold=args.gain_threshold)
    print(f"  MASS/SMASH: {len(ms_peaks)} detections (gain_threshold={args.gain_threshold})")

    # Save results
    results = {
        'n_samples': int(len(times)),
        'n_days': float(n_days),
        'cadence_s': float(dt),
        'data_source': 'real_cdf' if args.cdf else 'synthetic_fallback',
        'mass_smash': {
            'n_detections': int(len(ms_peaks)),
            'gain_threshold': float(args.gain_threshold),
            'peak_indices': ms_peaks.tolist(),
        },
        'pvi': {
            'n_detections': int(len(pvi_peaks)),
            'threshold': float(args.pvi_threshold),
            'peak_indices': pvi_peaks.tolist(),
        },
        'baseline': {
            'n_detections': int(len(bl_peaks)),
            'threshold_sigma': 2.5,
            'peak_indices': bl_peaks.tolist(),
        },
    }
    out_json = 'outputs/real_data_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    np.save('outputs/real_ms_peaks.npy', ms_peaks)
    np.save('outputs/real_pvi_peaks.npy', pvi_peaks)
    np.save('outputs/real_bl_peaks.npy', bl_peaks)
    np.save('outputs/real_ms_hit_count.npy', hit_count)
    np.save('outputs/real_ms_mdl_gain.npy', mdl_gain)

    # Overview plot
    plot_overview(times, Bz, B_mag, ms_peaks, pvi_peaks, bl_peaks,
                  'outputs/fig_real_data_overview.png')

    print("\n=== SUMMARY ===")
    print(f"  Data:        {'Real Wind/MFI CDF' if args.cdf else 'Synthetic fallback'}")
    print(f"  Duration:    {n_days:.1f} days  ({len(times)} samples at {dt:.0f}s)")
    print(f"  MASS/SMASH:  {len(ms_peaks)} detections")
    print(f"  PVI:         {len(pvi_peaks)} detections")
    print(f"  Baseline:    {len(bl_peaks)} detections")
    if args.cdf:
        # Expected: ~120 current sheets in 30-day March 2001 interval
        # (literature: ~4/day × 30 days, but varies; ~80-150 is reasonable)
        print(f"\n  Literature sanity check: expect ~80-150 sector boundary")
        print(f"  crossings in 30-day Wind/MFI March 2001 data.")
        print(f"  (Osman et al. 2014 find ~4/day near solar max)")


if __name__ == '__main__':
    main()
