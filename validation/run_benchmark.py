"""Run MASS/SMASH vs baseline benchmark on Wind MFI data."""
import numpy as np
import sys
import os
import json

from seamaware.pipeline import MASSSMASHConfig, detect_seam_candidates, run_mass_smash

from baseline import baseline_detector


def run_mass_smash_on_signal(
    signal, times, window_size_s=600, step_size_s=300, B_vector=None
):
    """
    Run MASS/SMASH seam detector on a long continuous signal by chunking.

    Uses the full MDL-gated pipeline (run_mass_smash) per chunk so only seams
    that genuinely reduce description length are reported.

    Args:
        signal: 1D signal array used for MDL scoring (typically Bz)
        times: timestamps in seconds
        window_size_s: chunk size in seconds
        step_size_s: step between chunks
        B_vector: optional (T, 3) array of full magnetic field vector.
            When provided, candidate generation uses the vector-native
            antipodal scanner (vector_mode=True) which scores all three
            components jointly.  MDL scoring still uses the 1D `signal`.

    Returns:
        detected_peaks: global sample indices of detected seams
        scores: corresponding MDL-improvement scores
    """
    dt = np.median(np.diff(times))
    window_samples = int(window_size_s / dt)
    step_samples = int(step_size_s / dt)

    use_vector = B_vector is not None and B_vector.ndim == 2 and B_vector.shape[1] >= 3

    config = MASSSMASHConfig(
        top_k_candidates=3,
        min_separation=int(60 / dt),
        antipodal_threshold=0.35,
        roughness_threshold=0.20,
        max_seams=1,      # one seam per chunk keeps it focused
        alpha=2.0,        # standard MDL penalty
        verbose=False,
        include_mlp=False,  # skip MLP for speed
        vector_mode=use_vector,
    )

    # hit_count[i] = number of overlapping windows where MDL favoured a seam near i
    hit_count = np.zeros(len(signal), dtype=np.int32)
    mdl_gain = np.zeros(len(signal), dtype=np.float64)

    N = len(signal)
    n_chunks = max(1, (N - window_samples) // step_samples)
    print(f"  Processing {n_chunks} chunks of {window_size_s}s ...")

    for chunk_i, start in enumerate(range(0, N - window_samples, step_samples)):
        end = start + window_samples
        chunk = signal[start:end]
        chunk_B = B_vector[start:end] if use_vector else None

        best, all_sols = run_mass_smash(chunk, config, B_vector=chunk_B)

        # Only count detection if MDL selected ≥1 seam (beat the no-seam baseline)
        if best.n_seams > 0:
            # MDL gain = no-seam MDL − best-with-seam MDL (positive = improvement)
            no_seam_mdl = all_sols[-1].total_mdl  # worst = no seams (sorted asc by MDL)
            # Actually all_sols is sorted ascending by MDL, so no-seam is somewhere in there
            no_seam_sol = next((s for s in all_sols if s.n_seams == 0), None)
            gain = (no_seam_sol.total_mdl - best.total_mdl) if no_seam_sol else 1.0

            for local_idx in best.seams:
                global_idx = start + local_idx
                if 0 <= global_idx < N:
                    hit_count[global_idx] += 1
                    mdl_gain[global_idx] += max(0.0, gain)

        if (chunk_i + 1) % 200 == 0:
            print(f"    chunk {chunk_i+1}/{n_chunks} ...")

    # Save raw arrays for post-processing / threshold tuning
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/ms_hit_count.npy', hit_count)
    np.save('outputs/ms_mdl_gain.npy', mdl_gain)

    return _filter_detections(hit_count, mdl_gain, dt,
                              min_hits=1, gain_threshold=20.0, min_sep_s=300)


def _filter_detections(hit_count, mdl_gain, dt,
                       min_hits=1, gain_threshold=20.0, min_sep_s=300):
    """Apply hit count + gain filters then global NMS."""
    # Accept positions seen in >= min_hits windows AND with cumulative gain > threshold
    mask = (hit_count >= min_hits) & (mdl_gain > gain_threshold)
    candidate_positions = np.where(mask)[0]
    candidate_scores = mdl_gain[candidate_positions]

    if len(candidate_positions) == 0:
        # Relax both filters
        mask = hit_count >= 1
        candidate_positions = np.where(mask)[0]
        candidate_scores = mdl_gain[candidate_positions]

    if len(candidate_positions) == 0:
        return np.array([], dtype=int), np.array([])

    # Global NMS: score-ranked greedy with min_sep_s separation
    min_sep_global = int(min_sep_s / dt)
    order = np.argsort(candidate_scores)[::-1]
    sorted_pos = candidate_positions[order]
    sorted_sc = candidate_scores[order]

    kept_pos = []
    kept_sc = []
    for pos, sc in zip(sorted_pos, sorted_sc):
        if all(abs(pos - k) >= min_sep_global for k in kept_pos):
            kept_pos.append(int(pos))
            kept_sc.append(sc)

    if not kept_pos:
        return np.array([], dtype=int), np.array([])

    order2 = np.argsort(kept_pos)
    detected_peaks = np.array(kept_pos, dtype=int)[order2]
    scores = np.array(kept_sc)[order2]
    return detected_peaks, scores


def evaluate_detector(detected_peaks, catalog_peaks, tolerance_samples=10):
    """
    Compute precision, recall, F1 against PVI catalog ground truth.

    A detection is a true positive if within tolerance_samples of a catalog event.
    """
    if len(detected_peaks) == 0:
        return {
            'precision': 0, 'recall': 0, 'f1': 0,
            'tp': 0, 'fp': 0, 'fn': len(catalog_peaks),
            'n_detected': 0, 'n_catalog': len(catalog_peaks),
        }

    matched_catalog = set()
    tp = 0

    for d in detected_peaks:
        for i, c in enumerate(catalog_peaks):
            if abs(int(d) - int(c)) <= tolerance_samples and i not in matched_catalog:
                tp += 1
                matched_catalog.add(i)
                break

    fp = len(detected_peaks) - tp
    fn = len(catalog_peaks) - len(matched_catalog)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_detected': len(detected_peaks), 'n_catalog': len(catalog_peaks),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run MASS/SMASH benchmark on Wind MFI data')
    parser.add_argument('--cdf', default=None,
                        help='Path to a locally downloaded Wind/MFI CDF file. '
                             'If omitted, tries PySPEDAS then falls back to synthetic data.')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)

    # Load or generate data
    if args.cdf is not None or not (os.path.exists('outputs/wind_mfi_B.npy') and
                                     os.path.exists('outputs/wind_mfi_times.npy')):
        sys.path.insert(0, os.path.dirname(__file__))
        from fetch_data import fetch_wind_mfi
        print(f"{'Loading CDF: ' + args.cdf if args.cdf else 'Fetching Wind MFI data ...'}")
        times_raw, Bx_raw, By_raw, Bz_raw, Bmag_raw = fetch_wind_mfi(cdf_path=args.cdf)
        np.save('outputs/wind_mfi_times.npy', times_raw)
        np.save('outputs/wind_mfi_B.npy', np.stack([Bx_raw, By_raw, Bz_raw, Bmag_raw], axis=1))
        print(f"Cached {len(times_raw)} samples to outputs/")

    B = np.load('outputs/wind_mfi_B.npy')
    times = np.load('outputs/wind_mfi_times.npy')
    catalog_peaks = np.load('outputs/catalog_peaks.npy')

    Bz = B[:, 2]
    B_mag = B[:, 3]

    print("Running MASS/SMASH on Bz (vector-native candidate generation) ...")
    ms_peaks, ms_scores = run_mass_smash_on_signal(Bz, times, B_vector=B[:, :3])

    print("Running baseline detector ...")
    bl_peaks, dBdt = baseline_detector(B_mag, times)

    # Primary metric: 150-second tolerance (appropriate for sign-flip-center vs gradient-peak offset)
    # Secondary metric: 30-second strict tolerance
    ms_results = evaluate_detector(ms_peaks, catalog_peaks, tolerance_samples=50)
    bl_results = evaluate_detector(bl_peaks, catalog_peaks, tolerance_samples=50)
    ms_strict = evaluate_detector(ms_peaks, catalog_peaks, tolerance_samples=10)
    bl_strict = evaluate_detector(bl_peaks, catalog_peaks, tolerance_samples=10)

    results = {
        'mass_smash': ms_results,
        'baseline': bl_results,
        'mass_smash_strict': ms_strict,
        'baseline_strict': bl_strict,
        'tolerance_primary_samples': 50,
        'tolerance_strict_samples': 10,
    }
    with open('outputs/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.save('outputs/ms_peaks.npy', ms_peaks)
    np.save('outputs/bl_peaks.npy', bl_peaks)

    print("\n=== RESULTS (tolerance = 150 s) ===")
    print(f"MASS/SMASH:  P={ms_results['precision']:.3f}  R={ms_results['recall']:.3f}  F1={ms_results['f1']:.3f}  (n_det={ms_results['n_detected']}, n_cat={ms_results['n_catalog']})")
    print(f"Baseline:    P={bl_results['precision']:.3f}  R={bl_results['recall']:.3f}  F1={bl_results['f1']:.3f}  (n_det={bl_results['n_detected']}, n_cat={bl_results['n_catalog']})")
    print("\n=== RESULTS (strict 30 s) ===")
    print(f"MASS/SMASH:  P={ms_strict['precision']:.3f}  R={ms_strict['recall']:.3f}  F1={ms_strict['f1']:.3f}")
    print(f"Baseline:    P={bl_strict['precision']:.3f}  R={bl_strict['recall']:.3f}  F1={bl_strict['f1']:.3f}")

    # ── Prediction 1 ──────────────────────────────────────────────────────────
    # Use a lower gain threshold for the rotation-angle test to get enough
    # events for statistical power (precision/recall trade-off is irrelevant
    # for the angle distribution comparison).
    sys.path.insert(0, os.path.dirname(__file__))
    from rotation_angles import run_prediction1_test

    hit_count = np.load('outputs/ms_hit_count.npy')
    mdl_gain  = np.load('outputs/ms_mdl_gain.npy')
    dt_val    = float(np.median(np.diff(times)))
    ms_peaks_p1, _ = _filter_detections(hit_count, mdl_gain, dt_val,
                                         min_hits=1, gain_threshold=2.0, min_sep_s=300)
    print(f"\nPrediction 1 detection set: {len(ms_peaks_p1)} events "
          f"(gain>2 threshold, vs {len(ms_peaks)} for benchmark)")

    print("Running Prediction 1 (rotation angle test)...")
    p1_results, verdict = run_prediction1_test(
        B[:, :3], catalog_peaks, ms_peaks_p1, dt=dt_val
    )

    print(f"\n=== PREDICTION 1: Rotation Angle Test ===")
    print(f"Verdict: {verdict}")
    print(f"Signal windows: {p1_results.get('signal_windows_s', [])}")
    for key in [f'window_{w}s' for w in [10, 30, 60, 120]]:
        r = p1_results.get(key, {})
        if 'pi_excess_ratio' in r:
            print(f"  {key}: π-ratio={r['pi_excess_ratio']:.2f}x  "
                  f"χ²p={r['chi2_p']:.4f}  surr-p={r['surrogate_p']:.4f}  "
                  f"{'★' if r.get('signal') else ''}")
