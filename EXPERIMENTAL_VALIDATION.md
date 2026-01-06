# Experimental Validation Protocol for SeamAware Framework

**Version:** 0.1.0  
**Date:** January 6, 2026  
**Goal:** Provide a clear, reproducible protocol to validate core claims: theoretical k* threshold emergence, flip atom correctness, MDL stability, and compression gains on regime-switching signals.

This document defines standardized hypotheses, environment, datasets, baselines, metrics, and step-by-step commands. Results are **not hardcoded**—run the experiments locally to generate CSVs/figures and fill placeholders below.

## 1. Goals & Hypotheses

| Hypothesis | Description | Success Criterion |
|------------|-------------|-------------------|
| H1: Flip Atom Correctness | All atoms are involutions (F²(x) = x) and apply correctly | ‖F²(x) - x‖ < 10⁻¹⁵; exact matches on synthetic cases |
| H2: MDL Numerical Stability | MDL computable and monotonic across variance ranges | No NaN/inf; better fits → lower MDL; negative MDL allowed for excellent fits |
| H3: k* Phase Transition | SeamAware outperforms baselines above theoretical k* ≈ 0.721 | Crossover SNR within 20% (fast Monte Carlo) or 15% (rigorous); acceptance fraction increases with SNR |
| H4: Compression Gains | 10–170% MDL reduction on regime-switching data | ΔMDL < 0 above threshold; Cohen's d > 2.0 for discrimination |

## 2. Environment

Reproduce with:
```bash
git clone https://github.com/MacMayo1993/Seam-Aware-Modeling.git
cd Seam-Aware-Modeling
pip install -e .  # Editable install
pip install pytest pytest-cov  # For testing/coverage
```

**Fixed Setup:**
- Python: 3.8+
- Key packages: numpy, scipy, matplotlib, pandas, statsmodels
- Random seeds: 42 (primary), 123 (secondary) for all stochastic runs

## 3. Datasets

All synthetic (generated on-the-fly for reproducibility):
- Base signal: Sine wave (freq=1–4 cycles, N=100–400)
- Regime switch: Sign flip at midpoint (t=N//2)
- Noise: Gaussian, targeted SNR ∈ [0.1, 2.0]
- Variations: Time reversal, variance jump (3–9×), polynomial trend (degree 1–2)

## 4. Baselines & Metrics

- **Baselines**: Fourier (K=5–20 components), Polynomial (degree 2–10)
- **SeamAware**: MASSFramework with auto flip atom selection
- **Primary Metric**: Minimum Description Length (MDL in bits; lower = better)
- **Secondary**: Effective SNR improvement, ΔMDL (SeamAware - Baseline), acceptance fraction

## 5. Reproducible Protocol

### 5.1 Quick Smoke Test (Verify Setup)
```bash
python -m seamaware.cli.demo  # Should show ~16–50% MDL gain + plot
pytest -q  # Should pass 25/25 tests
```

### 5.2 Core Unit/Integration Tests
```bash
pytest --cov=seamaware  # Run full suite + coverage
# Expected: 100% pass, coverage ~41% overall (~80% core)
```

### 5.3 Experiment Runs (Add Stubs if Needed)

> **Note:** If `seamaware/experiments/` module missing, add minimal stubs (e.g., snr_sweep.py) using patterns from cli/demo.py or scripts/generate_readme_visuals.py.

#### 5.3.1 SNR Sweep & Phase Boundary Estimation
```bash
python -m seamaware.experiments.snr_sweep \
  --N 200 \
  --trials 30 \  # Fast: ~10s
  --trials 100 \ # Rigorous: ~60s
  --seed 42 \
  --output artifacts/snr_sweep.csv

python -m seamaware.analysis.phase_boundary artifacts/snr_sweep.csv \
  --output figures/mdl_phase_transition.png
```

**Expected Outputs:**
- CSV columns: SNR, mean_ΔMDL, std_ΔMDL, accept_fraction
- Plot: Phase transition with shaded regions (red: no gain, green: gains)

**Result Placeholders (Fill After Run):**
- Fast crossover SNR: _____ (target <20% error from 0.721)
- Rigorous crossover: _____ (target <15% error)
- Max ΔMDL reduction: _____ %

#### 5.3.2 Ablation Studies
```bash
python -m seamaware.experiments.ablation \
  --atom SignFlip \  # Or TimeReversal, VarianceScale, etc.
  --output artifacts/ablation_signflip.csv
```

**Table Placeholder (Fill After Runs):**

| Flip Atom            | Params | Mean ΔMDL (High SNR) | Acceptance Fraction | Notes |
|----------------------|--------|----------------------|---------------------|-------|
| SignFlip            | 0      | _____                | _____               |       |
| TimeReversal        | 0      | _____                | _____               |       |
| VarianceScale       | 1      | _____                | _____               |       |
| PolynomialDetrend   | d+1    | _____                | _____               |       |

### 5.4 Generate Validation Figures
Use `scripts/generate_readme_visuals.py` for README assets, or extend for custom ablations.

## 6. Interpreting Results

- Compare your crossovers to theory (k* = 1/(2 ln 2) ≈ 0.7213)
- Larger N/more trials → lower error (asymptotic convergence)
- Report any deviations + seeds for debugging

This protocol is designed for automation (e.g., add to CI later). Contributions welcome to expand real-world datasets!

---

**Questions?** Open an issue or PR with your filled results/figures.
