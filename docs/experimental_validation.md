# Experimental Validation

This document makes the claims in the README fully reproducible.

## Goals & Hypotheses

- **H1 (MDL reduction):** MASS achieves lower description length (MDL, in bits) than non–seam-aware baselines on signals with a seam.
- **H2 (phase boundary):** There exists a noise-to-signal ratio threshold \(k^*\approx 0.721\) above which seam-aware modeling ceases to outperform the baseline on average.
- **H3 (localization):** The primary seam index \(\hat{t}\) is localized within ±3 samples of the ground truth on synthetic data with SNR ≥ 6 dB.

## Environment

- Python 3.9–3.12
- OS: Linux/macOS/Windows
- Create the environment:
  ```bash
  python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install -U pip
  pip install -e .
  pip install pytest matplotlib numpy scipy pandas
  ```

## Datasets

### Synthetic (canonical)

A length-N sinusoid with a sign-flip seam at index (t_0) and additive Gaussian noise:
[
\(x_t = \sin(2\pi f t) + \epsilon_t,\quad\)
\(y_t = \begin{cases}\
 x_t, & t < t_0 \\
 -x_t, & t \ge t_0\
\end{cases},\quad\)
\(\epsilon_t \sim \mathcal{N}(0, \sigma^2)\)
]

* Defaults: `N=4096`, `f=3/N`, `t0=N//2`, `seed=1337`.
* SNR sweep via \(\mathrm{SNR} = 10 \log_{10}(\mathrm{var}(x)/\sigma^2)\).

### (Optional) Real-world

If included, specify source, license, preprocessing, and how ground-truth seams are defined. Otherwise mark as “future work”.

## Baselines

* **Fourier (no seam):** Stationary Fourier model over the full series.
* **Piecewise Fourier (oracle):** Same as Fourier but split at true (t_0) (upper bound).
* *(Optional for robustness) AR/ARIMA using `statsmodels` with AIC/BIC order selection.*

## Metrics

* **MDL (bits):** total bits for model + residuals; reported mean ± 95% CI across seeds.
* **ΔMDL (bits):** baseline MDL − MASS MDL (positive is better).
* **Localization error (samples):** (|\hat{t} - t_0|).
* **Runtime (ms):** wall-clock per run for N=4096.

## Protocol

1. **Single-seam study (SNR sweep).**
   For SNR ∈ {0, 2, 4, 6, 8, 10, 12} dB and seeds {1..30}, generate synthetic series, run MASS and baselines, compute MDL and localization.
2. **Phase boundary estimate.**
   Fit a logistic or piecewise model to (P(ΔMDL > 0)) vs. SNR and report (k^*) where probability crosses 0.5.
3. **Ablations.**
   Toggle atoms (sign-flip, time-reversal, normalization) to quantify contribution to ΔMDL.
4. **Statistics.**
   Paired t-test on MDL (MASS vs baseline) across seeds per SNR; report p-values with Holm correction.

## Reproducibility: Commands

> All commands write CSVs to `artifacts/` and figures to `figures/`.

```bash
# 1) Quick smoke test & demo plot
python -m seamaware.cli.demo --N 4096 --snr_db 8 --seed 1337 --save figures/demo.png

# 2) SNR sweep (edit ranges as needed)
python -m seamaware.experiments.snr_sweep \
  --snr_db_list 0 2 4 6 8 10 12 \
  --seeds 1 30 \
  --N 4096 --freq 3 --t0 half \
  --out_dir artifacts/snr_sweep

# 3) Phase boundary estimation
python -m seamaware.analysis.phase_boundary \
  --csv artifacts/snr_sweep/results.csv \
  --out_fig figures/phase_boundary.png \
  --out_json artifacts/phase_boundary.json

# 4) Ablations
python -m seamaware.experiments.ablation \
  --snr_db 8 --seeds 1 30 \
  --atoms sign_flip time_reversal normalize \
  --out_dir artifacts/ablation
```

> If the `experiments/*` modules don’t exist yet, add minimal scripts under `seamaware/experiments/` that call your public API and write tidy CSVs:
> `snr_sweep.py`, `phase_boundary.py`, `ablation.py`.

## Results

### MDL vs SNR (N=4096)

| SNR (dB) | Baseline MDL (bits) | MASS MDL (bits) | ΔMDL (bits) | p-value |
| -------: | ------------------: | --------------: | ----------: | ------: |
|        0 |               *tbd* |           *tbd* |       *tbd* |   *tbd* |
|        2 |                     |                 |             |         |
|        4 |                     |                 |             |         |
|        6 |                     |                 |             |         |
|        8 |                     |                 |             |         |
|       10 |                     |                 |             |         |
|       12 |                     |                 |             |         |

### Phase Boundary

* Estimated (k^*): ***tbd*** (95% CI: *tbd*).
* Figure: `figures/phase_boundary.png`.

### Localization (SNR ≥ 6 dB)

| SNR (dB) |  mean | median | 95th % |
| -------: | ----: | -----: | -----: |
|        6 | *tbd* |  *tbd* |  *tbd* |
|        8 | *tbd* |  *tbd* |  *tbd* |
|       10 | *tbd* |  *tbd* |  *tbd* |
|       12 | *tbd* |  *tbd* |  *tbd* |

## Figures

1. **Demo reconstruction** (`figures/demo.png`): original, seam-aware reconstruction, residuals, detected seam(s).
2. **MDL vs SNR** (`figures/mdl_vs_snr.png`): means ± 95% CI.
3. **Phase boundary** (`figures/phase_boundary.png`): probability of ΔMDL>0 with fit and (k^*).
4. **Ablation bars** (`figures/ablation.png`): ΔMDL per atom toggle.

## Limitations

* Gaussian noise assumption in MDL derivation.
* Single seam; multiseam scenarios left for future work.
* Sensitivity when SNR < (k^*).

## How to Regenerate Everything

```bash
rm -rf artifacts figures
# Run 1→4 from "Reproducibility: Commands"
```
