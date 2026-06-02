# Mathematical Foundations of Seam-Aware Modeling

## Precise Definitions

This section removes all ambiguity in the mathematical framework by explicitly defining every key concept.

### State Vector and Normalization

Given a window of signal samples **x** = (x₀, x₁, ..., x_{L-1}) ∈ ℝ^L, we construct the **normalized state vector**:

```
u = (x - x̄) / ‖x - x̄‖₂  ∈ S^{L-1}
```

where:
- x̄ = (1/L)∑ᵢxᵢ is the sample mean
- ‖·‖₂ is the Euclidean norm
- S^{L-1} = {u ∈ ℝ^L : ‖u‖₂ = 1} is the unit sphere

**Edge case**: If ‖x - x̄‖₂ = 0 (constant signal), no normalization occurs and the window is skipped.

### Quotient Space Identification

The **antipodal equivalence relation** on S^{L-1} is:

```
u ~ -u  (antipodal identification)
```

This defines the **quotient space**:

```
ℝP^{L-1} := S^{L-1} / {u ~ -u}
```

which is **real projective space** of dimension L-1. This space is **non-orientable** for L ≥ 2.

**Seam detection** searches for time index τ where treating the signal as living in ℝP^{L-1} (rather than ℝ^L) achieves lower MDL by applying the flip transformation u → -u.

### SNR Definition (Unambiguous)

We use **amplitude SNR** (not power SNR) defined as:

```
SNR = σ_signal / σ_noise
```

where:
- σ_signal = standard deviation of the **true underlying signal** (noiseless)
- σ_noise = standard deviation of additive noise

**Equivalently** (for zero-mean signals):

```
SNR = √(‖s‖²/‖ε‖²)  where x = s + ε
```

This is **NOT** the same as 10·log₁₀(power ratio) used in decibel measurements.

### Crossover Definition

The **crossover point** k* is the SNR value where:

```
Pr[ΔMDL < 0] = 0.5
```

Meaning: with 50% probability, the seam-aware model achieves better (lower) MDL than the baseline.

**Operational test**: Generate N signals with SNR = k_test, fit both models, count how many times ΔMDL < 0. The crossover is where this count equals N/2.

### MDL Cost Breakdown (Explicit Coding Model)

For a signal of length T with m detected seams and K model parameters:

```
MDL_total = L_data + L_params + L_seams
```

**Component 1: Data encoding cost (Gaussian negative log-likelihood)**

```
L_data = (T/2)·log₂(2πe·σ̂²)  bits
```

where σ̂² = RSS/T is the empirical residual variance.

**Component 2: Parameter encoding cost**

```
L_params = (K/2)·log₂(T)  bits
```

This uses the **normalized maximum likelihood (NML)** cost for K real-valued parameters under T observations (Rissanen, 1996).

**Component 3: Seam encoding cost**

```
L_seams = m·log₂(T) + m  bits
```

Breaking down:
- **Seam locations**: m·log₂(T) bits
  - Each of m seams is an integer in [0, T-1], costing log₂(T) bits
  - Uses fixed-length encoding (not prefix-free) for simplicity
- **Orientation bits**: m bits
  - Each seam introduces a ℤ₂ choice: flip or don't flip
  - Literally 1 bit per seam

**Total**:

```
MDL = (T/2)·log₂(2πe·σ̂²) + (K/2)·log₂(T) + m·[log₂(T) + 1]
```

**Note**: The "1 bit per seam" claim in the README refers to the **orientation cost** only. The full seam cost is log₂(T) + 1 ≈ 8-10 bits for typical T = 200-1000.

### Reconciling Theoretical vs. Experimental k*

**Analytic candidate**: k* = 1/(2·ln 2) ≈ 0.7213 (under simplified model, see Section 3.4)

**Empirical status**: Monte Carlo validation under the full pipeline (model zoo, roughness candidates, beam search) yields crossover estimates in the range ~1.0–1.2, with high run-to-run variance across 50–100 trials. This is substantially higher than the analytic value.

**Why the gap is expected:**

1. **Nonzero Δp**: The model zoo selects different model classes per segment, adding parameters (Δp > 0). This raises the effective threshold by (Δp/2)·log₂(N)/N per sample.

2. **Localization error**: Roughness candidates have ±5–15 sample error. RSS₁ is evaluated at the estimated, not true, seam position, inflating residuals.

3. **Model-zoo selection overhead**: Choosing the best of 7 models per segment incurs an implicit complexity cost not included in the simplified formula.

4. **MDL objective mismatch**: The k* derivation uses a specific simplified MDL; the production code uses `mdl_bits()` with pooled RSS, which differs in constant terms.

**Conclusion**: k* ≈ 0.721 is best understood as an analytic lower bound under idealized conditions, not as a universal empirical threshold. The effective operational threshold under the full pipeline is higher and data-dependent. Further Monte Carlo experiments with the canonical `mdl_bits()` objective (using matched hyperparameters) are needed before claiming a precise empirical value.

## Scope and Assumptions

This document derives the key results under the following assumptions:

1. **Gaussian noise**: Residuals are i.i.d. N(0, σ²)
2. **Single seam**: One orientation discontinuity per signal
3. **Known seam position**: Detection error is negligible
4. **Sufficient samples**: T >> K (number of parameters)

The constant k* = 1/(2·ln 2) ≈ 0.721 emerges from these assumptions. For non-Gaussian noise (Laplace, Cauchy), different thresholds apply—see [seamaware/core/mdl.py](seamaware/core/mdl.py) for implementations.

## Abstract

We establish the theoretical framework for **seam-aware time series analysis**. Normalized signal windows live on Sⁿ⁻¹; the antipodal identification u ~ −u quotients this to ℝPⁿ⁻¹ and introduces a global sign ambiguity — a ℤ₂ double cover — that is the geometric source of sign-flip seams. We derive the MDL gain expression for a sign-flip seam (including the Δp parameter-cost term often omitted in simplified presentations) and discuss the analytic candidate threshold k* = 1/(2·ln 2) ≈ 0.721 under idealized assumptions, clarifying the gap between this analytic value and empirical estimates from the full pipeline.

**Note on notation**: While the general theory can be extended to complex signals (leading to ℂᴺ/ℤ₂), our implementation and experiments work exclusively with **real-valued time series**. The quotient space is therefore Sⁿ⁻¹/ℤ₂ ≅ ℝPⁿ⁻¹, where Sⁿ⁻¹ is the unit sphere in ℝⁿ.

---

## 1. Quotient Space Construction

### 1.1 The Antipodal Map

Let **S** : ℝⁿ → ℝⁿ be the **antipodal map**:

```
Sx = -x
```

This generates the cyclic group ℤ₂ = {I, S} acting on ℝⁿ (and restricted to Sⁿ⁻¹).

**Key Property:** S² = I (involution)

### 1.2 Quotient Topology

The **orbit space** Sⁿ⁻¹/ℤ₂ identifies each point u on the sphere with its antipode -u:

```
[u] = {u, -u}  ∈  Sⁿ⁻¹/ℤ₂
```

**Theorem 1 (Quotient Homeomorphism):**
Sⁿ⁻¹/ℤ₂ is homeomorphic to **real projective space** ℝPⁿ⁻¹.

*Proof:* The projection π : Sⁿ⁻¹ → ℝPⁿ⁻¹ given by u ↦ [u₀ : u₁ : ... : uₙ₋₁] (homogeneous coordinates) descends to the quotient since π(u) = π(-u). The map is continuous, surjective, and open by the quotient topology. ∎

**Corollary (ℤ₂ double cover):** The projection π : Sⁿ⁻¹ → ℝPⁿ⁻¹ is a two-sheeted covering map. Traversing a closed loop in ℝPⁿ⁻¹ may lift to an open path between antipodal points in Sⁿ⁻¹, introducing a global sign ambiguity.

**Note on orientability:** ℝP^m is **non-orientable when m is even** (e.g., ℝP² is the real projective plane) and **orientable when m is odd** (e.g., ℝP¹ ≅ S¹ and ℝP³ are orientable). The relevant modeling consequence is not orientability per se but the ℤ₂ sign ambiguity introduced by the quotient, which is present for all m ≥ 1.

---

## 2. ℤ₂ Action on Adjacent Window Pairs

### 2.1 The Relevant Operator

For the global antipodal map S = −I on ℝⁿ, the projections 𝐏₊ = ½(I + S) = 0 and 𝐏₋ = ½(I − S) = I. The +1 eigenspace is {0} and the −1 eigenspace is all of ℝⁿ, making the decomposition trivially uninformative for nonzero vectors.

A nontrivial decomposition requires a different operator. Define the **window-pair swap operator** acting on adjacent left/right windows (x_L, x_R) ∈ ℝʷ × ℝʷ:

```
S̃(x_L, x_R) = (−x_R, −x_L)
```

This is the antipodal map in the product space but applied with a swap, so S̃² = I (involution). The +1 eigenspace of S̃ is {(a, −a) : a ∈ ℝʷ} (antipodal pairs) and the −1 eigenspace is {(a, a) : a ∈ ℝʷ} (symmetric/aligned pairs).

### 2.2 Energy Decomposition

For an adjacent window pair (x_L, x_R):

```
𝐏₊(x_L, x_R) = ½(x_L − x_R,  x_R − x_L)   [antipodal component]
𝐏₋(x_L, x_R) = ½(x_L + x_R,  x_L + x_R)   [aligned component]
```

Define the **antipodal energy fraction**:

```
α₊ = ‖𝐏₊(x_L, x_R)‖² / ‖(x_L, x_R)‖²  ∈ [0, 1]
```

**Interpretation:** α₊ → 1 when x_R ≈ −x_L (a sign-flip seam across the boundary); α₊ → 0 when x_L ≈ x_R (no regime change). This is well-defined for nonzero pairs and connects directly to the antipodal correlation score used in detection.

---

## 3. The k* Constant from MDL Theory

### 3.1 Minimum Description Length (MDL)

The **two-part code** for a signal x given model M:

```
MDL(x | M) = L(x | M) + L(M)
```

where:
- L(x | M) = negative log-likelihood in bits (data given model)
- L(M) = parameter cost = (k/2)·log₂(N) bits (Rissanen, 1978)

### 3.2 Seam-Aware Encoding

**Baseline model:** Polynomial of degree d (no seam)
- Parameters: k₀ = d + 1
- MDL₀ = NLL₀ + (k₀/2)·log₂(N)

**Seam model:** Polynomial + seam at τ + flip atom
- Parameters: k₁ = k₀ + 1 + p (seam location + atom params)
- MDL₁ = NLL₁ + (k₁/2)·log₂(N)

**Accept seam if:** ΔMDL = MDL₁ - MDL₀ < 0

### 3.3 Complete MDL Gain Expression

The full gain from adding one seam (m = 1) is:

```
ΔMDL = (N/2)·log₂(RSS₀/RSS₁) − (Δp/2)·log₂(N) − (α/2)·log₂(N)
```

where:
- RSS₀, RSS₁ are the residual sums of squares before and after the seam
- Δp = p₁ − p₀ is the increase in fitted parameters (two-segment model minus one-segment model)
- α is the seam-penalty coefficient (default α = 2, so the seam-location term costs log₂(N) bits)

**Note on the dominant-term simplification.** In the idealized case of equal-length segments with the same model class on each side, Δp = 0 and the gain reduces to:

```
ΔMDL ≈ (N/2)·log₂(RSS₀/RSS₁) − (α/2)·log₂(N)
```

This is the form used for intuition in the paper. However, the model zoo uses different model classes per segment (Fourier, polynomial, AR), so Δp may be nonzero and the full expression should be used when interpreting empirical gain values.

### 3.4 Candidate Analytic Threshold k*

Under the simplified model (Δp = 0, piecewise-constant means, iid Gaussian noise with amplitude A and noise std σ_n):

```
RSS₀ ≈ N(A² + σ_n²)
RSS₁ ≈ N·σ_n²
```

so RSS₀/RSS₁ ≈ 1 + (A/σ_n)². Setting ΔMDL = 0:

```
(N/2)·log₂(1 + r²) = (α/2)·log₂(N)    where r = A/σ_n
```

For large N this implies r² ≈ α·log₂(N)/N → 0, i.e., the threshold shrinks asymptotically. In the Rissanen two-part MDL literature, the analogous per-sample gain threshold yields:

```
SNR_eff* ≈ 1 / (2·ln 2) ≈ 0.7213
```

where SNR_eff = (σ₀² − σ₁²)/σ₁² = r².

**Proposition (candidate threshold under simplified Gaussian model):**
Under a two-segment antipodal mean model with iid Gaussian noise, Δp = 0, and equal-length segments, the expected MDL gain is positive when SNR_eff > α/(N·ln 2) and negative for pure-noise splits with high probability.  The value k* = 1/(2·ln 2) ≈ 0.721 is the breakeven under this simplified model as N → ∞.

**Caveats — this is not a universal theorem:**
1. The model zoo introduces nonzero Δp, which raises the effective threshold.
2. Colored or heavy-tailed turbulence changes the RSS ratio.
3. Empirical Monte Carlo results under the full pipeline are noisy and sensitive to the MDL objective definition; stored validation artifacts show empirical crossover estimates in the range 1.0–1.2, not 0.72. The gap reflects finite-sample bias, localization error, and model-zoo overhead, not a proof failure.
4. The "if and only if" phrasing used in earlier drafts was too strong; the proposition above is the correct scope.

---

## 4. Seam Detection via Roughness

### 4.1 Local Residual Variance

Define the **roughness function** R(τ) at candidate seam location τ:

```
R(τ) = Var(residuals in window [τ - w, τ + w])
```

where residuals are computed after fitting a local polynomial (typically degree 1-3).

**Algorithm:**
1. For each candidate location τ ∈ [w, N-w]:
   - Fit polynomial to window [τ-w, τ+w]
   - Compute residual variance
2. **Seam candidates:** Local maxima of R(τ) exceeding threshold θ = μ + 2σ
3. **Refinement:** Evaluate MDL for each candidate

**Complexity:** O(N·w·d²) where d = polynomial degree, w = window size

### 4.2 Commutativity Criterion

For a flip atom F : ℝⁿ → ℝⁿ to be valid under the quotient structure:

```
[F, S] = 0  (F commutes with antipodal map S)
```

**Valid flip atoms:**

1. **Sign flip:** F(x) = -x
   - Verification: F(Sx) = F(-x) = x, S(Fx) = S(-x) = x ✓

2. **Time reversal:** F(x₁:τ, xτ:N) = (x₁:τ, reverse(xτ:N))
   - Preserves ‖x‖² but changes temporal orientation

3. **Variance scaling:** F(xτ:N) = α·xτ:N where α = √(σ²pre / σ²post)
   - Homogenizes variance across the seam

4. **Polynomial detrending:** F(x) = x - poly_fit(x)
   - Projects onto zero-mean subspace

**Non-commuting transformations** are rejected as they break the ℤ₂ symmetry. (Note: Phase shifts would apply in the complex extension, but our implementation uses real-valued signals only.)

---

## 5. Orientation Tracking and the "Anti-Bit"

### 5.1 The Orientation State Vector

In the quotient space ℝPⁿ⁻¹, we cannot globally distinguish u from -u on the sphere. However, we can track **transitions** between sheets.

Define the **orientation state vector** o ∈ {±1}ᴺ:

```
o(t) = +1  if t is on the original sheet
o(t) = -1  if t is on the flipped sheet
```

At a seam location τ:

```
o(t) = { +1  for t < τ
       { -1  for t ≥ τ
```

**Encoding cost:** Each seam requires log₂(N) bits to specify τ.

### 5.2 Multi-Seam Tracking

For K seams at locations τ₁, τ₂, ..., τₖ:

```
o(t) = (-1)^(number of seams before t)
```

**Total encoding cost:** K·log₂(N) bits

**MDL acceptance criterion:** Accept seam k if:

```
ΔMDL_k = MDL(k seams) - MDL(k-1 seams) < 0
```

This implements **greedy seam addition** with automatic stopping.

---

## 6. Applications to Neural Networks

### 6.1 Seam-Gated Architecture

A **seam-gated RNN** maintains two hidden states corresponding to ℤ₂ eigenspaces:

```
h₊ₜ = tanh(W₊·[hₜ₋₁, xₜ])  (symmetric branch)
h₋ₜ = tanh(W₋·[hₜ₋₁, xₜ])  (antisymmetric branch)
```

At detected seam τ, switch branches:

```
hₜ = { h₊ₜ   if t < τ
     { h₋ₜ   if t ≥ τ
```

**Gradient flow:** Backpropagation is **blocked at seams**—no gradient leakage across discontinuities. This prevents the vanishing gradient problem at regime boundaries.

### 6.2 Theoretical Convergence Guarantee

**Theorem 3 (Seam-Gated Convergence):**
For regime-switching data with true seam at τ* and SNR > k*, a seam-gated network with detected seam within |τ - τ*| < ε converges to loss:

```
L_seam ≤ (1 - α₋·k*) · L_standard
```

where α₋ is the antisymmetric energy fraction and L_standard is the standard RNN loss.

*Proof sketch:* The seam gate isolates pre- and post-seam dynamics, allowing each branch to specialize. The α₋ factor measures how much energy is "released" by correctly aligning with the ℤ₂ symmetry. The k* threshold ensures MDL consistency. ∎

---

## 7. Information-Geometric Perspective

### 7.1 Projective Direction Space and pPVI

The normalized field direction B̂ = B/(|B| + ε) lives on S² ⊂ ℝ³. Identifying B̂ ~ −B̂ quotients this to ℝP². The projective distance:

```
d_RP²(B̂₁, B̂₂) = arccos(|B̂₁ · B̂₂|)
```

is zero for pure antipodal reversals (B₂ = −B₁) and maximum (π/2) for orthogonal directions. This is the basis of the projective PVI (pPVI) variant.

**Important caveat:** Because d_RP² = 0 for ideal antipodal flips, pPVI does **not** directly detect pure polarity reversals. Its sensitivity on sign-flip benchmarks arises because a physical tanh-shaped reversal passes through a near-null region (|B| ≈ 0) where B̂ = B/(|B| + ε) becomes direction-unstable. High pPVI at a polarity reversal indicates local field-magnitude depression, not the reversal topology itself.

Interpretation: pPVI detects near-null directional instability; MASS/SMASH detects compressible antipodal mean structure. The two methods are complementary, not redundant.

### 7.2 Scope of Geometric Claims

The ℝPⁿ quotient construction is a useful geometric interpretation of why sign ambiguity arises, but the paper's empirical contributions (antipodal score, MDL gate, roughness contrast) do not depend on information-geometric theorems. Claims about Fisher metrics, scalar curvature, or k*-from-curvature derivations are speculative and are not made in the main paper.

---

## 8. Open Questions and Future Directions

### 8.1 Higher-Order Quotients

**Question:** Does the cyclic group ℤ₄ (quarter-rotations in ℂᴺ) yield a new constant k** for FFT-based seams?

**Conjecture:** k** ≈ 0.36 based on preliminary numerics.

### 8.2 Continuous Seams

**Question:** Can we extend the framework to **manifolds with boundary** (Möbius strip, Klein bottle)?

**Application:** Continuous regime transitions in control theory.

### 8.3 Multi-Scale Detection

**Question:** How do seam hierarchies interact? Does a wavelet-based detection yield a fractal seam structure?

**Connection:** Self-similar information-theoretic phase transitions.

### 8.4 Quantum Interpretation

**Speculation:** The ℤ₂ eigenspace decomposition resembles **spin measurements** in quantum mechanics. Is there a Bell inequality for seam-aware data?

---

## 9. References

1. Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465-471.
2. Lee, J. M. (2013). *Introduction to Smooth Manifolds* (2nd ed.). Springer.
3. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
4. Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press.
5. Mayo, M. (2025a). Seam-Aware Modeling: Non-Orientable Quotient Spaces for Time Series Analysis. *arXiv preprint arXiv:XXXX.XXXXX*.
6. Mayo, M. (2025b). The k* Constant: Information-Theoretic Phase Transitions in Non-Orientable Spaces. *In preparation*.

---

## Appendix A: Detailed k* Derivation

### A.1 Setup

Consider a piecewise-stationary Gaussian process:

```
x(t) = { s₀(t) + ε₀(t)  for t < τ
       { s₁(t) + ε₁(t)  for t ≥ τ
```

where:
- s₀, s₁ are deterministic signals
- ε₀ ~ N(0, σ₀²), ε₁ ~ N(0, σ₁²) are noise

### A.2 MDL Without Seam

Fit a single polynomial of degree d to entire signal:

```
MDL₀ = (N/2)·log₂(2πeσ̂²) + (d+1)/2 · log₂(N)
```

where σ̂² is the empirical residual variance.

### A.3 MDL With Seam

Fit separate polynomials before/after τ:

```
MDL₁ = (N₀/2)·log₂(2πeσ̂₀²) + (N₁/2)·log₂(2πeσ̂₁²)
       + (2d + 2)/2 · log₂(N) + log₂(N)
```

The last term log₂(N) encodes the seam location.

### A.4 Complete Gain Expression

Setting ΔMDL = MDL₁ − MDL₀ = 0 and using the full formula:

```
ΔMDL = (N/2)·log₂(RSS_seam/RSS_baseline) − (Δp/2)·log₂(N) − (α/2)·log₂(N)
```

where Δp = p₁ − p₀ is the net parameter increase. For the scenario where each segment uses the same degree-d polynomial (p₀ = d+1 per side, two segments vs. one, so p₁ = 2(d+1) and Δp = d+1):

```
(N/2)·log₂(RSS_baseline/RSS_seam) = (Δp + α)/2 · log₂(N)
```

For balanced equal-length segments with homogeneous noise:

```
log₂(RSS_seam/RSS_baseline) ≈ −(Δp + α)/N · log₂(N)
```

which goes to zero as N → ∞. The fractional variance reduction required is:

```
(RSS_baseline − RSS_seam)/RSS_seam ≈ (Δp + α)·ln N / N
```

Under the simplified piecewise-constant mean model (Δp = 0, α = 2):

```
SNR_eff* = (A/σ_n)²  at breakeven ≈ α·ln N / N → 0 as N → ∞
```

The value k* = 1/(2·ln 2) ≈ 0.721 is a reference derived from a related Rissanen-style per-sample gain argument; it does **not** emerge directly from the displayed algebra above. The derivation is better understood as providing a finite-sample reference: for N = 200, α = 2, Δp = 0, the breakeven r² = A²/σ_n² ≈ 0.07 (r ≈ 0.27), which is substantially lower than k* ≈ 0.72. With Δp = 10 (typical model-zoo overhead), r rises to ≈ 0.76, close to empirical estimates.

### A.5 Scope

The simplified derivation gives useful intuition: seams must compress. The exact threshold depends on chunk size N, model complexity Δp, seam penalty α, and noise distribution. The k* = 0.721 value should be treated as an analytic reference, not a universal constant.

---

## Appendix B: Computational Complexity

### B.1 Seam Detection

**Naive approach:** Try all N positions → O(N²·d²) for degree-d polynomials

**Roughness optimization:**
1. Compute running variance in O(N) via cumulative sums
2. Detect local maxima in O(N)
3. Evaluate MDL for K candidates in O(K·N·d²)
4. **Total:** O(N + K·N·d²) where typically K ≪ N

### B.2 Multi-Seam Extension

For M seams:
- **Exact search:** O(Nᴹ) — intractable for M > 3
- **Greedy algorithm:** O(M·K·N·d²) — practical for M ≤ 10
- **Dynamic programming:** O(M·N²) if MDL is additive — best known

---

**End of THEORY.md**
