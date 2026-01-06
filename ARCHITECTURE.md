# SeamAware Architecture

## Design Principles

1. **Separation of concerns:**
   - `core/` = mathematical operations (seam detection, flip atoms, MDL)
   - `models/` = complete modeling frameworks (MASS, baselines)
   - `theory/` = k* calculations, quotient space math
   - `compression/` = compression applications (FlipZip, multi-scale detection)
   - `utils/` = data generation, visualization, helper functions

2. **Type safety:** Use Python 3.9+ type hints throughout
3. **Immutability:** Flip atoms don't modify input arrays (return copies)
4. **Testability:** Every module has corresponding `tests/test_*.py`
5. **Reproducibility:** All random operations use seeded generators

## Core Abstractions

### FlipAtom (Abstract Base Class)

```python
from abc import ABC, abstractmethod
import numpy as np

class FlipAtom(ABC):
    """
    Base class for seam-aware transformations.

    A valid FlipAtom must:
    1. Commute with the antipodal map S: x → -x
    2. Have an inverse (or be an involution)
    3. Report parameter count for MDL
    """

    @abstractmethod
    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        """
        Apply transformation at seam location.

        Args:
            data: Input signal (length N)
            seam: Seam location τ ∈ [0, N)

        Returns:
            Transformed signal (same shape as input)
        """
        pass

    @abstractmethod
    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        """
        Inverse transformation (must satisfy F⁻¹(F(x)) = x).

        Args:
            data: Transformed signal
            seam: Original seam location

        Returns:
            Original signal
        """
        pass

    @abstractmethod
    def num_params(self) -> int:
        """
        Parameter count for MDL calculation.

        Returns:
            Number of parameters (seam location excluded)
        """
        pass

    def fit_params(self, data: np.ndarray, seam: int) -> None:
        """
        Optional: Learn transformation parameters from data.

        Args:
            data: Signal to analyze
            seam: Seam location
        """
        pass
```

### Orientation Tracker (NEW - Not in October code)

```python
from typing import List
import numpy as np

class OrientationTracker:
    """
    Maintains "anti-bit" state across seams in quotient space ℂᴺ/ℤ₂.

    Tracks whether current position is on:
    - Original sheet (orientation = +1)
    - Flipped sheet (orientation = -1)

    This enables:
    1. Proper signal reconstruction after multiple flips
    2. MDL cost accounting for seam locations
    3. Visualization of quotient space trajectory
    """

    def __init__(self, length: int):
        """
        Initialize tracker for signal of given length.

        Args:
            length: Signal length N
        """
        self.length = length
        self.orientations = np.ones(length, dtype=np.int8)
        self.seams: List[int] = []

    def flip_at(self, seam: int) -> None:
        """
        Toggle orientation starting at seam.

        Args:
            seam: Location τ where flip occurs
        """
        if seam < 0 or seam >= self.length:
            raise ValueError(f"Seam {seam} out of bounds [0, {self.length})")

        self.orientations[seam:] *= -1
        self.seams.append(seam)

    def get_orientation(self, t: int) -> int:
        """
        Return ±1 orientation at time t.

        Args:
            t: Time index

        Returns:
            +1 (original sheet) or -1 (flipped sheet)
        """
        return int(self.orientations[t])

    def encoding_cost_bits(self) -> float:
        """
        Cost in bits to encode seam locations.

        Returns:
            k·log₂(N) bits for k seams in signal of length N
        """
        k = len(self.seams)
        return k * np.log2(self.length) if k > 0 else 0.0

    def reset(self) -> None:
        """Reset to all +1 orientations."""
        self.orientations.fill(1)
        self.seams.clear()
```

---

## Key Modules

### `core/mdl.py`

Implements Rissanen's MDL with proper bit counting:

```python
import numpy as np

def compute_mdl(data: np.ndarray,
                prediction: np.ndarray,
                num_params: int) -> float:
    """
    Compute Minimum Description Length in bits.

    MDL = NLL(data | model) + (k/2)·log₂(N)

    Args:
        data: Observed signal (length N)
        prediction: Model prediction
        num_params: Number of model parameters k

    Returns:
        Total description length in bits

    Reference:
        Rissanen, J. (1978). Modeling by shortest data description.
        Automatica, 14(5), 465-471.
    """
    n = len(data)
    if n != len(prediction):
        raise ValueError("Data and prediction must have same length")

    residuals = data - prediction
    sigma2 = np.var(residuals) + 1e-10  # Regularization

    # Negative log-likelihood (Gaussian assumption)
    # -log₂ p(x | μ, σ²) = (1/2)·log₂(2πeσ²) + (x-μ)²/(2σ²·ln 2)
    nll = (n / 2) * np.log2(2 * np.pi * np.e * sigma2)

    # Parameter cost (Rissanen's normalized maximum likelihood)
    param_cost = (num_params / 2) * np.log2(n)

    return nll + param_cost


def delta_mdl(mdl_baseline: float, mdl_seam: float) -> float:
    """
    Compute MDL improvement (negative = better).

    Args:
        mdl_baseline: MDL without seam
        mdl_seam: MDL with seam

    Returns:
        ΔMDL = mdl_seam - mdl_baseline (accept if < 0)
    """
    return mdl_seam - mdl_baseline
```

### `theory/k_star.py`

Computes the k* constant and validates convergence:

```python
import numpy as np
from typing import Dict, Tuple
from ..core.mdl import compute_mdl

def compute_k_star() -> float:
    """
    The universal seam-aware modeling constant.

    k* = 1 / (2·ln 2) ≈ 0.7213

    This is the SNR threshold where 1-bit seam encoding
    cost equals per-sample MDL reduction.

    Returns:
        k* constant
    """
    return 1.0 / (2.0 * np.log(2))


def validate_k_star_convergence(
    signal_length: int = 200,
    seam_location: int = 100,
    snr_range: Tuple[float, float] = (0.1, 2.0),
    num_snr_points: int = 20,
    num_trials: int = 50,
    seed: int = 42
) -> Dict:
    """
    Monte Carlo validation of k* phase boundary.

    Generates synthetic signals with controlled SNR and verifies
    that ΔMDL < 0 occurs at SNR ≈ k* ≈ 0.721.

    Args:
        signal_length: Length of synthetic signals
        seam_location: Where to introduce flip
        snr_range: (min_snr, max_snr) to test
        num_snr_points: Number of SNR values to sample
        num_trials: Monte Carlo repetitions per SNR
        seed: Random seed for reproducibility

    Returns:
        {
            'snr_values': array of SNR values tested,
            'delta_mdl_mean': mean ΔMDL at each SNR,
            'delta_mdl_std': standard deviation of ΔMDL,
            'crossover_snr': estimated SNR where ΔMDL = 0,
            'theoretical_k_star': 0.7213...,
            'relative_error': |crossover - k*| / k*
        }
    """
    from ..core.seam_detection import detect_seams_roughness
    from ..core.flip_atoms import SignFlipAtom
    from ..models.baselines import PolynomialBaseline

    rng = np.random.default_rng(seed)
    snr_values = np.linspace(snr_range[0], snr_range[1], num_snr_points)

    delta_mdl_results = []

    for snr in snr_values:
        trial_deltas = []

        for _ in range(num_trials):
            # Generate signal with known seam
            t = np.linspace(0, 4*np.pi, signal_length)
            signal = np.sin(t)
            signal[seam_location:] *= -1  # True seam

            # Add noise to achieve target SNR
            noise_power = np.var(signal) / snr
            noise = rng.normal(0, np.sqrt(noise_power), signal_length)
            noisy_signal = signal + noise

            # Baseline: polynomial fit (no seam)
            baseline = PolynomialBaseline(degree=3)
            pred_baseline = baseline.fit_predict(noisy_signal)
            mdl_baseline = compute_mdl(noisy_signal, pred_baseline,
                                       baseline.num_params())

            # Seam-aware: detect + flip
            flip_atom = SignFlipAtom()
            detected_seams = detect_seams_roughness(noisy_signal,
                                                     window=20,
                                                     threshold_sigma=2.0)

            if len(detected_seams) > 0:
                # Use closest seam to true location
                best_seam = min(detected_seams,
                               key=lambda s: abs(s - seam_location))
                flipped = flip_atom.apply(noisy_signal, best_seam)
                pred_seam = baseline.fit_predict(flipped)
                mdl_seam = compute_mdl(flipped, pred_seam,
                                       baseline.num_params() +
                                       flip_atom.num_params() + 1)  # +1 for seam loc
                delta = mdl_seam - mdl_baseline
            else:
                delta = np.inf  # No seam detected → no improvement

            trial_deltas.append(delta)

        delta_mdl_results.append(trial_deltas)

    # Compute statistics
    delta_mdl_mean = np.array([np.mean(d) for d in delta_mdl_results])
    delta_mdl_std = np.array([np.std(d) for d in delta_mdl_results])

    # Find crossover point (where ΔMDL ≈ 0)
    # Use linear interpolation
    sign_changes = np.where(np.diff(np.sign(delta_mdl_mean)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation between idx and idx+1
        x0, x1 = snr_values[idx], snr_values[idx+1]
        y0, y1 = delta_mdl_mean[idx], delta_mdl_mean[idx+1]
        crossover_snr = x0 - y0 * (x1 - x0) / (y1 - y0)
    else:
        crossover_snr = np.nan

    k_star_theoretical = compute_k_star()
    relative_error = abs(crossover_snr - k_star_theoretical) / k_star_theoretical

    return {
        'snr_values': snr_values,
        'delta_mdl_mean': delta_mdl_mean,
        'delta_mdl_std': delta_mdl_std,
        'crossover_snr': crossover_snr,
        'theoretical_k_star': k_star_theoretical,
        'relative_error': relative_error
    }
```

---

## Testing Strategy

### Unit Tests

**tests/test_flip_atoms.py**: Verify F⁻¹(F(x)) = x for all atoms
```python
def test_sign_flip_involution():
    """SignFlip is its own inverse."""
    x = np.random.randn(100)
    atom = SignFlipAtom()
    seam = 50
    y = atom.apply(x, seam)
    x_recovered = atom.inverse(y, seam)
    np.testing.assert_array_almost_equal(x, x_recovered)
```

**tests/test_mdl.py**: Validate bit counting against known examples
```python
def test_mdl_monotonicity():
    """Better fit → lower MDL."""
    x = np.sin(np.linspace(0, 2*np.pi, 100))
    pred_good = x + 0.1 * np.random.randn(100)
    pred_bad = x + 1.0 * np.random.randn(100)

    mdl_good = compute_mdl(x, pred_good, num_params=2)
    mdl_bad = compute_mdl(x, pred_bad, num_params=2)

    assert mdl_good < mdl_bad
```

**tests/test_seam_detection.py**: Check detection on synthetic signals
```python
def test_detect_known_seam():
    """Detect pre-planted seam within tolerance."""
    t = np.linspace(0, 4*np.pi, 200)
    signal = np.sin(t)
    true_seam = 100
    signal[true_seam:] *= -1

    detected = detect_seams_roughness(signal, window=20)

    assert len(detected) > 0
    closest = min(detected, key=lambda s: abs(s - true_seam))
    assert abs(closest - true_seam) < 5  # Within 2.5%
```

### Integration Tests

**tests/test_mass_framework.py**: End-to-end on synthetic benchmarks
```python
def test_mass_on_sign_flip():
    """MASS detects sign flip and achieves lower MDL."""
    signal = generate_sign_flip_signal(length=200, seam=100, noise=0.1)

    mass = MASSFramework()
    result = mass.fit_predict(signal)

    baseline_mdl = compute_baseline_mdl(signal)

    assert result.mdl_score < baseline_mdl
    assert abs(result.seam_location - 100) < 10
```

**tests/test_k_star_convergence.py**: Monte Carlo validation of k* ≈ 0.721
```python
def test_k_star_within_tolerance():
    """Crossover SNR matches theoretical k* within 10%."""
    results = validate_k_star_convergence(
        signal_length=200,
        num_trials=100
    )

    assert results['relative_error'] < 0.10
```

### Benchmarks

**tests/benchmarks/synthetic_suite.py**: Sign flip, variance shift, polynomial kink
**tests/benchmarks/real_data_tests.py**: HVAC, EEG, wind turbine data

---

## Performance Considerations

### Time Complexity

1. **Seam detection:** O(N·W·d²)
   - N = signal length
   - W = window size (typically 10-50)
   - d = polynomial degree (typically 1-3)
   - **Optimization:** Use running statistics for O(N·W)

2. **MDL computation:** O(K·N)
   - K = number of candidate seams
   - **Optimization:** Cache baseline predictions

3. **Multi-seam search:** O(M·N²) for M seams
   - **Optimization:** Greedy algorithm reduces to O(M·K·N)

### Space Complexity

- All operations create copies (functional style)
- **Memory usage:** O(N) for signal + O(K) for candidates
- **Large signals (N > 10⁶):** Consider streaming or chunking

### Parallelization Opportunities

- Seam detection windows are independent → parallelize across windows
- Monte Carlo trials are independent → parallelize validation
- Multi-scale detection uses wavelet tree → parallelize scales

---

## Extension Points

### Custom Flip Atoms

Inherit from `FlipAtom` and implement required methods:

```python
class PhaseShiftAtom(FlipAtom):
    """Rotate complex phase at seam (EXPERIMENTAL - may break ℤ₂ symmetry)."""

    def __init__(self, phase: float = np.pi):
        self.phase = phase

    def apply(self, data: np.ndarray, seam: int) -> np.ndarray:
        result = data.copy()
        if np.iscomplexobj(data):
            result[seam:] *= np.exp(1j * self.phase)
        return result

    def inverse(self, data: np.ndarray, seam: int) -> np.ndarray:
        result = data.copy()
        if np.iscomplexobj(data):
            result[seam:] *= np.exp(-1j * self.phase)
        return result

    def num_params(self) -> int:
        return 1  # phase parameter
```

### Custom Seam Detectors

Replace default roughness-based detector:

```python
def my_seam_detector(data: np.ndarray, **kwargs) -> List[int]:
    """
    Custom seam detection logic.

    Returns:
        List of candidate seam indices
    """
    # Your algorithm here
    candidates = []
    # ...
    return candidates

# Plug into MASS:
mass = MASSFramework(seam_detector=my_seam_detector)
```

### Custom Baselines

Implement `fit_predict` and `num_params`:

```python
class ARBaseline:
    """Autoregressive baseline (statsmodels wrapper)."""

    def __init__(self, order: int = 5):
        self.order = order
        self.model = None

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.ar_model import AutoReg
        self.model = AutoReg(data, lags=self.order).fit()
        return self.model.fittedvalues

    def num_params(self) -> int:
        return self.order + 1  # AR coefficients + intercept
```

---

## Code Style and Conventions

### Type Hints

Use Python 3.9+ type annotations everywhere:

```python
from typing import List, Tuple, Optional, Union
import numpy as np

def detect_seams(
    data: np.ndarray,
    window: int = 20,
    threshold: Optional[float] = None
) -> List[int]:
    ...
```

### Docstrings

Follow NumPy docstring convention:

```python
def function(param1: int, param2: str) -> bool:
    """
    One-line summary.

    Longer description with details about what the function does.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> function(5, "test")
    True
    """
```

### Error Handling

Be specific with exceptions:

```python
if seam < 0 or seam >= len(data):
    raise ValueError(f"Seam index {seam} out of bounds [0, {len(data)})")

if not np.isfinite(data).all():
    raise ValueError("Data contains NaN or inf values")
```

---

## Continuous Integration

**GitHub Actions** (`.github/workflows/tests.yml`):
- Run pytest on Python 3.9, 3.10, 3.11
- Check code coverage (target: >80%)
- Validate k* convergence test passes
- Build documentation
- Lint with black and mypy

**Pre-commit hooks**:
- Format with `black`
- Type check with `mypy`
- Sort imports with `isort`

---

## Future Architecture Plans

### Phase 2: Neural Networks

- **Seam-Gated RNN**: Implement dual-branch architecture
- **Attention mechanisms**: Learn seam locations end-to-end
- **PyTorch module**: `seamaware.nn.SeamGatedLSTM`

### Phase 3: Compression

- **FlipZip codec**: Arithmetic coding + seam-aware preprocessing
- **Streaming API**: Process signals larger than memory
- **Multi-scale wavelet seams**: Hierarchical detection

### Phase 4: Real-Time Applications

- **Online seam detection**: Sliding window approach
- **Embedded systems**: C++ implementation with Python bindings
- **GPU acceleration**: CUDA kernels for large-scale detection

---

**End of ARCHITECTURE.md**
