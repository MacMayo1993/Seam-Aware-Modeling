# SeamAware: Non-Orientable Modeling for Time Series Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What is SeamAware?

**SeamAware** detects and exploits **orientation discontinuities** (seams) in time series data—structural features that standard methods treat as noise. By recognizing that certain signals naturally inhabit **non-orientable quotient spaces** (ℂᴺ/ℤ₂ ≅ ℝℙᴺ⁻¹), we achieve:

- **10-170% compression improvement** over standard methods
- **Provable MDL reduction** via seam-gated transformations
- **Robust regime-switching detection** without hidden states
- **Emergence of k* ≈ 0.721** as universal information-theoretic threshold

### The Core Insight

Standard methods assume data lives in **orientable spaces** (ℝⁿ or ℂⁿ). SeamAware recognizes that:

```
Signal + Noise → Detect seam → Apply ℤ₂ quotient → Lower MDL
```

At the seam location τ, we apply a **flip atom** (sign inversion, time reversal, variance scaling, or polynomial detrending) that exploits latent antipodal symmetry. The cost of tracking orientation (1 bit per seam) is offset by improved model fit **if and only if** the signal-to-noise ratio exceeds k* ≈ 0.721.

**This constant is not empirical—it emerges from MDL theory.**

### Quick Example

```python
from seamaware import MASSFramework
from seamaware.models import FourierBaseline
from seamaware.core.mdl import compute_mdl
import numpy as np

# Generate regime-switching data
t = np.linspace(0, 4*np.pi, 200)
signal = np.sin(t)
signal[100:] *= -1  # Hidden sign flip at t=100

# Standard Fourier baseline
fourier = FourierBaseline(K=12)
fourier_pred = fourier.fit_predict(signal)
fourier_mdl = compute_mdl(signal, fourier_pred, fourier.num_params())
# → 850.3 bits

# SeamAware MASS framework
mass = MASSFramework()
result = mass.fit_predict(signal)
# → Detects seam at t=102
# → Applies SignFlip atom
# → 712.1 bits (16% reduction)

print(f"Seam detected: {result.seam_used}")  # 102 (within 2% of truth)
print(f"MDL improvement: {(fourier_mdl - result.mdl_score) / fourier_mdl:.1%}")
```

### Installation

```bash
pip install seamaware
```

Or from source:
```bash
git clone https://github.com/MacMayo1993/Seam-Aware-Modeling.git
cd Seam-Aware-Modeling
pip install -e .
```

### Mathematical Foundations

The theory behind SeamAware connects:

1. **Non-orientable manifolds**: ℝℙⁿ as the quotient Sⁿ/ℤ₂ via antipodal map x → -x
2. **Information geometry**: k* = 1/(2·ln 2) emerges from minimum description length
3. **Group representation theory**: ℤ₂ eigenspace decomposition via projection operators 𝐏₊/𝐏₋
4. **Seam-gated neural networks**: Architectures that switch basis at detected seams

See [THEORY.md](THEORY.md) for rigorous derivations and [docs/mathematical_foundations.pdf](docs/mathematical_foundations.pdf) for full proofs.

### Who Should Use This?

- **Signal processing researchers** working with regime-switching data
- **Compression engineers** seeking provable gains beyond Huffman/arithmetic coding
- **Time series analysts** in finance, energy, biomedical applications
- **Topologists interested in applied non-orientable geometry**

### Features

- **Core Detection**: Roughness-based seam detection with statistical thresholding
- **Flip Atoms**: Sign flip, time reversal, variance scaling, polynomial detrending
- **Orientation Tracking**: Novel "anti-bit" framework for tracking position in quotient space
- **MDL Framework**: Rigorous information-theoretic model selection
- **k* Validation**: Monte Carlo tests confirming theoretical phase boundary
- **Baseline Comparisons**: Fourier, AR, polynomial models

### Project Status

This is research software under active development. APIs may change as we refine the mathematical framework.

### Citation

If you use SeamAware in your research, please cite:

```bibtex
@article{mayo2025seamaware,
  title={Seam-Aware Modeling: Non-Orientable Quotient Spaces for Time Series Analysis},
  author={Mayo, Mac},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

### License

Apache 2.0 — see [LICENSE](LICENSE)

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Contact

- **Author**: Mac Mayo
- **Issues**: [GitHub Issues](https://github.com/MacMayo1993/Seam-Aware-Modeling/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MacMayo1993/Seam-Aware-Modeling/discussions)

### Acknowledgments

This work builds on foundational research in:
- Minimum Description Length (Rissanen, 1978)
- Differential geometry and topology (Lee, 2013)
- Information geometry (Amari, 2016)
