# SeamAware Interactive Applications

This directory contains interactive visualization tools for exploring SeamAware's non-orientable modeling approach.

## 📊 Streamlit Visualizer

**`streamlit_visualizer.py`** - Interactive web app for real-time comparison of standard vs. SeamAware methods.

### Launch

```bash
# From the repository root
streamlit run apps/streamlit_visualizer.py
```

The app will open in your browser at `http://localhost:8501`.

### Features

#### 🎯 Preloaded Examples

Six regime-switching scenarios with ground truth:

1. **Sine Wave (Single Seam)**: Classic orientation flip in clean periodic signal
2. **HVAC Regime Switching**: Real-world heating/cooling cycles with exponential dynamics
3. **ECG Polarity Inversion**: Medical device lead reversal scenario
4. **Audio Phase Flip**: Stereo channel mismatch (out-of-phase left/right)
5. **Multi-Seam (3 Flips)**: Multiple orientation discontinuities
6. **Variance Shift**: Volatility regime change (e.g., financial markets)

#### ⚙️ Interactive Controls

- **Baseline Model**: Choose Fourier or Polynomial, adjust parameters (K or degree)
- **Seam Detection**: Tune threshold σ, minimum distance between seams
- **Visualization**: Toggle roughness curve, MDL breakdown, residual analysis
- **Random Seed**: Reproducible noise generation

#### 📈 Real-Time Metrics

- Side-by-side reconstruction comparison (Baseline vs. SeamAware)
- MDL cost breakdown (parameter, seam, data encoding)
- Residual variance comparison
- Seam detection accuracy (detected vs. ground truth)
- Compression gain percentage

### Dependencies

```bash
pip install streamlit matplotlib numpy scipy pandas
```

Or install the full SeamAware package:

```bash
pip install -e .
```

### Screenshots

The visualizer provides:
- Top row: Baseline (left) vs SeamAware (right) reconstructions
- Middle row: Residual plots showing error distributions
- Bottom row: MDL breakdown bars or roughness curve
- Sidebar: Full parameter control panel

### Use Cases

- **Research**: Explore parameter sensitivity, regime-switching behavior
- **Education**: Demonstrate non-orientable geometry concepts visually
- **Debugging**: Validate seam detection on custom signals
- **Demos**: Show SeamAware's value to collaborators/users

---

## 🔮 Future Applications

Planned additions to this directory:

- **Jupyter Dashboard** (Voila-based): Notebook-style interactive tool
- **Gradio Demo**: Lightweight alternative to Streamlit
- **Batch Analyzer**: Upload CSV, analyze multiple signals
- **Model Comparison**: Benchmark SeamAware against HMMs, change-point detectors

---

**Questions?** Open an issue on [GitHub](https://github.com/MacMayo1993/Seam-Aware-Modeling/issues).
