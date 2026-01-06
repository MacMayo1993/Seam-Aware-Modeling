# SeamAware Examples

This directory contains interactive examples demonstrating the SeamAware framework.

## Quick Start Notebook

**[quick_start.ipynb](quick_start.ipynb)** - Comprehensive introduction to SeamAware

### What's Covered

1. **Synthetic Signal Generation** - Create signals with known seams
2. **Seam Detection** - Roughness-based detection methods
3. **Flip Atom Application** - Apply orientation transformations
4. **MDL Computation** - Measure description length improvements
5. **k* Validation** - Monte Carlo verification of phase boundary
6. **Visualizations** - Publication-quality plots

### Running the Notebook

```bash
# Install Jupyter if needed
pip install jupyter

# Launch notebook
jupyter notebook quick_start.ipynb
```

### Expected Results

- **Seam detection accuracy**: Within 2-5% of true location
- **MDL reduction**: 10-50% for SNR > k* ≈ 0.721
- **Phase transition**: Clear MDL improvement above k*

## Additional Examples (Coming Soon)

- `financial_data.ipynb` - Regime switching in stock prices
- `biomedical_signals.ipynb` - ECG/EEG with polarity reversals
- `energy_consumption.ipynb` - Load forecasting with seasonal flips
- `advanced_flip_atoms.ipynb` - Time reversal, variance scaling, detrending

## Real-World Applications

SeamAware is effective for:

- **Financial time series**: Regime changes, volatility shifts
- **Biomedical signals**: Polarity inversions in ECG/EEG
- **Energy systems**: Seasonal consumption patterns
- **Climate data**: Phase transitions in oscillations
- **Manufacturing**: Equipment mode switches

## Contributing Examples

Have an interesting use case? We welcome contributions!

1. Create a Jupyter notebook in this directory
2. Follow the structure of `quick_start.ipynb`
3. Include:
   - Clear description of the problem
   - Data loading/generation
   - SeamAware analysis
   - Comparison with baselines
   - Visualizations
4. Submit a pull request

## Questions?

- **Documentation**: See main [README.md](../README.md)
- **Theory**: See [THEORY.md](../THEORY.md)
- **Issues**: [GitHub Issues](https://github.com/MacMayo1993/Seam-Aware-Modeling/issues)
