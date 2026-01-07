# Changelog

All notable changes to SeamAware will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-07

### Added
- **Comprehensive mathematical definitions** in `docs/theory.md`:
  - Explicit state vector normalization formulas
  - Unambiguous SNR definition (amplitude ratio)
  - Complete MDL cost breakdown with coding model
  - Reconciliation of theoretical vs empirical k* threshold
- **Key Definitions table** in README for reproducibility
- **Reproducibility section** with single-command demo validation
- **CITATION.cff** for standardized software citation
- **CHANGELOG.md** for tracking project history

### Changed
- **Repository structure reorganization**:
  - Moved all documentation to `docs/` folder
  - Relocated MASS_SMASH implementation to `examples/`
  - Consolidated tests into `tests/` directory
  - Removed duplicate `setup.py` (standardized on `pyproject.toml`)
- **Updated README**:
  - Fixed installation instructions (removed premature PyPI claim)
  - Removed "coming soon" placeholder badges
  - Updated citation format from arXiv placeholder to @software
  - Clarified "1 bit per seam" refers to orientation cost only
- **Consistent notation**: Changed all ℂᴺ/ℤ₂ references to Sⁿ⁻¹/ℤ₂ ≅ ℝPⁿ⁻¹

### Fixed
- Removed duplicate `README (4).md` file
- Synchronized package version across configuration files (0.2.0)
- Updated all internal documentation cross-references

### Removed
- `setup.py` (superseded by `pyproject.toml`)
- `test_results.txt` (added to .gitignore)
- Placeholder arXiv badge and Binder/Colab "coming soon" badges

## [0.1.0] - 2025-01-06

### Added
- Initial release of SeamAware framework
- Core seam detection via roughness analysis
- Flip atom implementations (sign flip, time reversal, variance scaling)
- MDL-based model selection framework
- k* ≈ 0.721 threshold derivation and validation
- Comprehensive test suite (25/25 passing)
- Examples and Jupyter notebooks
- Full documentation (THEORY.md, ARCHITECTURE.md, etc.)

### Features
- 10-63% MDL reduction over standard baselines
- Support for Fourier, polynomial, and AR baselines
- Monte Carlo validation of k* phase boundary
- CLI demo tool (`python -m seamaware.cli.demo`)

---

[0.2.0]: https://github.com/MacMayo1993/Seam-Aware-Modeling/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MacMayo1993/Seam-Aware-Modeling/releases/tag/v0.1.0
